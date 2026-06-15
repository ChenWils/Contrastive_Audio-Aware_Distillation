from transformers import WhisperForConditionalGeneration, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder

from transformers.models.blip_2.modeling_blip_2 import Blip2QFormerModel
from transformers.models.blip_2.configuration_blip_2 import Blip2QFormerConfig
import torch.nn as nn
import torch
import os

class QformerConnector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config.encoder_model_id == "openai/whisper-medium":
            self.config.target_layer_ids = [5, 11, 17, 23]
        elif self.config.encoder_model_id == "openai/whisper-small":
            self.config.target_layer_ids = [2, 5, 8, 11]
        elif self.config.encoder_model_id == "openai/whisper-tiny":
            self.config.target_layer_ids = [0, 1, 2, 3]
        elif self.config.encoder_model_id == "openai/whisper-large-v3":
            self.config.target_layer_ids = [7, 15, 23, 31]
        else:
            raise NotImplementedError(f"model_id {self.config.encoder_model_id} not implemented")


        self.layer_prompts = nn.ParameterList([
            nn.Parameter(torch.randn(1, self.config.prompt_size, self.config.encoder_config.d_model)) for _ in range(len(self.config.target_layer_ids))]
        )

        self.layer_weights = nn.Parameter(torch.zeros(self.config.prompt_size, len(self.config.target_layer_ids), dtype=torch.float))

        if self.config.connector_mode == "qformer_BLIP2" or self.config.connector_mode == "qformer_BLIP2_transcription":
            qformer_config = Blip2QFormerConfig()
            qformer_config.num_hidden_layers = self.config.qformer_num_hidden_layers
            qformer_config.hidden_size = self.config.encoder_config.d_model
            qformer_config.encoder_hidden_size = self.config.encoder_config.d_model
            qformer_config.num_attention_heads = self.config.encoder_config.encoder_attention_heads

            qformer_config.vocab_size = 1 # nop
            qformer_config.attention_dropout = 0.0
            qformer_config.cross_attention_frequency = 2
            qformer_config.use_qformer_text_input = False


            self.qformer = Blip2QFormerModel(qformer_config)
            self.proj = nn.Sequential(
                    nn.LayerNorm(self.config.encoder_config.d_model),
                    nn.Linear(self.config.encoder_config.d_model, self.config.llm_config.hidden_size) # project to llm hidden size
                )
        elif self.config.connector_mode == "qformer_1" or self.config.connector_mode == "qformer_1_transcription":
            # init Qformerblock
            qformer_config = BertConfig()
            qformer_config.num_hidden_layers = self.config.qformer_num_hidden_layers
            qformer_config.num_attention_heads = self.config.encoder_config.encoder_attention_heads
            qformer_config.hidden_size = self.config.encoder_config.d_model
            qformer_config.add_cross_attention = True
            qformer_config.is_decoder = True

            self.qformer = BertEncoder(qformer_config)
            self.proj = nn.Sequential(
                    nn.LayerNorm(self.config.encoder_config.d_model),
                    nn.Linear(self.config.encoder_config.d_model, self.config.llm_config.hidden_size) # project to llm hidden size
                )
        else:
            raise NotImplementedError(f"connector_mode {self.config.connector_mode} not implemented")
        
        if self.config.connector_mode == "qformer_BLIP2_transcription" or self.config.connector_mode == "qformer_1_transcription":
            # if with transcription, need to project llm hidden size to encoder hidden size
            self.transcription_proj = nn.Sequential(
                nn.Linear(self.config.llm_config.hidden_size, self.config.encoder_config.d_model) # project llm hidden size to encoder hidden size
            )


    def forward(self, encoder_hidden_states):
        """
        input: 
            encoder_hidden_states: layerwise hidden states from the encoder
        """
        layer_prompt_outputs = []
        for idx, encoder_hidden_state in enumerate(encoder_hidden_states):
            if idx in self.config.target_layer_ids:
                layer_prompt = self.layer_prompts[self.config.target_layer_ids.index(idx)].expand(encoder_hidden_state.size(0), -1, -1)
                qformer_output = self.qformer(
                    hidden_states=layer_prompt,
                    encoder_hidden_states=encoder_hidden_state,
                )
                layer_prompt_output = qformer_output.last_hidden_state
                layer_prompt_outputs.append(layer_prompt_output)
        
        layer_prompt_outputs = torch.stack(layer_prompt_outputs, dim=0)
        layer_prompt_outputs = layer_prompt_outputs.permute(1, 2, 0, 3)
        self.norm_weights = torch.nn.functional.softmax(self.layer_weights, dim=-1).unsqueeze(-1)
        output = (layer_prompt_outputs * self.norm_weights).sum(dim=2) # (b, prompt_size, d_llm)
        output = self.proj(output)
        
        return output
        


class WhisperPerception(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.whisper = WhisperForConditionalGeneration.from_pretrained(
            self.config.encoder_model_id, cache_dir=os.getenv("HF_HOME")
        )

        self.connector = QformerConnector(config)


    def forward(self, input_features, attention_mask=None, transcription_embeddings_list=None, **kwargs):
        bs = input_features.size(0)

        speech_features = self.forward_whisper(input_features=input_features, transcription_embeddings_list=transcription_embeddings_list)
        speech_feature_lengths = [self.config.prompt_size] * speech_features.size(0) # (b, )
        
        
        return speech_features, speech_feature_lengths


    def forward_whisper(self, input_features, attention_mask=None, transcription_embeddings_list=None, **kwargs):
        """
        2024.07.07 @kehan
        copy from previous implementation for qformer_1
        
        """
        bs = input_features.size(0)
        
        expected_seq_length = self.whisper.model.encoder.config.max_source_positions * self.whisper.model.encoder.conv1.stride[0] * self.whisper.model.encoder.conv2.stride[0]

        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )
        

        inputs_embeds = nn.functional.gelu(self.whisper.model.encoder.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.whisper.model.encoder.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.whisper.model.encoder.embed_positions.weight[:self.whisper.model.encoder.config.max_source_positions, :] # @kehan

        hidden_states = inputs_embeds + embed_pos
        # hidden_states = nn.functional.dropout(hidden_states, p=self.whisper.model.encoder.dropout, training=self.training)
        features_length = hidden_states.size(1)

        if (self.config.connector_mode == "qformer_1"
            or self.config.connector_mode == "qformer_BLIP2_transcription" 
            or self.config.connector_mode == "qformer_1_transcription"
            or self.config.connector_mode == "qformer_BLIP2"):
            layer_prompt_outputs = []
            for idx, encoder_layer in enumerate(self.whisper.model.encoder.layers):
                
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask=None,
                    layer_head_mask=None,
                    output_attentions=None,
                )
                hidden_states = layer_outputs[0]

                if idx in self.connector.config.target_layer_ids:
                    # use different prompt for different layers
                    layer_prompt = self.connector.layer_prompts[self.connector.config.target_layer_ids.index(idx)].expand(bs, -1, -1)

                    if self.config.connector_mode == "qformer_BLIP2_transcription" or self.config.connector_mode == "qformer_1_transcription":
                        layer_prompt, attention_mask = self._prepare_layer_prompt_with_transcription(layer_prompt, transcription_embeddings_list)
                        attention_mask = None
                    else:
                        attention_mask = None
                    
                    # Qformer is a BERTEncoder(but set to decoder) from huggingface Transformers
                    qformer_output = self.connector.qformer(
                        layer_prompt,
                        attention_mask=attention_mask,
                        encoder_hidden_states=hidden_states,
                    )
                    
                    layer_prompt_output = qformer_output.last_hidden_state[:, :self.config.prompt_size, :] # (b, prompt_size, d_model)
                    layer_prompt_outputs.append(layer_prompt_output) # list of (b, prompt_size, d_model)

            layer_prompt_outputs = torch.stack(layer_prompt_outputs, dim=0) # (layer, b, prompt_size, d_model)
            layer_prompt_outputs = layer_prompt_outputs.permute(1, 2, 0, 3) # (b, prompt_size, layer, d_model)
            
            self.norm_weights = torch.nn.functional.softmax(self.connector.layer_weights, dim=-1).unsqueeze(-1) # (prompt_size, layer, 1)
            prompt_output = (layer_prompt_outputs * self.norm_weights).sum(dim=2) # (b, prompt_size, d_model)
            assert prompt_output.size(1) == self.config.prompt_size, prompt_output.size()
            prompt_output = self.connector.proj(prompt_output)
            
            return prompt_output

        else:
            raise NotImplementedError(f"mode {self.mode} not implemented")
        
    
    def _prepare_layer_prompt_with_transcription(self, layer_prompt, transcription_embeddings_list):
        bs = layer_prompt.size(0)
        # layer_prompt: (b, prompt_size, d_model)
        max_len = max([transcription_embeddings.size(0) for transcription_embeddings in transcription_embeddings_list])

        query_att_mask = torch.ones([bs, self.config.prompt_size], device=layer_prompt.device)
        text_att_mask = torch.zeros([bs, max_len], device=layer_prompt.device)

        padded_embeddings = []
        for i, transcription_embeddings in enumerate(transcription_embeddings_list):
            padded_tensor = torch.zeros(max_len, transcription_embeddings.size(1), device=layer_prompt.device)
            padded_tensor[:transcription_embeddings.size(0), :] = transcription_embeddings
            padded_embeddings.append(padded_tensor)

            text_att_mask[i, :transcription_embeddings.size(0)] = 1

        
        padded_embeddings = torch.stack(padded_embeddings, dim=0)
        transcription_embeddings = self.connector.transcription_proj(padded_embeddings)
        
        concated_layer_prompt = torch.cat([layer_prompt, transcription_embeddings], dim=1)

        
        attention_mask = torch.cat([query_att_mask, text_att_mask], dim=1)

        return concated_layer_prompt, attention_mask
    

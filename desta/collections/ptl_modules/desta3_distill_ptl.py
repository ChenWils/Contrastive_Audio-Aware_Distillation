from desta.collections.ptl_modules.desta3_ptl import DeSTA3PTLModule
import torch
from typing import Optional, Tuple, Union
import logging
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import DynamicCache, Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from desta.collections.desta3.data.distill_dataset import DistillDataset, BaseAudioTextDataset
from desta.collections.desta3.data.sampler import DistributedLengthBasedBatchSampler, LengthBasedBatchSampler
from torch.utils.data import DataLoader
from torch.nn import functional as F
import logging


class DistillationPTLModule(DeSTA3PTLModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.add_noise = self.cfg.model.add_noise
        self.kd_alpha = self.cfg.model.kd_alpha
    
    def forward(self, batch):
        
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        batch_features = batch["batch_features"]
        batch_transcription_ids = batch["batch_transcription_ids"]
        batch_start_positions = batch["batch_start_positions"]
        labels = batch.get("labels", None)

        batch_speech_features, batch_speech_feature_lengths = self.model.perception(batch_features)


        audio_input_embeds = self.model._prepare_inputs_for_llm(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            batch_speech_features=batch_speech_features, 
            batch_speech_feature_lengths=batch_speech_feature_lengths, 
            batch_transcription_ids=batch_transcription_ids, 
            batch_start_positions=batch_start_positions
        )

        if self.training:
            audio_start_answer_positions = batch["audio_start_answer_positions"]
            seed_start_answer_positions = batch["seed_start_answer_positions"]

            seed_input_ids = batch["seed_input_ids"]
            seed_input_embeds = self.model.llm_model.model.embed_tokens(seed_input_ids)
            
            if self.add_noise:
                audio_input_embeds, seed_input_embeds = self._add_noise(audio_input_embeds, seed_input_embeds, audio_start_answer_positions, seed_start_answer_positions)


        audio_output = self.model.llm_model(
            inputs_embeds=audio_input_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        self.log("train/ntp_loss", audio_output.loss.item(), sync_dist=True, batch_size=batch["input_ids"].size(0))


        if self.training:
            seed_attention_mask = batch["seed_attention_mask"]
            with torch.no_grad():
                seed_output = self.model.llm_model(
                    inputs_embeds=seed_input_embeds,
                    attention_mask=seed_attention_mask,
                    output_hidden_states=True,
                )

            kd_loss = self.compute_distillation_loss(
                audio_output.hidden_states, seed_output.hidden_states, 
                audio_start_answer_positions, seed_start_answer_positions
            )

            

            audio_output.loss = (1 - self.kd_alpha) * audio_output.loss + self.kd_alpha * kd_loss
            self.log("train/kd_loss", kd_loss.item(), sync_dist=True, batch_size=batch["input_ids"].size(0))
        return audio_output
    
    def _add_noise(self, audio_input_embeds, seed_input_embeds, audio_start_answer_positions, seed_start_answer_positions):
        # Create new copies of the input tensors

        # Add same noise only to the portions after answer positions
        for i in range(len(audio_start_answer_positions)):
            audio_start = audio_start_answer_positions[i].item()
            seed_start = seed_start_answer_positions[i].item()

            audio_answer_embeds = audio_input_embeds[i, audio_start:]
            seed_answer_embeds = seed_input_embeds[i, seed_start:]
            
            max_length = min(audio_answer_embeds.size(0), seed_answer_embeds.size(0))
            noise = torch.randn_like(audio_answer_embeds[:max_length])
            # replace the answer embeddings with noise
            # USE noise
            audio_answer_embeds[:max_length] = noise
            seed_answer_embeds[:max_length] = noise

            audio_input_embeds[i, audio_start:audio_start+max_length] = audio_answer_embeds[:max_length]
            seed_input_embeds[i, seed_start:seed_start+max_length] = seed_answer_embeds[:max_length]
           

        return audio_input_embeds, seed_input_embeds

    def compute_distillation_loss(self, audio_hidden_states, seed_hidden_states, 
                             audio_start_answer_positions, seed_start_answer_positions,
                             temperature=2.0):
        
        total_loss = 0.0
        for layer_audio_hidden_states, layer_seed_hidden_states in zip(audio_hidden_states, seed_hidden_states):
            # layer wise
            for i, (audio_hidden_state, seed_hidden_state) in enumerate(zip(layer_audio_hidden_states, layer_seed_hidden_states)):
                # in batch

                audio_start = audio_start_answer_positions[i].item()
                seed_start = seed_start_answer_positions[i].item()

                
                # Only compute the loss after answer positions
                audio_hidden_state = audio_hidden_state[audio_start:]
                seed_hidden_state = seed_hidden_state[seed_start:]

                # truncate the length of audio_hidden_state and seed_hidden_state to the same length
                max_length = min(audio_hidden_state.size(0), seed_hidden_state.size(0))
                audio_hidden_state = audio_hidden_state[:max_length]
                seed_hidden_state = seed_hidden_state[:max_length]

                audio_probs = F.log_softmax(audio_hidden_state / temperature, dim=-1)
                seed_probs = F.softmax(seed_hidden_state / temperature, dim=-1)

                loss_kd = F.kl_div(
                    audio_probs,
                    seed_probs,
                    reduction='batchmean'
                ) * (temperature ** 2)

                total_loss += loss_kd
            
        return total_loss / len(audio_hidden_states)
            
    
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        loss = 0
        perplexity = 0
        predictions = []

        outputs = self(batch)
        loss = outputs.loss
        perplexity = torch.exp(loss)
        loss = torch.tensor(0.0)
        perplexity = torch.tensor(0.0)

        predictions = self.predict_step(batch, batch_idx)

        batch_size = batch["input_ids"].size(0)
        
        self.log("val_loss", loss.item(), sync_dist=True, batch_size=batch_size)
        self.log("val_ppl", perplexity.item(), sync_dist=True, batch_size=batch_size)


        return {"val_loss": loss, "val_ppl": perplexity, "predictions": predictions}


    # Dataloader
    def _build_dataloader(self, data_cfg):
        return_seed = data_cfg.get("return_seed", False)
        if return_seed:
            dataset = DistillDataset(
                cfg=self.cfg,
                data_cfg=data_cfg,
                tokenizer=self.tokenizer,
                processor=self.processor,
                return_seed=True
            )
        else:
            dataset = BaseAudioTextDataset(
                cfg=self.cfg,
                data_cfg=data_cfg,
                tokenizer=self.tokenizer,
                processor=self.processor
            )
        logging.info(dataset[0])

        if data_cfg.get("batch_sampler", None) == "length_based":
            if self.trainer.world_size > 1:
                batch_sampler = DistributedLengthBasedBatchSampler(
                    dataset,
                    max_batch_length=data_cfg.max_batch_length,
                    num_replicas=self.trainer.world_size,
                    rank=self.trainer.global_rank,
                    shuffle=data_cfg.shuffle,
                )
            else:
                batch_sampler = LengthBasedBatchSampler(
                    dataset,
                    max_batch_length=data_cfg.max_batch_length,
                    shuffle=data_cfg.shuffle,
                    drop_last=data_cfg.drop_last,
                )
            dataloader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=data_cfg.num_workers,
                pin_memory=data_cfg.pin_memory,
                collate_fn=dataset.collate_fn,
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=data_cfg.batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=data_cfg.shuffle,
                num_workers=data_cfg.num_workers,
                pin_memory=data_cfg.pin_memory,
                drop_last=data_cfg.drop_last,
            )
        return dataloader
    
    def llm_forward(
        self,
        audio_input_embeds,
        audio_attention_mask,
        audio_start_answer_positions,
        seed_input_embeds,
        seed_attention_mask,
        seed_start_answer_positions,

        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.model.llm_model.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.llm_model.model.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.model.llm_model.model.config.use_cache
        return_dict = return_dict if return_dict is not None else self.model.llm_model.model.config.use_return_dict

        # Setup for cache and positions
        if use_cache and past_key_values is None:
            audio_past_key_values = DynamicCache()
            seed_past_key_values = DynamicCache()
    

        # Initialize two separate paths for audio and seed
        audio_hidden_states = audio_input_embeds
        seed_hidden_states = seed_input_embeds

        # Setting up audio path cache position and position IDs
        audio_past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        audio_cache_position = torch.arange(
            audio_past_seen_tokens, audio_past_seen_tokens + audio_hidden_states.shape[1], 
            device=audio_hidden_states.device
        )
        audio_position_ids = audio_cache_position.unsqueeze(0)

        # Setting up seed path cache position and position IDs  
        seed_past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        seed_cache_position = torch.arange(
            seed_past_seen_tokens, seed_past_seen_tokens + seed_hidden_states.shape[1], 
            device=seed_hidden_states.device
        )
        seed_position_ids = seed_cache_position.unsqueeze(0)

        # Create causal masks for both paths
        audio_causal_mask = self.model.llm_model.model._update_causal_mask(
            audio_attention_mask, audio_hidden_states, audio_cache_position, past_key_values, output_attentions
        )
        
        seed_causal_mask = self.model.llm_model.model._update_causal_mask(
            seed_attention_mask, seed_hidden_states, seed_cache_position, past_key_values, output_attentions
        )

        # Create position embeddings for both paths
        audio_position_embeddings = self.model.llm_model.model.rotary_emb(audio_hidden_states, audio_position_ids)
        seed_position_embeddings = self.model.llm_model.model.rotary_emb(seed_hidden_states, seed_position_ids)

        # Setup for tracking hidden states
        audio_all_hidden_states = () if output_hidden_states else None
        seed_all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # Process through decoder layers
        loss = 0
        for decoder_layer in self.model.llm_model.model.layers[: self.model.llm_model.model.config.num_hidden_layers]:
            if output_hidden_states:
                audio_all_hidden_states += (audio_hidden_states,)
                seed_all_hidden_states += (seed_hidden_states,)

            # Process audio path
            audio_layer_outputs = decoder_layer(
                audio_hidden_states,
                attention_mask=audio_causal_mask,
                position_ids=audio_position_ids,
                past_key_value=audio_past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=audio_cache_position,
                position_embeddings=audio_position_embeddings,
                **flash_attn_kwargs,
            )

            seed_layer_outputs = decoder_layer(
                seed_hidden_states,
                attention_mask=seed_causal_mask,
                position_ids=seed_position_ids,
                past_key_value=seed_past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=seed_cache_position,
                position_embeddings=seed_position_embeddings,
                **flash_attn_kwargs,
            )

            # Update hidden states
            audio_hidden_states = audio_layer_outputs[0]
            seed_hidden_states = seed_layer_outputs[0]
            
            # Calculate MSE loss between the hidden states at this layer,
            # but only starting from the answer positions
                        
            layer_loss = self.calculate_distillation_loss(
                audio_hidden_states, seed_hidden_states, audio_start_answer_positions, seed_start_answer_positions
            )
        
                
            loss += layer_loss

            # Collect attention if needed
            if output_attentions:
                all_self_attns += (audio_layer_outputs[1],)  # Using audio attentions for output

        
        # Apply final norm to both paths
        # audio_hidden_states = self.model.llm_model.model.norm(audio_hidden_states)
        # seed_hidden_states = self.model.llm_model.model.norm(seed_hidden_states)

        # # Add final hidden states
        # if output_hidden_states:
        #     audio_all_hidden_states += (audio_hidden_states,)
        #     seed_all_hidden_states += (seed_hidden_states,)

        # # Final MSE between normalized outputs, using answer positions
        # final_loss = self.calculate_distillation_loss(
        #     audio_hidden_states, seed_hidden_states, audio_start_answer_positions, seed_start_answer_positions
        # )
        # loss += final_loss
        
        
        # # Create output using audio path as primary
        output = BaseModelOutputWithPast(
            last_hidden_state=audio_hidden_states,
            past_key_values=audio_past_key_values if use_cache else None,
            hidden_states=audio_all_hidden_states,
            attentions=all_self_attns,
        )
        output.loss = loss
        return output
        

        
        # return output if return_dict else output.to_tuple()
    
    def calculate_distillation_loss(self, audio_hidden_states, seed_hidden_states, audio_start_answer_positions, seed_start_answer_positions):
        batch_size = audio_hidden_states.shape[0]
        loss = 0
        for i in range(batch_size):
            # Get starting positions for this batch item
            audio_start = audio_start_answer_positions[i]
            seed_start = seed_start_answer_positions[i]
            
            # Extract hidden states from the answer positions onwards
            audio_answer_hidden = audio_hidden_states[i, audio_start:, :]
            seed_answer_hidden = seed_hidden_states[i, seed_start:, :]
            
            
            min_length = min(audio_answer_hidden.shape[0], seed_answer_hidden.shape[0])

            # Compare only the overlapping parts
            batch_loss = self.criterion(
                audio_answer_hidden[:min_length], 
                seed_answer_hidden[:min_length]
            )
            loss += batch_loss
        loss /= batch_size
        return loss
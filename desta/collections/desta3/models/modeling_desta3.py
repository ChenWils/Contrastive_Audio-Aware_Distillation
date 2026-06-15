
from transformers import PretrainedConfig, PreTrainedModel, AutoModelForCausalLM, AutoConfig
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoProcessor
from desta.collections.desta3.modules.whisper_perception import WhisperPerception
import os
import torch
from collections import OrderedDict
import logging
from desta.collections.utils.audio import AudioSegment
from desta.collections.desta3.data.simple_dataset import _prepare_audio_context_and_start_positions


class DeSTA3Config(PretrainedConfig):
    model_type = "desta3"

    def __init__(self, llm_model_id="kehanlu/llama-3.2-8B-Instruct", encoder_model_id="openai/whisper-large-v3", connector_mode="qformer_1", qformer_num_hidden_layers=2, prompt_size=64, first_n_layers=-1, **kwargs):
        super().__init__(**kwargs)

        self.llm_model_id = llm_model_id
        self.encoder_model_id = encoder_model_id
        self.connector_mode = connector_mode
        self.qformer_num_hidden_layers = qformer_num_hidden_layers
        self.prompt_size = prompt_size

        self.llm_config = AutoConfig.from_pretrained(self.llm_model_id)
        self.encoder_config = AutoConfig.from_pretrained(self.encoder_model_id)

        self.first_n_layers = first_n_layers if first_n_layers is not None else -1

class DeSTA3Model(PreTrainedModel):
    config_class = DeSTA3Config

    def __init__(self, config, cache_dir=None, token=None, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config

        token = token if token else os.getenv("HF_TOKEN")
        cache_dir = cache_dir if cache_dir else os.getenv("HF_HOME")

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.config.llm_model_id,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            token=token,
        )
        if self.config.first_n_layers > 0:
            logging.warn(f"Truncating LLM model to first {self.config.first_n_layers} layers.")
            self.llm_model.model.layers = self.llm_model.model.layers[: self.config.first_n_layers]
        self.perception = WhisperPerception(self.config)

        self.configure_trainable_parameters()

    def forward(self, input_ids,
                attention_mask, 
                batch_features, 
                batch_transcription_ids,
                batch_start_positions,
                labels=None,
                **kwargs):

        
        # batch_speech_features, batch_speech_feature_lengths = self.perception(batch_features)

        inputs_embeds = self._prepare_inputs_for_llm(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            batch_features=batch_features,
            batch_transcription_ids=batch_transcription_ids, 
            batch_start_positions=batch_start_positions
        )


        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs 

    def _prepare_inputs_for_llm(self, 
                               input_ids,
                               attention_mask,
                               batch_features,
                               batch_transcription_ids,
                               batch_start_positions
        ):

        N_speech = len(batch_start_positions)
        
        # Get list of transcription embeddings
        transcription_embeddings_list = []
        with torch.no_grad():
            for speech_batch_idx in range(N_speech):
                transcription_embeddings = self.llm_model.model.embed_tokens(
                    batch_transcription_ids[speech_batch_idx].squeeze(0)
                ) # (length, dim)
                transcription_embeddings_list.append(transcription_embeddings)

        # Forward speech encoder and connector
        batch_speech_features, batch_speech_feature_lengths = self.perception(input_features=batch_features, transcription_embeddings_list=transcription_embeddings_list)

        assert len(batch_start_positions) == len(batch_transcription_ids) == batch_speech_features.size(0) == len(batch_speech_feature_lengths), "batch_start_positions, batch_transcription_ids, speech_features, speech_feature_lengths must have the same length."


        # [---- Other text embeddings ----][---- placeholder embeddings ----][---- Other text embeddings ----]
        inputs_embeds = self.llm_model.model.embed_tokens(input_ids)
        
        
        for speech_batch_idx in range(N_speech):
            start_position = batch_start_positions[speech_batch_idx] # tuple (text_idx, speech_start_position)
            text_batch_idx = start_position[0]
            speech_start_position = start_position[1]

            # get the speech features   
            speech_features = batch_speech_features[speech_batch_idx]
            speech_feature_length = batch_speech_feature_lengths[speech_batch_idx]

            # get transcription embeddings
            transcription_embeddings = transcription_embeddings_list[speech_batch_idx] # (length, dim)

            # # concat the speech features and transcription embeddings
            speech_embeddings = torch.cat([speech_features, transcription_embeddings], dim=0)

            assert speech_embeddings.size(0) == (speech_feature_length + transcription_embeddings.size(0))

            # # replace the input_embeds with the speech features
            # # [---- Other text embeddings ----][---- speech features + transcription embeddings ----][---- Other text embeddings ----]
            target_slice = slice(speech_start_position, speech_start_position + speech_embeddings.size(0))
            inputs_embeds[text_batch_idx, target_slice] = speech_embeddings
            


            if input_ids[text_batch_idx, speech_start_position-1] == 128096:
                logging.warning(input_ids[text_batch_idx, speech_start_position-1: speech_start_position + speech_embeddings.size(0)+1])

            # # clean GPU memory
            del speech_features, speech_feature_length, transcription_embeddings, speech_embeddings

        return inputs_embeds
        
    def state_dict(self):
        trainable_state_dict = OrderedDict()
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_state_dict[name] = param.data.clone().detach()
        return trainable_state_dict

    def _generate_step(self, inputs, pad_token_id, generation_kwargs):
        input_ids = inputs["context_input_ids"] # only context inputs
        attention_mask = inputs["context_attention_mask"] # only context attention mask
        batch_start_positions = inputs["context_batch_start_positions"]

        batch_transcription_ids = inputs["batch_transcription_ids"]
        # batch_speech_features, batch_speech_feature_lengths = self.perception()

        # get the generated text
        inputs_embeds = self._prepare_inputs_for_llm(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            batch_features=inputs["batch_features"],
            batch_transcription_ids=batch_transcription_ids, 
            batch_start_positions=batch_start_positions
        )
        
        generated_ids = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            **generation_kwargs
        )

        return generated_ids

    # def _prepare_weak_inputs_with_zero_audio(self, input_ids, strong_inputs_embeds,
    #                                          batch_transcription_ids, batch_start_positions,
    #                                          batch_features):
    #     """
    #     Prepare weak model inputs with same length as strong model,
    #     but with audio features replaced by zeros (keeping transcription).

    #     This ensures both models have identical sequence lengths for proper contrastive decoding.

    #     Structure:
    #     - Strong: [text, AUDIO_FEATURES, transcription, text]
    #     - Weak:   [text, ZERO_PADDING,   transcription, text]  ← Same length!
    #     """
    #     N_speech = len(batch_start_positions)

    #     # Get transcription embeddings
    #     transcription_embeddings_list = []
    #     with torch.no_grad():
    #         for speech_batch_idx in range(N_speech):
    #             transcription_embeddings = self.llm_model.model.embed_tokens(
    #                 batch_transcription_ids[speech_batch_idx].squeeze(0)
    #             )
    #             transcription_embeddings_list.append(transcription_embeddings)

    #     # Forward speech encoder to get audio feature lengths
    #     batch_speech_features, batch_speech_feature_lengths = self.perception(
    #         input_features=batch_features,
    #         transcription_embeddings_list=transcription_embeddings_list
    #     )

    #     # Start with text embeddings (same as strong model initial state)
    #     weak_inputs_embeds = self.llm_model.model.embed_tokens(input_ids)

    #     # For each audio position, insert [ZERO_PADDING + transcription]
    #     for speech_batch_idx in range(N_speech):
    #         start_position = batch_start_positions[speech_batch_idx]
    #         text_batch_idx = start_position[0]
    #         speech_start_position = start_position[1]

    #         # Get audio feature length (e.g., 64) and transcription
    #         speech_feature_length = batch_speech_feature_lengths[speech_batch_idx]
    #         transcription_embeddings = transcription_embeddings_list[speech_batch_idx]

    #         # Create zero padding for audio features
    #         hidden_dim = weak_inputs_embeds.size(-1)
    #         zero_audio_features = torch.zeros(
    #             speech_feature_length,
    #             hidden_dim,
    #             device=weak_inputs_embeds.device,
    #             dtype=weak_inputs_embeds.dtype
    #         )

    #         # Concatenate [ZERO_PAD + transcription]
    #         weak_audio_embeddings = torch.cat([zero_audio_features, transcription_embeddings], dim=0)

    #         # Replace the slice (same position as strong model)
    #         target_slice = slice(speech_start_position, speech_start_position + weak_audio_embeddings.size(0))
    #         weak_inputs_embeds[text_batch_idx, target_slice] = weak_audio_embeddings

    #     return weak_inputs_embeds

    def _generate_step_contrastive(self, inputs, pad_token_id, generation_kwargs, contrastive_alpha=1.0):
        """
        Contrastive decoding: generates tokens by contrasting strong model (with audio) vs weak model (text-only)
        Formula: logits_final = logits_strong - alpha * logits_weak

        Args:
            inputs: dict with batch_features, batch_transcription_ids, context_input_ids, etc.
            pad_token_id: padding token id
            generation_kwargs: generation parameters (max_new_tokens, temperature, top_p, etc.)
            contrastive_alpha: weight for contrastive decoding (typically 0.1-0.5)
        """
        input_ids = inputs["context_input_ids"]
        attention_mask = inputs["context_attention_mask"]
        batch_start_positions = inputs["context_batch_start_positions"]
        batch_transcription_ids = inputs["batch_transcription_ids"]

        # Prepare strong model inputs (with audio features)
        strong_inputs_embeds = self._prepare_inputs_for_llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            batch_features=inputs["batch_features"],
            batch_transcription_ids=batch_transcription_ids,
            batch_start_positions=batch_start_positions
        )

        # Prepare weak model inputs (text-only, no audio - simple and fast)
        # Note: This creates a sequence length mismatch with strong model
        # (e.g., strong: 75 tokens, weak: 5 tokens), but works well with proper alpha tuning
        weak_inputs_embeds = self.llm_model.model.embed_tokens(input_ids)

        # Extract generation parameters
        max_new_tokens = generation_kwargs.get('max_new_tokens', 50)
        temperature = generation_kwargs.get('temperature', 1.0)
        top_p = generation_kwargs.get('top_p', 1.0)
        top_k = generation_kwargs.get('top_k', 50)
        do_sample = generation_kwargs.get('do_sample', True)

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Initialize generation
        generated_tokens = []
        current_attention_mask = attention_mask.clone()

        # KV caches for both models
        strong_past_key_values = None
        weak_past_key_values = None

        # Current inputs
        strong_current_embeds = strong_inputs_embeds
        weak_current_embeds = weak_inputs_embeds

        # Per-sequence finish tracking for batch generation
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_new_tokens):
            # Forward pass for strong model (with audio)
            strong_outputs = self.llm_model(
                inputs_embeds=strong_current_embeds,
                attention_mask=current_attention_mask,
                past_key_values=strong_past_key_values,
                use_cache=True,
                return_dict=True
            )
            strong_logits = strong_outputs.logits[:, -1, :]  # (batch_size, vocab_size)
            strong_past_key_values = strong_outputs.past_key_values

            # Forward pass for weak model (text-only)
            weak_outputs = self.llm_model(
                inputs_embeds=weak_current_embeds,
                attention_mask=current_attention_mask,
                past_key_values=weak_past_key_values,
                use_cache=True,
                return_dict=True
            )
            weak_logits = weak_outputs.logits[:, -1, :]  # (batch_size, vocab_size)
            weak_past_key_values = weak_outputs.past_key_values

            # Apply contrastive decoding
            contrastive_logits = strong_logits - contrastive_alpha * weak_logits

            # Apply temperature
            if temperature != 1.0:
                contrastive_logits = contrastive_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = contrastive_logits < torch.topk(contrastive_logits, top_k)[0][..., -1, None]
                contrastive_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(contrastive_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                contrastive_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = torch.softmax(contrastive_logits, dim=-1)
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

            # For sequences that were ALREADY finished, replace their token with pad
            # This happens BEFORE we check if they just finished (so EOS is preserved)
            next_token_masked = next_token.masked_fill(is_finished.unsqueeze(-1), pad_token_id)

            # Append the masked token (includes EOS for newly finished, PAD for already finished)
            generated_tokens.append(next_token_masked)

            # Update finish tracking: check if any sequences generated EOS this step
            # Use the ORIGINAL next_token (before masking) to detect newly finished sequences
            newly_finished = (next_token.squeeze(-1) == pad_token_id)
            is_finished = is_finished | newly_finished

            # Early stopping: if all sequences finished, break
            if is_finished.all():
                break

            # Prepare inputs for next step
            # Use the masked token (pad for finished sequences) for consistency
            next_token_embeds = self.llm_model.model.embed_tokens(next_token_masked)
            strong_current_embeds = next_token_embeds
            weak_current_embeds = next_token_embeds

            # Update attention mask
            current_attention_mask = torch.cat([
                current_attention_mask,
                torch.ones((batch_size, 1), device=device, dtype=current_attention_mask.dtype)
            ], dim=1)

        # Concatenate all generated tokens
        generated_tokens = torch.cat(generated_tokens, dim=1)

        # Return only generated tokens (not including input_ids)
        # This matches the behavior expected by predict_step
        return generated_tokens


    def configure_trainable_parameters(self):

        known_parameters = []
        # freeze LLM parameters
        for name, params in self.llm_model.named_parameters():
            params.requires_grad = False
            known_parameters.append(f"llm_model.{name}")

        # freeze encoder parameters
        for name, params in self.perception.whisper.named_parameters():
            params.requires_grad = False
            known_parameters.append(f"perception.whisper.{name}")


        # make other parameters trainable
        self.trainable_parameter_names = []
        trainable_parameters = []
        for name, params in self.named_parameters():
            if name not in known_parameters:
                params.requires_grad = True
                self.trainable_parameter_names.append(name)
                trainable_parameters.append(params)



    # Easy of generation
    def _setup_hf_generation(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_id, cache_dir=os.getenv("HF_HOME"))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.audio_locator = "<|AUDIO|>"
        self.placeholder_token = "<|reserved_special_token_87|>"
        
        self.tokenizer.add_tokens([self.audio_locator])
        self.processor = AutoProcessor.from_pretrained(self.config.encoder_model_id, cache_dir=os.getenv("HF_HOME"))

        # VAD
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        (self.get_speech_timestamps, _, _, _, _) = utils



    def generate(self, messages, generation_kwargs=None, return_audios=False, use_contrastive_decoding=False, contrastive_alpha=1.0):
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful voice assistance.",
            },
            {
                "role": "user",
                "content": "Hello! this is my audio <|AUDIO|>. Help me transcribe." # "<|AUDIO|>" is a special token to mark the audio position
                "audios": [
                    "audio": "/path/to/filepath", # path to 
                    "transcription": None # None or provided if you have the decoded result
                ]
            },
        ]
        """
        if not hasattr(self, "tokenizer"):
            self._setup_hf_generation()

        if isinstance(messages, list):
            if isinstance(messages[0], dict):
                messages_list = [messages]
            else: 
                messages_list = messages
        else:
            raise ValueError("messages should be a list of dictionaries or a list of lists.")

        all_audios = []
        all_transcriptions = []
        for messages in messages_list:
            for message in messages:
                content = message["content"]
                audios = message.get("audios", [])
                assert len(audios) == content.count(self.audio_locator), "audio count does not match (<|AUDIO|>) count"

                for audio in audios:
                    all_audios.append(audio["audio"])
                    all_transcriptions.append(audio.get("transcription"))

        if len(all_audios) > 0:
            """
            If audios are provided, run:
            1. get features and transcription
            2. prepare LLM inputs
            3. run generation
            """

            batch_features = []
            asr_features = []
            asr_indices = []
            for i, (audio, trans) in enumerate(zip(all_audios, all_transcriptions)):
                if not os.path.exists(audio):
                    raise ValueError(f"Audio file {audio} does not exist.")

                feature = AudioSegment.from_file(
                    audio,
                    target_sr=16000,
                    channel_selector="average"
                ).samples

                batch_features.append(feature)

                is_speech = self.get_speech_timestamps(feature, self.vad_model)
                if is_speech and trans is None:
                    asr_features.append(feature)
                    asr_indices.append(i)
                if not is_speech:
                    all_transcriptions[i] = " "
            
            batch_features = self.processor(batch_features, sampling_rate=16000, return_tensors="pt").input_features
            batch_features = batch_features.to(self.device)
            audio_size_list = [self.config.prompt_size] * len(batch_features)


            # RUN ASR
            if asr_features:
                asr_features = self.processor(asr_features, sampling_rate=16000, return_tensors="pt").input_features
                asr_features = asr_features.to(self.device)

                transcriptions = self.perception.whisper.generate(
                    input_features=asr_features,
                    attention_mask=None,
                    max_new_tokens=300
                )
                transcriptions = self.processor.batch_decode(
                    transcriptions,
                    skip_special_tokens=True,
                )
            else:
                # no audio needs ASR result
                transcriptions = []

            
            for i, transcription in zip(asr_indices, transcriptions):
                all_transcriptions[i] = transcription.strip()
                    
            transcription_size_list = [
                len(self.tokenizer.tokenize(text, add_special_tokens=False)) for text in all_transcriptions
            ]


            audio_context_list = []
            start_positions_list = []
            for messages in messages_list:
                audio_context = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                audio_context = audio_context.replace(self.audio_locator, f"<start_audio>{self.audio_locator}<end_audio>")

                audio_context, start_positions = self._prepare_audio_context_and_start_positions(
                        token_list=self.tokenizer.tokenize(audio_context), 
                        audio_locator=self.audio_locator,
                        audio_size_list=audio_size_list,
                        transcription_size_list=transcription_size_list,
                        placeholder_token=self.placeholder_token
                    )


                audio_context = self.tokenizer.convert_tokens_to_string(audio_context)
                audio_context_list.append(audio_context)

                start_positions_list.append(start_positions)

            audio_context_inputs = self.tokenizer(
                audio_context,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_length=True,
                add_special_tokens=False,
            )

            audio_context_batch_start_positions = []
            for i in range(audio_context_inputs["length"].size(0)):
                total_length = audio_context_inputs["length"][i]
                pad_length = total_length - audio_context_inputs["attention_mask"][i].sum()

                for start_position in start_positions_list[i]:
                    audio_context_batch_start_positions.append((i, start_position + pad_length))

            batch_transcription_ids = []
            for transcription in all_transcriptions:
                batch_transcription_ids.append(
                    self.tokenizer.encode(transcription, add_special_tokens=False, return_tensors="pt").long().to(self.device)
                )



            inputs = {
                "batch_features": batch_features,
                "batch_transcription_ids": batch_transcription_ids,

                "context_input_ids": audio_context_inputs["input_ids"],
                "context_attention_mask": audio_context_inputs['attention_mask'],
                "context_batch_start_positions": audio_context_batch_start_positions,
            }
            inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            if use_contrastive_decoding:
                generated_ids = self._generate_step_contrastive(
                    inputs,
                    pad_token_id=self.tokenizer.pad_token_id,
                    generation_kwargs=generation_kwargs,
                    contrastive_alpha=contrastive_alpha
                )
            else:
                generated_ids = self._generate_step(
                    inputs,
                    pad_token_id=self.tokenizer.pad_token_id,
                    generation_kwargs=generation_kwargs
                )

            if return_audios:
                return generated_ids, list(zip(all_audios, all_transcriptions))
            
            return generated_ids

        else:
            """
            if no audios are provided, it's identical to the original LLM generation
            """

            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(inputs, return_tensors="pt").to(self.device)

            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            generated_ids = self.llm_model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                eos_token_id=terminators,
                **generation_kwargs
            )
            generated_ids = generated_ids[:, inputs["input_ids"].shape[-1]:]

            if return_audios:
                return generated_ids, []
            return generated_ids
        

        


        

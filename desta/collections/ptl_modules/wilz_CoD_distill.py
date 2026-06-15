from desta.collections.ptl_modules.desta3_ptl import DeSTA3PTLModule
from desta.collections.desta3.models.modeling_desta3 import DeSTA3Model, DeSTA3Config
import torch
from typing import Optional, Tuple, Union
import logging
from peft import LoraConfig, get_peft_model
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import DynamicCache, Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from desta.collections.desta3.data.distill_dataset import DistillDataset, BaseAudioTextDataset
from desta.collections.desta3.data.sampler import DistributedLengthBasedBatchSampler, LengthBasedBatchSampler
from torch.utils.data import DataLoader
from torch.nn import functional as F
import logging
import traceback


### Using Dataloader, validation step, optimizer config and state dict from DeSTA3PTLModule

class CoD_DistillPTLModule(DeSTA3PTLModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Apply FFN-only LoRA to student model for efficient fine-tuning
        # Store original embed_tokens before PEFT wrapping
        # if lora == True:
            # original_embed_tokens = self.model.llm_model.model.embed_tokens
            
            # lora_config = LoraConfig(
            #     r=8,  # rank
            #     lora_alpha=16,  # scaling parameter  
            #     target_modules=["gate_proj", "up_proj", "down_proj"],  # FFN layers only
            #     lora_dropout=0.1,
            #     bias="none",
            #     task_type="CAUSAL_LM"
            # )
            # self.model.llm_model = get_peft_model(self.model.llm_model, lora_config)
            
            # # Restore embed_tokens access for compatibility
            # self.model.llm_model.model.embed_tokens = original_embed_tokens
            # print(f"Applied FFN LoRA to student model: rank={lora_config.r}, alpha={lora_config.lora_alpha}")
        
        teacher_model_config = DeSTA3Config(
            llm_model_id=self.cfg.teacher_model.llm.model_id,
            encoder_model_id=self.cfg.teacher_model.encoder.model_id,
            connector_mode=self.cfg.teacher_model.connector.mode,
            qformer_num_hidden_layers=self.cfg.teacher_model.connector.num_hidden_layers,
            prompt_size=self.cfg.teacher_model.connector.prompt_size,
            first_n_layers=self.cfg.teacher_model.llm.first_n_layers if hasattr(self.cfg.teacher_model.llm, "first_n_layers") else -1,
        )
        print("="*100)
        self.teacher_model = DeSTA3Model(teacher_model_config) #from pretrain
        
        # remove whisper decoder during PTL training (we only use Whisper decoder during inference)
        del self.teacher_model.perception.whisper.model.decoder
        del self.teacher_model.perception.whisper.proj_out
        
        ckpt_path = getattr(cfg.teacher_model, "ckpt_path", None)
        if ckpt_path is not None and ckpt_path.lower() != "null":
            state = torch.load(ckpt_path, map_location="cpu")
            state = state.get("state_dict", state)

            # 1) 去常見前綴：teacher_model./model./module.
            def _strip(k: str) -> str:
                for pref in ("teacher_model.", "model.", "module."):
                    if k.startswith(pref):
                        return k[len(pref):]
                return k
            state = { _strip(k): v for k, v in state.items() }

            # 2) 你已刪掉的模組權重不要載（避免造成 unexpected）
            state = {k: v for k, v in state.items()
                    if not (k.startswith("perception.whisper.model.decoder.")
                            or k.startswith("perception.whisper.proj_out."))}

            # 3) 僅保留模型現有 key（減少噪音）
            model_keys = set(self.teacher_model.state_dict().keys())
            state = {k: v for k, v in state.items() if k in model_keys}

            # 4) 真的載入（只呼叫一次）
            incompat = self.teacher_model.load_state_dict(state, strict=False)

            # 5) 簡潔統計與 LLAMA vs 非 LLAMA 拆分
            missing = list(getattr(incompat, "missing_keys", getattr(incompat, "missing", [])))
            unexpected = list(getattr(incompat, "unexpected_keys", getattr(incompat, "unexpected", [])))
            missing_llama = [k for k in missing if k.startswith("llm_model.")]
            missing_non_llama = [k for k in missing if not k.startswith("llm_model.")]

            logging.info(f"Teacher ckpt loaded: {ckpt_path}")
            logging.info(f"  loaded_keys={len(state)} | missing={len(missing)} "
                        f"(llama={len(missing_llama)}, non-llama={len(missing_non_llama)}) "
                        f"| unexpected={len(unexpected)}")

            if missing_non_llama:
                logging.info("Missing (Non-LLaMA) examples: %s", missing_non_llama[:20])
            if unexpected:
                logging.info("Unexpected examples: %s", unexpected[:20])

        for p in self.teacher_model.parameters():   # 凍結參數
            p.requires_grad_(False)
            
        self.teacher_model.to("cuda")

        # # tokenizer
        # self.teacher_tokenizer = AutoTokenizer.from_pretrained(self.cfg.teacher_model.llm.model_id, cache_dir=os.getenv("HF_HOME"))
        # self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token
        # self.teacher_tokenizer.pad_token_id = self.teacher_tokenizer.eos_token_id
        # self.teacher_tokenizer.padding_side = "left"
        # self.teacher_tokenizer.add_tokens([self.cfg.dataset.audio_locator]) #warning cfg.dataset.audio_locator nee adding teacher?
        
        
        self.add_noise = self.cfg.model.add_noise
        self.kd_alpha  = self.cfg.model.kd_alpha
        self.strategy  = getattr(self.cfg.model, "strategy", "wise")   # average | shallow | deep | wise
        self.kd_temperature = getattr(self.cfg.model, "kd_temperature", 2.0)
        # Add hard label distillation option
        self.use_hard_labels = getattr(self.cfg.model, "use_hard_labels", False)
        # 保留 MSE criterion 以備不時之需
        self.criterion = torch.nn.MSELoss(reduction="mean")
        
        # Initialize projection layers dict for different layer pairs
        self.projection_layers = torch.nn.ModuleDict()
        
        print(self.strategy)

    # -------- core forward (student + teacher + KD) ----------
    def forward(self, batch):
        # === unpack batch ===
        input_ids              = batch["input_ids"]
        attention_mask         = batch["attention_mask"]
        batch_features         = batch["batch_features"]
        batch_transcription_ids= batch["batch_transcription_ids"]
        batch_start_positions  = batch["batch_start_positions"]
        labels                 = batch.get("labels", None)

        # 1) encoder
        # batch_speech_features, batch_speech_feature_lengths = self.model.perception(batch_features)

        # 2) prepare llm embeds for student(audio)
        student_input_embeds = self.model._prepare_inputs_for_llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            batch_features=batch_features,
            # batch_speech_feature_lengths=batch_speech_feature_lengths,
            batch_transcription_ids=batch_transcription_ids,
            batch_start_positions=batch_start_positions,
        )

        # If training, also build teacher(teacher) path
        if self.training:
            student_start_answer_positions = batch["audio_start_answer_positions"]
            teacher_start_answer_positions  = batch["audio_start_answer_positions"] # Warning: might not right (seed__start_answer_positions)

            teacher_input_ids      = batch["input_ids"]# warning batch["seed_input_ids"]
            # teacher_attention_mask = batch["seed_attention_mask"]
            
            # teacher_input_embeds   = self.model.llm_model.model.embed_tokens(teacher_input_ids)
            teacher_input_embeds = self.teacher_model._prepare_inputs_for_llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            batch_features=batch_features,
            # batch_speech_feature_lengths=batch_speech_feature_lengths,
            batch_transcription_ids=batch_transcription_ids,
            batch_start_positions=batch_start_positions,
            )

            if self.add_noise:
                student_input_embeds, teacher_input_embeds = self._add_noise(
                    student_input_embeds, teacher_input_embeds,
                    student_start_answer_positions, teacher_start_answer_positions
                )

        # 3) student forward
        student_output = self.model.llm_model(
            inputs_embeds        = student_input_embeds,
            attention_mask       = attention_mask,
            labels               = labels,
            output_hidden_states = True,
            use_cache            = False,
        )
        if self.training:
            self.log("train/ntp_loss", student_output.loss.item(),
                     sync_dist=True, batch_size=input_ids.size(0))

        # 4) teacher forward + KD
        if self.training:
            with torch.no_grad():
                teacher_output = self.teacher_model.llm_model(
                    inputs_embeds        = teacher_input_embeds,
                    attention_mask       = attention_mask,
                    output_hidden_states = True,
                    use_cache            = False,
                )

            if self.use_hard_labels:
                # Hard label distillation: use teacher's predictions as hard labels
                hard_kd_loss = self.hard_label_loss(
                    student_logits = student_output.logits,
                    teacher_logits = teacher_output.logits,
                    student_start_answer_positions = student_start_answer_positions,
                    teacher_start_answer_positions = teacher_start_answer_positions,
                    attention_mask = attention_mask
                )
                self.log("train/hard_kd_loss", hard_kd_loss.item(),
                         sync_dist=True, batch_size=input_ids.size(0))
                kd_loss = hard_kd_loss
                
            else:
                # Soft label distillation (original implementation)
                soft_kd_loss = self.output_distribution_loss(
                    student_logits = student_output.logits,
                    teacher_logits = teacher_output.logits,
                    student_start_answer_positions = student_start_answer_positions,
                    teacher_start_answer_positions = teacher_start_answer_positions,
                    temperature = self.kd_temperature
                )
                
                self.log("train/soft_kd_loss", soft_kd_loss.item(),
                         sync_dist=True, batch_size=input_ids.size(0))
                kd_loss = soft_kd_loss
                
            layer_kd_loss = self.layer_level_loss(
                    student_hidden_states = student_output.hidden_states,
                    teach_hidden_states  = teacher_output.hidden_states,
                    student_start_answer_positions = student_start_answer_positions,
                    teacher_start_answer_positions  = teacher_start_answer_positions,
                    temperature = self.kd_temperature,
                    strategy    = self.strategy
                )
            
            kd_loss = 0.7 * kd_loss + 0.3 * layer_kd_loss

            student_output.loss = (1 - self.kd_alpha) * student_output.loss + self.kd_alpha * kd_loss

        return student_output

    # ---------------- PTL hooks ----------------
    def training_step(self, batch, batch_idx):
        try:
            outputs = self(batch)
            loss = outputs.loss
            
        except Exception as e:
            logging.error(f"Error in training step: {e}")
            logging.error(traceback.format_exc())
            logging.error(f"Batch: {batch}")
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
        ppl = torch.exp(loss)
        bs  = batch["input_ids"].size(0)
        self.log("train/loss_total", loss, prog_bar=True,
                 sync_dist=True, batch_size=bs)
        self.log("train/ppl", ppl, prog_bar=True,
                 sync_dist=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        # distill val 不一定要算 KD，直接用 predict
        self.model.eval()
        preds = self.predict_step(batch, batch_idx)
        loss = torch.tensor(0.0, device=self.device)
        ppl  = torch.tensor(0.0, device=self.device)
        bs   = batch["input_ids"].size(0)
        self.log("val_loss", loss.item(), sync_dist=True, batch_size=bs)
        self.log("val_ppl",  ppl.item(),  sync_dist=True, batch_size=bs)
        return {"val_loss": loss, "val_ppl": ppl, "predictions": preds}

    # ---------------- Hard label distillation ----------------
    def hard_label_loss(
        self,
        student_logits,
        teacher_logits,
        student_start_answer_positions,
        teacher_start_answer_positions,
        attention_mask
    ):
        """
        Hard label distillation: Use teacher's predictions as ground truth labels for student.
        """
        total_loss = 0.0
        valid_samples = 0
        
        batch_size = min(len(student_logits), len(teacher_logits))
        
        for i in range(batch_size):
            if i >= len(student_start_answer_positions) or i >= len(teacher_start_answer_positions):
                continue
                
            student_start = student_start_answer_positions[i].item()
            teacher_start = teacher_start_answer_positions[i].item()
            
            # Get logits after answer positions
            student_answer_logits = student_logits[i][student_start:]
            teacher_answer_logits = teacher_logits[i][teacher_start:]
            
            # Align sequence lengths
            max_length = min(student_answer_logits.size(0), teacher_answer_logits.size(0))
            if max_length == 0:
                continue
                
            student_answer_logits = student_answer_logits[:max_length]
            teacher_answer_logits = teacher_answer_logits[:max_length]
            
            # Get attention mask for this sequence
            if attention_mask is not None and i < len(attention_mask):
                seq_attention_mask = attention_mask[i][student_start:student_start + max_length]
            else:
                seq_attention_mask = torch.ones(max_length, device=student_answer_logits.device)
            
            # Get teacher's hard predictions (argmax)
            with torch.no_grad():
                teacher_hard_labels = torch.argmax(teacher_answer_logits, dim=-1)
            
            # Compute cross-entropy loss between student logits and teacher hard labels
            # Flatten for cross-entropy computation
            student_logits_flat = student_answer_logits.view(-1, student_answer_logits.size(-1))
            teacher_labels_flat = teacher_hard_labels.view(-1)
            mask_flat = seq_attention_mask.view(-1)
            
            # Apply mask to ignore padded positions
            active_indices = mask_flat.bool()
            if active_indices.sum() == 0:
                continue
                
            active_student_logits = student_logits_flat[active_indices]
            active_teacher_labels = teacher_labels_flat[active_indices]
            
            # Compute cross-entropy loss
            sample_loss = F.cross_entropy(
                active_student_logits,
                active_teacher_labels,
                reduction='mean'
            )
            
            total_loss += sample_loss
            valid_samples += 1
        
        # Average across valid samples
        if valid_samples > 0:
            return total_loss / valid_samples
        else:
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)

    # ---------------- layer-wise KD ----------------
    def layer_level_loss(
        self,
        student_hidden_states,
        teach_hidden_states,
        student_start_answer_positions,
        teacher_start_answer_positions,
        temperature: float = 2.0,
        strategy: str = "base",
        shallow_k: int = 4,
        projection_method: str = "linear"
    ):
        """
        strategy:
          - wise: 逐層 KL，平均 (你原本的做法)
          - shallow: 只取前 K 層做 wise
          - deep: 只取後 K 層做 wise
          - average: 把所有層 concat 起來一次算 KL
        """
        s_hs = list(student_hidden_states)
        t_hs = list(teach_hidden_states)

        def get_or_create_projection_layer(layer_idx, student_dim, teacher_dim):
            """Get or create projection layer for specific layer pair"""
            if student_dim == teacher_dim:
                return None
                
            layer_key = f"layer_{layer_idx}_{teacher_dim}_to_{student_dim}"
            if layer_key not in self.projection_layers:
                if projection_method == "linear":
                    # Always project teacher to student dimension (more common in distillation)
                    self.projection_layers[layer_key] = torch.nn.Linear(
                        teacher_dim, student_dim
                    ).to(s_hs[0].device)
                else:
                    return None
            
            return self.projection_layers[layer_key]

        def align_hidden_dimensions(student_hidden, teacher_hidden, layer_idx, method=projection_method):
            """Align hidden dimensions between student and teacher"""
            if student_hidden.size(-1) == teacher_hidden.size(-1):
                return student_hidden, teacher_hidden
            
            if method == "linear":
                projection_layer = get_or_create_projection_layer(
                    layer_idx, student_hidden.size(-1), teacher_hidden.size(-1)
                )
                if projection_layer is not None:
                    # Always project teacher to student dimension
                    teacher_hidden = projection_layer(teacher_hidden)
            
            elif method == "truncate":
                min_dim = min(student_hidden.size(-1), teacher_hidden.size(-1))
                student_hidden = student_hidden[..., :min_dim]
                teacher_hidden = teacher_hidden[..., :min_dim]
                
            elif method == "pad":
                max_dim = max(student_hidden.size(-1), teacher_hidden.size(-1))
                if student_hidden.size(-1) < max_dim:
                    pad_size = max_dim - student_hidden.size(-1)
                    student_hidden = F.pad(student_hidden, (0, pad_size))
                if teacher_hidden.size(-1) < max_dim:
                    pad_size = max_dim - teacher_hidden.size(-1)
                    teacher_hidden = F.pad(teacher_hidden, (0, pad_size))
            
            return student_hidden, teacher_hidden

        # Handle different strategies
        if strategy == "shallow":
            num_layers_s = len(s_hs)
            num_layers_t = len(t_hs)
            target_layers_s = [7, 14, 21, 28]                  
            target_layers_t = [8, 16, 24, 32]
            idx_s = [l - 1 for l in target_layers_s if 1 <= l <= num_layers_s]
            idx_t = [l - 1 for l in target_layers_t if 1 <= l <= num_layers_t]
            s_hs = [s_hs[i] for i in idx_s]
            t_hs = [t_hs[i] for i in idx_t]
        
        elif strategy == "wise":
            num_layers_s = len(s_hs)
            num_layers_t = len(t_hs)
            target_layers_s = [2, 11, 19, 28]                  
            target_layers_t = [2, 8, 13, 30]
            idx_s = [l - 1 for l in target_layers_s if 1 <= l <= num_layers_s]
            idx_t = [l - 1 for l in target_layers_t if 1 <= l <= num_layers_t]
            s_hs = [s_hs[i] for i in idx_s]
            t_hs = [t_hs[i] for i in idx_t]
            
        elif strategy == "deep":
            s_hs, t_hs = s_hs[-shallow_k:], t_hs[-shallow_k:]
            
        elif strategy == "average":
            step_s = 7
            step_t = 8
            s_hs = s_hs[::step_s]
            t_hs = t_hs[::step_t]
            
        elif strategy == "base":
            s_hs = None
            t_hs = None
        # # Ensure same number of layers for comparison
        # min_layers = min(len(s_hs), len(t_hs))
        # s_hs = s_hs[:min_layers]
        # t_hs = t_hs[:min_layers]

        # Compute loss
        layer_loss = 0.0
        valid_computations = 0
        
        # Skip layer-level loss computation if hidden states are None (base strategy)
        if s_hs is None or t_hs is None:
            return 0.0 #torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        
        for layer_idx, (layer_student_hidden_states, layer_teach_hidden_states) in enumerate(zip(s_hs, t_hs)):
            # layer wise
            for i, (student_hidden_state, teach_hidden_state) in enumerate(zip(layer_student_hidden_states, layer_teach_hidden_states)):
                # in batch
                if i >= len(student_start_answer_positions) or i >= len(teacher_start_answer_positions):
                    continue
                    
                student_start = student_start_answer_positions[i].item()
                teach_start = teacher_start_answer_positions[i].item()

                # Only compute the loss after answer positions
                student_hidden_state = student_hidden_state[student_start:]
                teach_hidden_state = teach_hidden_state[teach_start:]

                # truncate the length to the same length
                max_length = min(student_hidden_state.size(0), teach_hidden_state.size(0))
                if max_length == 0:
                    continue
                    
                student_hidden_state = student_hidden_state[:max_length]
                teach_hidden_state = teach_hidden_state[:max_length]

                # Align hidden dimensions with layer-specific projection
                student_hidden_state, teach_hidden_state = align_hidden_dimensions(
                    student_hidden_state, teach_hidden_state, layer_idx
                )

                # Compute MSE loss between hidden states
                loss_kd = F.mse_loss(student_hidden_state, teach_hidden_state, reduction='mean')

                layer_loss += loss_kd
                valid_computations += 1
            
        # Proper normalization by all valid computations
        if valid_computations > 0:
            return layer_loss / valid_computations
        else:
            return 0.0 #torch.tensor(0.0, device=s_hs[0].device, requires_grad=True)

    # ---------------- output distribution KD ----------------
    def output_distribution_loss(
        self,
        student_logits,
        teacher_logits,
        student_start_answer_positions,
        teacher_start_answer_positions,
        temperature: float = 2.0
    ):
        """Compute KL divergence loss on output token distributions"""
        output_distribution_loss = 0.0
        valid_output_pairs = 0
        
        batch_size = min(len(student_logits), len(teacher_logits))
        for i in range(batch_size):
            if i >= len(student_start_answer_positions) or i >= len(teacher_start_answer_positions):
                continue
                
            student_start = student_start_answer_positions[i].item()
            teacher_start = teacher_start_answer_positions[i].item()
            
            # Get logits after answer positions
            student_output_logits = student_logits[i][student_start:]
            teacher_output_logits = teacher_logits[i][teacher_start:]
            
            # Align sequence lengths
            max_length = min(student_output_logits.size(0), teacher_output_logits.size(0))
            if max_length == 0:
                continue
                
            student_output_logits = student_output_logits[:max_length]
            teacher_output_logits = teacher_output_logits[:max_length]
            
            # Compute KL divergence on output distributions
            student_log_probs = F.log_softmax(student_output_logits / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_output_logits / temperature, dim=-1)
            
            output_kl_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='batchmean'
            ) * (temperature ** 2)
            
            output_distribution_loss += output_kl_loss
            valid_output_pairs += 1
        
        # Average output distribution loss
        if valid_output_pairs > 0:
            return output_distribution_loss / valid_output_pairs
        else:
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)



    def _add_noise(self, student_input_embeds, teach_input_embeds, student_start_answer_positions, teacher_start_answer_positions):
            # Create new copies of the input tensors

            # Add same noise only to the portions after answer positions
            for i in range(len(student_start_answer_positions)):
                student_start = student_start_answer_positions[i].item()
                teach_start = teacher_start_answer_positions[i].item()

                student_answer_embeds = student_input_embeds[i, student_start:]
                teach_answer_embeds = teach_input_embeds[i, teach_start:]
                
                max_length = min(student_answer_embeds.size(0), teach_answer_embeds.size(0))
                noise = torch.randn_like(student_answer_embeds[:max_length])
                # replace the answer embeddings with noise
                # USE noise
                student_answer_embeds[:max_length] = noise
                teach_answer_embeds[:max_length] = noise

                student_input_embeds[i, student_start:student_start+max_length] = student_answer_embeds[:max_length]
                teach_input_embeds[i, teach_start:teach_start+max_length] = teach_answer_embeds[:max_length]
            

            return student_input_embeds, teach_input_embeds
    
    def state_dict(self):
        """
        Override parent state_dict to ensure projection layers are included in checkpoints
        """
        trainable_state_dict = super().state_dict()
        
        # Explicitly ensure projection layers are included
        for name, param in self.projection_layers.named_parameters():
            full_name = f"projection_layers.{name}"
            if param.requires_grad:
                trainable_state_dict[full_name] = param.data.clone().detach()
        
        return trainable_state_dict
        
        
    # def _build_dataloader(self, data_cfg):
    #         return_seed = data_cfg.get("return_seed", False)
    #         if return_seed:
    #             dataset = DistillDataset(
    #                 cfg=self.cfg,
    #                 data_cfg=data_cfg,
    #                 tokenizer=self.tokenizer,
    #                 processor=self.processor,
    #                 return_seed=True
    #             )
    #         else:
    #             dataset = BaseAudioTextDataset(
    #                 cfg=self.cfg,
    #                 data_cfg=data_cfg,
    #                 tokenizer=self.tokenizer,
    #                 processor=self.processor
    #             )
    #         logging.info(dataset[0])

    #         if data_cfg.get("batch_sampler", None) == "length_based":
    #             if self.trainer.world_size > 1:
    #                 batch_sampler = DistributedLengthBasedBatchSampler(
    #                     dataset,
    #                     max_batch_length=data_cfg.max_batch_length,
    #                     num_replicas=self.trainer.world_size,
    #                     rank=self.trainer.global_rank,
    #                     shuffle=data_cfg.shuffle,
    #                 )
    #             else:
    #                 batch_sampler = LengthBasedBatchSampler(
    #                     dataset,
    #                     max_batch_length=data_cfg.max_batch_length,
    #                     shuffle=data_cfg.shuffle,
    #                     drop_last=data_cfg.drop_last,
    #                 )
    #             dataloader = DataLoader(
    #                 dataset,
    #                 batch_sampler=batch_sampler,
    #                 num_workers=data_cfg.num_workers,
    #                 pin_memory=data_cfg.pin_memory,
    #                 collate_fn=dataset.collate_fn,
    #             )
    #         else:
    #             dataloader = DataLoader(
    #                 dataset,
    #                 batch_size=data_cfg.batch_size,
    #                 collate_fn=dataset.collate_fn,
    #                 shuffle=data_cfg.shuffle,
    #                 num_workers=data_cfg.num_workers,
    #                 pin_memory=data_cfg.pin_memory,
    #                 drop_last=data_cfg.drop_last,
    #             )
    #         return dataloader
        


    # model:
    #   kd_alpha: 0.5
    #   kd_temperature: 2.0
    #   strategy: wise   # or average / shallow / deep
    #   shallow_k: 4
    #   use_hard_labels: false   # true for hard label distillation, false for soft label distillation
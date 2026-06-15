from desta.collections.ptl_modules.desta3_ptl import DeSTA3PTLModule
from desta.collections.desta3.models.modeling_desta3 import DeSTA3Model, DeSTA3Config
import torch
from typing import Optional, Tuple, Union
import logging
import os
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
import math


### Contrastive Distillation Module
### Key differences from CoD:
### 1. Added multi-layer distillation support for MSE and cross-attention modes
###    - AUTO-SAMPLING: Uses step approach (every Nth layer) like CoD
###    - Each layer pair has its own projection layer (not shared!)
###    - No manual layer specification needed!
### 2. Added contrastive learning on output logits (KL divergence) and representations (MSE/cross-attention)
### 3. Teacher performs two forward passes: with audio (positive) and without audio (negative)
### 4. Contrastive formula: (1 + β) * with_audio - β * without_audio
### 5. TWO separate beta values: contrastive_beta (logits) and repr_contrastive_beta (repr)
###    - Setting repr_contrastive_beta=0 → NON-contrastive mode (like CoD's layer_level_loss)
### 6. Loss weighting matches CoD: 70% logit-level loss, 30% representation-level loss

class ContrastiveDistillPTLModule(DeSTA3PTLModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Check if we're in evaluation-only mode (skip teacher initialization to save VRAM)
        eval_only = os.environ.get("DESTA_EVAL_ONLY", "0") == "1"

        if not eval_only:
            # Initialize teacher model (only during training)
            teacher_model_config = DeSTA3Config(
                llm_model_id=self.cfg.teacher_model.llm.model_id,
                encoder_model_id=self.cfg.teacher_model.encoder.model_id,
                connector_mode=self.cfg.teacher_model.connector.mode,
                qformer_num_hidden_layers=self.cfg.teacher_model.connector.num_hidden_layers,
                prompt_size=self.cfg.teacher_model.connector.prompt_size,
                first_n_layers=self.cfg.teacher_model.llm.first_n_layers if hasattr(self.cfg.teacher_model.llm, "first_n_layers") else -1,
            )
            print("="*100)
            print("Initializing Contrastive Distillation Module")
            print("="*100)
            self.teacher_model = DeSTA3Model(teacher_model_config)

            # Remove whisper decoder during PTL training (we only use Whisper decoder during inference)
            del self.teacher_model.perception.whisper.model.decoder
            del self.teacher_model.perception.whisper.proj_out

            # Load teacher checkpoint
            ckpt_path = getattr(cfg.teacher_model, "ckpt_path", None)
            if ckpt_path is not None and ckpt_path.lower() != "null":
                state = torch.load(ckpt_path, map_location="cpu")
                state = state.get("state_dict", state)

                # 1) Strip common prefixes: teacher_model./model./module.
                def _strip(k: str) -> str:
                    for pref in ("teacher_model.", "model.", "module."):
                        if k.startswith(pref):
                            return k[len(pref):]
                    return k
                state = { _strip(k): v for k, v in state.items() }

                # 2) Remove deleted module weights (avoid unexpected keys)
                state = {k: v for k, v in state.items()
                        if not (k.startswith("perception.whisper.model.decoder.")
                                or k.startswith("perception.whisper.proj_out."))}

                # 3) Only keep model existing keys (reduce noise)
                model_keys = set(self.teacher_model.state_dict().keys())
                state = {k: v for k, v in state.items() if k in model_keys}

                # 4) Load state dict
                incompat = self.teacher_model.load_state_dict(state, strict=False)

                # 5) Log statistics
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

            # Freeze teacher parameters
            for p in self.teacher_model.parameters():
                p.requires_grad_(False)

            self.teacher_model.to("cuda")
        else:
            print("="*100)
            print("EVALUATION MODE: Skipping teacher model initialization (saves VRAM)")
            print("="*100)
            self.teacher_model = None  # Set to None to avoid errors

        # Contrastive learning hyperparameters
        self.kd_alpha = self.cfg.model.kd_alpha  # Weight for total KD loss vs NTP loss
        self.contrastive_beta = getattr(self.cfg.model, "contrastive_beta", 0.5)  # β for logit contrastive
        self.repr_contrastive_beta = getattr(self.cfg.model, "repr_contrastive_beta", None)  # β for repr contrastive (if None, uses contrastive_beta)
        # If repr_contrastive_beta not specified, use same as contrastive_beta
        if self.repr_contrastive_beta is None:
            self.repr_contrastive_beta = self.contrastive_beta
        self.kd_temperature = getattr(self.cfg.model, "kd_temperature", 2.0)

        # Loss component flags - enable/disable each type
        self.use_standard_kd = getattr(self.cfg.model, "use_standard_kd", False)  # Standard KL divergence
        self.use_logit_contrastive = getattr(self.cfg.model, "use_logit_contrastive", True)  # Logit contrastive (MSE)
        self.use_repr_contrastive = getattr(self.cfg.model, "use_repr_contrastive", True)  # Representation contrastive (MSE)

        # Representation contrastive loss type: "mse" or "cross_attention"
        self.repr_contrastive_loss_type = getattr(self.cfg.model, "repr_contrastive_loss_type", "mse")
        if self.repr_contrastive_loss_type not in ["mse", "cross_attention"]:
            raise ValueError(f"repr_contrastive_loss_type must be 'mse' or 'cross_attention', got '{self.repr_contrastive_loss_type}'")

        # Cross-attention loss scaling factor (to match magnitude with other losses)
        # Cross-attention loss is typically small (-1.5 to +1.5), while other losses are larger
        # Scale up to prevent it from being dominated by MSE/KL losses
        self.cross_attn_loss_scale = getattr(self.cfg.model, "cross_attn_loss_scale", 10.0)

        # Multi-layer distillation configuration (AUTO-SAMPLING like CoD)
        self.distill_layer_mode = getattr(self.cfg.model, "distill_layer_mode", "last")

        # Automatic layer sampling parameters (like CoD)
        self.distill_student_step = getattr(self.cfg.model, "distill_student_step", 7)  # Every 7th layer
        self.distill_teacher_step = getattr(self.cfg.model, "distill_teacher_step", 8)  # Every 8th layer

        # Validation for average mode
        if self.distill_layer_mode not in ["last", "average"]:
            raise ValueError(f"distill_layer_mode must be 'last' or 'average', got '{self.distill_layer_mode}'")

        if self.distill_layer_mode == "average":
            logging.info(f"[Multi-Layer Distillation] Mode: average (AUTO-SAMPLING like CoD)")
            logging.info(f"  Student: Every {self.distill_student_step}th layer")
            logging.info(f"  Teacher: Every {self.distill_teacher_step}th layer")

        # Check at least one loss is enabled
        if not any([self.use_standard_kd, self.use_logit_contrastive, self.use_repr_contrastive]):
            raise ValueError("At least one loss component must be enabled: use_standard_kd, use_logit_contrastive, or use_repr_contrastive")

        # Initialize projection layers dict for different layer pairs (CoD-style on-demand creation)
        self.projection_layers = torch.nn.ModuleDict()

        # Entropy calculation configuration
        self.enable_entropy_calculation = getattr(self.cfg.model, "entropy_calculation", False)
        if self.enable_entropy_calculation:
            # Storage for all samples across FIRST epoch only
            self.entropy_with_audio_samples = []
            self.entropy_contrastive_samples = []
            self.entropy_computed = False  # Flag to ensure we only compute once
            logging.info("Entropy calculation enabled (first epoch only, saves to JSON)")

        # Evaluation entropy calculation (separate from training)
        # Uses SAME K-cand and K-agg as LogTokU for fair comparison
        self.enable_eval_entropy = getattr(self.cfg.model, "enable_eval_entropy", False)
        if self.enable_eval_entropy:
            # Storage for evaluation entropy samples (will store dicts with K-agg variants)
            self.eval_entropy_samples = []
            logging.info("Evaluation entropy calculation enabled (uses same K-cand and K-agg as LogTokU)")

        # Shared K parameters for BOTH entropy and LogTokU (for fair comparison)
        # K-cand: Top-K logits for computing per-token uncertainty (Section 3.4, Equation 3)
        self.topk_cand = getattr(self.cfg.model, "topk_cand", 20)  # Default K=20 based on paper ablation
        # K-agg: Compute multiple aggregation strategies simultaneously for comparison
        # Will compute K-agg = [1, 10, 20, all] at once (Section 5.1, Equation 7)
        self.kagg_values = [1, 10, 20, "all"]  # All K-agg values to evaluate

        # LogTokU (AU/EU) calculation configuration
        self.enable_logtoku = getattr(self.cfg.model, "enable_logtoku", False)
        if self.enable_logtoku:
            # Storage for evaluation samples
            # AU/EU: Single values per sample (average over all tokens)
            # Uncertainty: Dicts with K-agg variants per sample
            self.eval_au_samples = []
            self.eval_eu_samples = []
            self.eval_uncertainty_samples = []
            logging.info(f"LogTokU (AU/EU) calculation enabled:")
            logging.info(f"  K-cand (top-K logits): {self.topk_cand} (shared with entropy)")
            logging.info(f"  K-agg (worst-K tokens): {self.kagg_values} (only for uncertainty/entropy, not AU/EU)")
            logging.info(f"  AU/EU: Single values (average over all tokens)")
            logging.info(f"  Uncertainty: Multiple K-agg values (AU*EU, Eq 8: R=-AU*EU)")

        print(f"Contrastive Distillation Config:")
        print(f"  kd_alpha: {self.kd_alpha}")
        print(f"  contrastive_beta (logits): {self.contrastive_beta}")
        print(f"  repr_contrastive_beta (repr): {self.repr_contrastive_beta}")
        print(f"  kd_temperature: {self.kd_temperature}")
        print(f"  Loss Components:")
        print(f"    - Standard KD (KL divergence): {self.use_standard_kd}")
        print(f"    - Logit Contrastive (KL divergence, beta={self.contrastive_beta}): {self.use_logit_contrastive}")
        print(f"    - Representation Contrastive ({self.repr_contrastive_loss_type.upper()}, beta={self.repr_contrastive_beta}): {self.use_repr_contrastive}")
        if self.repr_contrastive_beta == 0:
            print(f"      NOTE: repr_contrastive_beta=0 → Non-contrastive mode (uses teacher with audio only)")
        if self.use_repr_contrastive and self.repr_contrastive_loss_type == "cross_attention":
            print(f"      Cross-Attention loss scale: {self.cross_attn_loss_scale}")

    # -------- Core forward (student + teacher + contrastive KD) ----------
    def forward(self, batch):
        # === Unpack batch ===
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        batch_features = batch["batch_features"]
        batch_transcription_ids = batch["batch_transcription_ids"]
        batch_start_positions = batch["batch_start_positions"]
        labels = batch.get("labels", None)

        # 1) Prepare LLM embeds for student (with audio)
        student_input_embeds = self.model._prepare_inputs_for_llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            batch_features=batch_features,
            batch_transcription_ids=batch_transcription_ids,
            batch_start_positions=batch_start_positions,
        )

        # 2) Student forward pass
        student_output = self.model.llm_model(
            inputs_embeds=student_input_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            use_cache=False,
        )

        if self.training:
            self.log("train/ntp_loss", student_output.loss.item(),
                     sync_dist=True, batch_size=input_ids.size(0))

        # 3) Teacher forward passes + Contrastive KD
        if self.training:
            student_start_answer_positions = batch["audio_start_answer_positions"]
            teacher_start_answer_positions = batch["audio_start_answer_positions"]

            with torch.no_grad():
                # Teacher forward WITH audio (positive pair)
                teacher_input_embeds_with_audio = self.teacher_model._prepare_inputs_for_llm(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    batch_features=batch_features,
                    batch_transcription_ids=batch_transcription_ids,
                    batch_start_positions=batch_start_positions,
                )

                teacher_output_with_audio = self.teacher_model.llm_model(
                    inputs_embeds=teacher_input_embeds_with_audio,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )

                # OPTIMIZATION: Only compute teacher WITHOUT audio if needed for contrastive learning
                # When BOTH contrastive_beta=0 AND repr_contrastive_beta=0, skip to save ~50% teacher compute
                need_without_audio = (
                    (self.use_logit_contrastive and self.contrastive_beta != 0) or
                    (self.use_repr_contrastive and self.repr_contrastive_beta != 0)
                )

                if need_without_audio:
                    # Teacher forward WITHOUT audio (negative pair)
                    # Use same preparation pipeline but with zero audio features to ensure sequence length matches
                    teacher_input_embeds_without_audio = self.teacher_model._prepare_inputs_for_llm(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        batch_features=torch.zeros_like(batch_features),  # Zero audio features
                        batch_transcription_ids=batch_transcription_ids,
                        batch_start_positions=batch_start_positions,
                    )

                    teacher_output_without_audio = self.teacher_model.llm_model(
                        inputs_embeds=teacher_input_embeds_without_audio,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                    )
                else:
                    # FAST PATH: Both betas are 0, without_audio is unused
                    # Reuse with_audio outputs to avoid None errors in loss functions
                    teacher_output_without_audio = teacher_output_with_audio

            # === Compute loss components based on config ===
            loss_components = []
            loss_weights = []

            # 1. Standard KD (KL divergence with with-audio teacher only)
            if self.use_standard_kd:
                standard_kd = self.standard_kl_loss(
                    student_logits=student_output.logits,
                    teacher_logits=teacher_output_with_audio.logits,
                    student_start_answer_positions=student_start_answer_positions,
                    teacher_start_answer_positions=teacher_start_answer_positions,
                    temperature=self.kd_temperature
                )
                self.log("train/standard_kd_loss", standard_kd.item(),
                         sync_dist=True, batch_size=input_ids.size(0))
                loss_components.append(standard_kd)
                loss_weights.append(0.7)  # 70% weight for logit-level loss (matches CoD)

            # 2. Logit Contrastive (KL divergence on contrastive logits)
            if self.use_logit_contrastive:
                logit_contrastive = self.logit_contrastive_loss(
                    student_logits=student_output.logits,
                    teacher_logits_with_audio=teacher_output_with_audio.logits,
                    teacher_logits_without_audio=teacher_output_without_audio.logits,
                    student_start_answer_positions=student_start_answer_positions,
                    teacher_start_answer_positions=teacher_start_answer_positions,
                    contrastive_beta=self.contrastive_beta,
                    temperature=self.kd_temperature
                )
                self.log("train/logit_contrastive_loss", logit_contrastive.item(),
                         sync_dist=True, batch_size=input_ids.size(0))
                loss_components.append(logit_contrastive)
                loss_weights.append(0.7)  # 70% weight for logit-level loss (matches CoD)

            # === Entropy Calculation (if enabled and not yet computed) ===
            # Only compute in first epoch since teacher is frozen
            if self.enable_entropy_calculation and not self.entropy_computed:
                self._compute_and_log_entropy(
                    teacher_logits_with_audio=teacher_output_with_audio.logits,
                    teacher_logits_without_audio=teacher_output_without_audio.logits,
                    teacher_start_answer_positions=teacher_start_answer_positions,
                    batch_size=input_ids.size(0)
                )

            # 3. Representation Contrastive (MSE/cross-attention on contrastive hidden states)
            if self.use_repr_contrastive:
                repr_contrastive = self.contrastive_representation_loss(
                    student_hidden_states=student_output.hidden_states,
                    teacher_hidden_states_with_audio=teacher_output_with_audio.hidden_states,
                    teacher_hidden_states_without_audio=teacher_output_without_audio.hidden_states,
                    student_start_answer_positions=student_start_answer_positions,
                    teacher_start_answer_positions=teacher_start_answer_positions,
                    contrastive_beta=self.repr_contrastive_beta
                )
                self.log("train/repr_contrastive_loss", repr_contrastive.item(),
                         sync_dist=True, batch_size=input_ids.size(0))
                loss_components.append(repr_contrastive)
                loss_weights.append(0.3)  # 30% weight for representation-level loss (matches CoD)

            # === Combine losses with 70-30 weighting (matches CoD) ===
            # Logit losses (standard_kd or logit_contrastive): 70%
            # Representation loss (repr_contrastive): 30%
            total_weight = sum(loss_weights)
            total_kd_loss = sum(w * loss for w, loss in zip(loss_weights, loss_components)) / total_weight

            self.log("train/total_kd_loss", total_kd_loss.item(),
                     sync_dist=True, batch_size=input_ids.size(0))

            # Final loss: NTP + KD
            student_output.loss = (1 - self.kd_alpha) * student_output.loss + self.kd_alpha * total_kd_loss

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
        bs = batch["input_ids"].size(0)
        self.log("train/loss_total", loss, prog_bar=True,
                 sync_dist=True, batch_size=bs)
        self.log("train/ppl", ppl, prog_bar=True,
                 sync_dist=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation doesn't need KD, use predict directly
        self.model.eval()
        preds = self.predict_step(batch, batch_idx)
        loss = torch.tensor(0.0, device=self.device)
        ppl = torch.tensor(0.0, device=self.device)
        bs = batch["input_ids"].size(0)
        self.log("val_loss", loss.item(), sync_dist=True, batch_size=bs)
        self.log("val_ppl", ppl.item(), sync_dist=True, batch_size=bs)
        return {"val_loss": loss, "val_ppl": ppl, "predictions": preds}

    def predict_step(self, batch, batch_idx):
        """
        Override predict_step to compute entropy and LogTokU on GENERATED PREDICTIONS.

        IMPORTANT: This computes uncertainty on the actual generated tokens, not ground truth!

        This method:
        1. Generates predictions with logit capture (output_scores=True)
        2. If enable_eval_entropy or enable_logtoku is True:
           - Captures logits from GENERATION PROCESS (not ground truth)
           - Computes entropy/AU/EU/uncertainty on GENERATED tokens
           - Length matches generated prediction (not ground truth)
        3. Records predictions with all computed metrics:
           - au: Single AU value (average over all GENERATED tokens)
           - eu: Single EU value (average over all GENERATED tokens)
           - au_sequence: Full per-token AU values for GENERATED sequence (list)
           - eu_sequence: Full per-token EU values for GENERATED sequence (list)
           - uncertainty_sequence: Full per-token uncertainty for GENERATED sequence (list)
           - entropy_k{1,10,20,all}: Entropy with different K-agg values
           - uncertainty_k{1,10,20,all}: Uncertainty with different K-agg values

        What this measures:
        - "How uncertain was the model when generating each token of its prediction?"
        - Useful for: Detecting hallucinations, measuring confidence in real predictions
        - Length: Matches generated prediction length (not ground truth length)
        """
        self.model.eval()

        # === GENERATION WITH LOGIT CAPTURE (for entropy/AU/EU on PREDICTIONS) ===
        # Generate predictions and capture generation-time logits
        if hasattr(self.cfg.model, 'generation_kwargs'):
            # Convert OmegaConf to dict if needed
            generation_kwargs = dict(self.cfg.model.generation_kwargs)
        else:
            generation_kwargs = {}

        # Enable logit capture if metrics are enabled
        if self.enable_eval_entropy or self.enable_logtoku:
            generation_kwargs['output_scores'] = True
            generation_kwargs['return_dict_in_generate'] = True

        # Generate with logit capture
        generated_outputs = self.model._generate_step(
            batch,
            pad_token_id=self.tokenizer.eos_token_id,
            generation_kwargs=generation_kwargs
        )

        # Extract generated IDs and scores (logits)
        if isinstance(generated_outputs, dict):
            generated_ids = generated_outputs['sequences']
            generation_scores = generated_outputs.get('scores', None)  # Tuple of [batch, vocab] tensors
        else:
            # Fallback if dict not returned
            generated_ids = generated_outputs
            generation_scores = None

        # === ENTROPY AND LOGTOKU CALCULATION ON GENERATION LOGITS ===
        batch_entropies = []
        batch_aus = []
        batch_eus = []
        batch_uncertainties = []
        batch_au_seqs = []
        batch_eu_seqs = []
        batch_uncertainty_seqs = []

        if (self.enable_eval_entropy or self.enable_logtoku) and generation_scores is not None:
            # Convert generation scores to logits tensor
            # generation_scores is a tuple of tensors, each [batch, vocab_size]
            # Stack to get [batch, seq_len, vocab_size]
            #
            # KEY: These are the ACTUAL logits produced during generation!
            # - Each position corresponds to one generated token
            # - Length = number of tokens in generated prediction
            # - This measures: "How uncertain was model when generating this prediction?"
            with torch.no_grad():
                # Stack all generation steps into single tensor
                generation_logits = torch.stack(generation_scores, dim=1)  # [batch, gen_len, vocab_size]

                batch_size = generation_logits.size(0)

                # Compute metrics for each sample using GENERATED logits (not ground truth!)
                for i in range(batch_size):
                    sample_logits = generation_logits[i]  # [gen_len, vocab_size]

                    # Skip if no tokens generated
                    if sample_logits.size(0) == 0:
                        batch_entropies.append(None)
                        batch_aus.append(None)
                        batch_eus.append(None)
                        batch_uncertainties.append(None)
                        batch_au_seqs.append(None)
                        batch_eu_seqs.append(None)
                        batch_uncertainty_seqs.append(None)
                        continue

                    # Compute entropy on generation logits
                    if self.enable_eval_entropy:
                        entropy_dict = self._compute_region_entropy(sample_logits)
                        batch_entropies.append(entropy_dict)
                        self.eval_entropy_samples.append(entropy_dict)
                    else:
                        batch_entropies.append(None)

                    # Compute AU/EU/Uncertainty on generation logits
                    if self.enable_logtoku:
                        au_val, eu_val, uncertainty_dict, au_seq, eu_seq, uncertainty_seq = self._compute_region_au_eu(sample_logits)
                        batch_aus.append(au_val)
                        batch_eus.append(eu_val)
                        batch_uncertainties.append(uncertainty_dict)
                        batch_au_seqs.append(au_seq)
                        batch_eu_seqs.append(eu_seq)
                        batch_uncertainty_seqs.append(uncertainty_seq)

                        # Store for aggregate statistics
                        self.eval_au_samples.append(au_val)
                        self.eval_eu_samples.append(eu_val)
                        self.eval_uncertainty_samples.append(uncertainty_dict)
                    else:
                        batch_aus.append(None)
                        batch_eus.append(None)
                        batch_uncertainties.append(None)
                        batch_au_seqs.append(None)
                        batch_eu_seqs.append(None)
                        batch_uncertainty_seqs.append(None)
        else:
            # All calculations disabled or no scores available
            batch_size = batch["input_ids"].size(0)
            batch_entropies = [None] * batch_size
            batch_aus = [None] * batch_size
            batch_eus = [None] * batch_size
            batch_uncertainties = [None] * batch_size
            batch_au_seqs = [None] * batch_size
            batch_eu_seqs = [None] * batch_size
            batch_uncertainty_seqs = [None] * batch_size

        # Process generated outputs (same as parent class)
        batch["context_input_ids"][batch["context_input_ids"] == -100] = self.tokenizer.eos_token_id
        batch["labels"][batch["labels"] == -100] = self.tokenizer.eos_token_id
        generated_ids[generated_ids == -100] = self.tokenizer.eos_token_id

        contexts = self.tokenizer.batch_decode(batch["context_input_ids"], skip_special_tokens=False)
        labels = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
        preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Record predictions WITH entropy/AU/EU/uncertainty
        # Note: AU/EU are single values (average over all tokens, only K-cand)
        #       Entropy/Uncertainty have multiple K-agg variants (K-cand + K-agg)
        #       AU/EU sequences are full per-token values for detailed analysis
        for context, label, pred, metadata, entropy_dict, au_val, eu_val, uncertainty_dict, au_seq, eu_seq, unc_seq in zip(
            contexts, labels, preds, batch["metadata"], batch_entropies, batch_aus, batch_eus, batch_uncertainties,
            batch_au_seqs, batch_eu_seqs, batch_uncertainty_seqs
        ):
            metadata.update({
                "context": context,
                "prediction": pred,
                "label": label,
            })

            # Add AU/EU: Single values (average over ALL tokens, only affected by K-cand)
            if au_val is not None:
                metadata["au"] = au_val
            if eu_val is not None:
                metadata["eu"] = eu_val

            # Add AU/EU sequences: Full per-token values (list of floats)
            if au_seq is not None:
                metadata["au_sequence"] = au_seq
            if eu_seq is not None:
                metadata["eu_sequence"] = eu_seq
            if unc_seq is not None:
                metadata["uncertainty_sequence"] = unc_seq

            # Add entropy for ALL K-agg values (affected by K-cand + K-agg)
            # entropy_dict contains results for K-agg = [1, 10, 20, all]
            if entropy_dict is not None:
                for k_agg, ent_val in entropy_dict.items():
                    metadata[f"entropy_k{k_agg}"] = ent_val

            # Add uncertainty for ALL K-agg values (affected by K-cand + K-agg)
            # uncertainty_dict contains results for K-agg = [1, 10, 20, all]
            if uncertainty_dict is not None:
                for k_agg, unc_val in uncertainty_dict.items():
                    metadata[f"uncertainty_k{k_agg}"] = unc_val

            self.prediction_step_outputs.append(metadata)

        return {"loss": 0}

    def _compute_eval_entropy_per_sample(self, logits, answer_start_positions):
        """
        Compute entropy for answer regions during EVALUATION using K-cand and K-agg.
        Returns entropy per sample for ALL K-agg values simultaneously.

        This computes entropy on the GROUND TRUTH answer region, not generated tokens.
        Uses SAME K-cand and K-agg as LogTokU for fair comparison.

        Args:
            logits: [batch, seq_len, vocab_size] - Student model logits
            answer_start_positions: [batch] - Starting positions for answers

        Returns:
            List of entropy dictionaries, one per sample in batch
            Each dict has keys {1, 10, 20, "all"} with entropy values
        """
        batch_size = logits.size(0)
        entropy_list = []

        for i in range(batch_size):
            if i >= len(answer_start_positions):
                entropy_list.append(None)
                continue

            answer_start = answer_start_positions[i].item()

            # Extract answer region: [answer_length, vocab_size]
            answer_logits = logits[i, answer_start:, :]

            # Skip if answer region is empty
            if answer_logits.size(0) == 0:
                entropy_list.append(None)
                continue

            # Compute entropy for ALL K-agg values (same method as LogTokU)
            entropy_dict = self._compute_region_entropy(answer_logits)
            entropy_list.append(entropy_dict)

        return entropy_list

    # ---------------- Contrastive representation loss (dispatcher) ----------------
    def contrastive_representation_loss(
        self,
        student_hidden_states,
        teacher_hidden_states_with_audio,
        teacher_hidden_states_without_audio,
        student_start_answer_positions,
        teacher_start_answer_positions,
        contrastive_beta: float = 0.5
    ):
        """
        Contrastive loss on final output representations.

        This method dispatches to the appropriate loss function based on configuration:
        - "mse": MSE-based contrastive loss with formula (1 + β) * with_audio - β * without_audio
        - "cross_attention": Similarity-based contrastive loss with formula -((1+β)*sim_pos - β*sim_neg)

        Args:
            student_hidden_states: Student model hidden states
            teacher_hidden_states_with_audio: Teacher hidden states WITH audio (positive)
            teacher_hidden_states_without_audio: Teacher hidden states WITHOUT audio (negative)
            student_start_answer_positions: Starting positions for student answers
            teacher_start_answer_positions: Starting positions for teacher answers
            contrastive_beta: Contrastive weighting factor

        Returns:
            loss: Scalar tensor representing the contrastive loss
        """
        if self.repr_contrastive_loss_type == "mse":
            return self._mse_contrastive_loss(
                student_hidden_states,
                teacher_hidden_states_with_audio,
                teacher_hidden_states_without_audio,
                student_start_answer_positions,
                teacher_start_answer_positions,
                contrastive_beta
            )
        elif self.repr_contrastive_loss_type == "cross_attention":
            return self.cross_attention_contrastive_loss(
                student_hidden_states,
                teacher_hidden_states_with_audio,
                teacher_hidden_states_without_audio,
                student_start_answer_positions,
                teacher_start_answer_positions,
                contrastive_beta
            )
        else:
            raise ValueError(f"Unknown repr_contrastive_loss_type: {self.repr_contrastive_loss_type}")

    # ---------------- MSE-based Contrastive Loss (Supports Multi-Layer) ----------------
    def _mse_contrastive_loss(
        self,
        student_hidden_states,
        teacher_hidden_states_with_audio,
        teacher_hidden_states_without_audio,
        student_start_answer_positions,
        teacher_start_answer_positions,
        contrastive_beta: float = 0.5
    ):
        """
        MSE-based contrastive loss on hidden state representations.
        Formula: MSE(student, (1 + β) * with_audio - β * without_audio)

        Supports two modes:
          - "last": Only use final layer (original behavior)
          - "average": Use user-specified layers and average the loss

        The student's representation should be close to:
        - Positive: teacher with audio (enhanced by 1+β)
        - Negative: teacher without audio (reduced by β)
        """
        if self.distill_layer_mode == "last":
            # Original behavior: only use final layer
            return self._compute_mse_single_layer(
                student_hidden_states[-1],
                teacher_hidden_states_with_audio[-1],
                teacher_hidden_states_without_audio[-1],
                student_start_answer_positions,
                teacher_start_answer_positions,
                contrastive_beta,
                layer_key="default"
            )

        elif self.distill_layer_mode == "average":
            # Multi-layer mode: auto-sample layers using step approach (like CoD)
            # Sample every Nth layer from both student and teacher
            student_hs_list = list(student_hidden_states)[::self.distill_student_step]  # Every 7th
            teacher_with_audio_list = list(teacher_hidden_states_with_audio)[::self.distill_teacher_step]  # Every 8th
            teacher_without_audio_list = list(teacher_hidden_states_without_audio)[::self.distill_teacher_step]

            # Align to same number of layers
            num_layers = min(len(student_hs_list), len(teacher_with_audio_list), len(teacher_without_audio_list))
            total_loss = 0.0

            for layer_idx in range(num_layers):
                student_hidden = student_hs_list[layer_idx]
                teacher_hidden_with_audio = teacher_with_audio_list[layer_idx]
                teacher_hidden_without_audio = teacher_without_audio_list[layer_idx]

                # Create layer key in CoD format (matches CoD exactly)
                teacher_dim = teacher_hidden_with_audio.size(-1)
                student_dim = student_hidden.size(-1)
                layer_key = f"layer_{layer_idx}_{teacher_dim}_to_{student_dim}"

                # Keep track of actual layer indices for logging
                s_idx = layer_idx * self.distill_student_step
                t_idx = layer_idx * self.distill_teacher_step

                layer_loss = self._compute_mse_single_layer(
                    student_hidden,
                    teacher_hidden_with_audio,
                    teacher_hidden_without_audio,
                    student_start_answer_positions,
                    teacher_start_answer_positions,
                    contrastive_beta,
                    layer_key=layer_key  # Use layer-specific projection
                )

                total_loss += layer_loss

                # Log individual layer losses for monitoring
                if self.training:
                    # hidden_states[i] corresponds to layer (i-1) for i>0, or embedding for i=0
                    if s_idx == 0:
                        layer_name = "embedding"
                    else:
                        layer_name = str(s_idx - 1)  # hidden_states[7] = layer 6
                    self.log(f"train/mse_layer_{layer_name}", layer_loss.item(),
                             sync_dist=True, batch_size=student_hidden.size(0))

            # Return average loss
            return total_loss / num_layers

        else:
            raise ValueError(f"Unknown distill_layer_mode: {self.distill_layer_mode}")

    # ---------------- Helper: Compute MSE Loss for Single Layer ----------------
    def _compute_mse_single_layer(
        self,
        student_hidden,
        teacher_hidden_with_audio,
        teacher_hidden_without_audio,
        student_start_answer_positions,
        teacher_start_answer_positions,
        contrastive_beta: float = 0.5,
        layer_key: str = "default"
    ):
        """
        Compute MSE contrastive loss for a single layer.

        Args:
            student_hidden: [batch_size, seq_len, hidden_dim]
            teacher_hidden_with_audio: [batch_size, seq_len, hidden_dim]
            teacher_hidden_without_audio: [batch_size, seq_len, hidden_dim]
            student_start_answer_positions: Starting positions for student answers
            teacher_start_answer_positions: Starting positions for teacher answers
            contrastive_beta: Weighting factor
            layer_key: Key to identify which projection layer to use

        Returns:
            loss: Scalar tensor representing the MSE loss
        """
        total_loss = 0.0
        valid_samples = 0
        batch_size = student_hidden.size(0)

        for i in range(batch_size):
            if i >= len(student_start_answer_positions) or i >= len(teacher_start_answer_positions):
                continue

            student_start = student_start_answer_positions[i].item()
            teacher_start = teacher_start_answer_positions[i].item()

            # Get representations after answer positions
            student_repr = student_hidden[i][student_start:]
            teacher_repr_with_audio = teacher_hidden_with_audio[i][teacher_start:]
            teacher_repr_without_audio = teacher_hidden_without_audio[i][teacher_start:]

            # Align sequence lengths
            max_length = min(
                student_repr.size(0),
                teacher_repr_with_audio.size(0),
                teacher_repr_without_audio.size(0)
            )
            if max_length == 0:
                continue

            student_repr = student_repr[:max_length]
            teacher_repr_with_audio = teacher_repr_with_audio[:max_length]
            teacher_repr_without_audio = teacher_repr_without_audio[:max_length]

            # Align dimensions using layer-specific projection
            student_repr, teacher_repr_with_audio = self._align_hidden_dimensions(
                student_repr, teacher_repr_with_audio, layer_key=layer_key
            )

            # OPTIMIZATION: When beta=0, skip projecting without_audio (it gets multiplied by 0)
            if contrastive_beta != 0:
                student_repr, teacher_repr_without_audio = self._align_hidden_dimensions(
                    student_repr, teacher_repr_without_audio, layer_key=layer_key
                )
                # Compute contrastive target: (1 + β) * with_audio - β * without_audio
                contrastive_target = (1 + contrastive_beta) * teacher_repr_with_audio - \
                                     contrastive_beta * teacher_repr_without_audio
            else:
                # FAST PATH: beta=0, contrastive_target = 1.0 * with_audio - 0.0 * without_audio = with_audio
                contrastive_target = teacher_repr_with_audio

            # MSE loss between student representation and contrastive target
            sample_loss = F.mse_loss(student_repr, contrastive_target, reduction='mean')

            total_loss += sample_loss
            valid_samples += 1

        # Average across valid samples
        if valid_samples > 0:
            return total_loss / valid_samples
        else:
            return torch.tensor(0.0, device=student_hidden.device, requires_grad=True)

    # ---------------- Standard KL Divergence Loss (no contrastive) ----------------
    def standard_kl_loss(
        self,
        student_logits,
        teacher_logits,
        student_start_answer_positions,
        teacher_start_answer_positions,
        temperature: float = 2.0
    ):
        """
        Standard KL divergence loss between student and teacher (with-audio only).
        No contrastive formula applied.
        """
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

    # ---------------- Logit Contrastive Loss (KL divergence on contrastive logits) ----------------
    def logit_contrastive_loss(
        self,
        student_logits,
        teacher_logits_with_audio,
        teacher_logits_without_audio,
        student_start_answer_positions,
        teacher_start_answer_positions,
        contrastive_beta: float = 0.5,
        temperature: float = 2.0
    ):
        """
        Contrastive loss on logits using KL divergence.
        Formula: KL(student_logits || (1+β)*with_audio - β*without_audio)

        The contrastive target is computed in logit space, then converted to probability
        distributions for KL divergence calculation.
        """
        total_loss = 0.0
        valid_samples = 0

        batch_size = min(len(student_logits), len(teacher_logits_with_audio), len(teacher_logits_without_audio))
        for i in range(batch_size):
            if i >= len(student_start_answer_positions) or i >= len(teacher_start_answer_positions):
                continue

            student_start = student_start_answer_positions[i].item()
            teacher_start = teacher_start_answer_positions[i].item()

            # Get logits after answer positions
            student_output_logits = student_logits[i][student_start:]
            teacher_output_logits_with_audio = teacher_logits_with_audio[i][teacher_start:]

            # OPTIMIZATION: When beta=0, skip slicing without_audio logits (unused)
            if contrastive_beta != 0:
                teacher_output_logits_without_audio = teacher_logits_without_audio[i][teacher_start:]

            # Align sequence lengths
            if contrastive_beta != 0:
                max_length = min(
                    student_output_logits.size(0),
                    teacher_output_logits_with_audio.size(0),
                    teacher_output_logits_without_audio.size(0)
                )
            else:
                # FAST PATH: beta=0, only need to align student and teacher_with_audio
                max_length = min(
                    student_output_logits.size(0),
                    teacher_output_logits_with_audio.size(0)
                )
            if max_length == 0:
                continue

            student_output_logits = student_output_logits[:max_length]
            teacher_output_logits_with_audio = teacher_output_logits_with_audio[:max_length]

            # OPTIMIZATION: When beta=0, skip slicing and computing contrastive formula
            if contrastive_beta != 0:
                teacher_output_logits_without_audio = teacher_output_logits_without_audio[:max_length]
                # Compute contrastive target logits: (1 + β) * with_audio - β * without_audio
                contrastive_target_logits = (1 + contrastive_beta) * teacher_output_logits_with_audio - \
                                            contrastive_beta * teacher_output_logits_without_audio
            else:
                # FAST PATH: beta=0, target = 1.0 * with_audio - 0.0 * without_audio = with_audio
                contrastive_target_logits = teacher_output_logits_with_audio

            # Convert to probability distributions with temperature scaling
            student_log_probs = F.log_softmax(student_output_logits / temperature, dim=-1)
            teacher_probs = F.softmax(contrastive_target_logits / temperature, dim=-1)

            # KL divergence loss with temperature scaling
            sample_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='batchmean'
            ) * (temperature ** 2)

            total_loss += sample_loss
            valid_samples += 1

        # Average across valid samples
        if valid_samples > 0:
            return total_loss / valid_samples
        else:
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)

    # ---------------- Helper: Compute Cross-Attention Loss for Single Layer ----------------
    def _compute_cross_attn_single_layer(
        self,
        student_hidden,
        teacher_hidden_with_audio,
        teacher_hidden_without_audio,
        student_start_answer_positions,
        teacher_start_answer_positions,
        contrastive_beta: float = 0.5,
        layer_key: str = "default"
    ):
        """
        Compute cross-attention contrastive loss for a single layer.

        Args:
            student_hidden: [batch_size, seq_len, hidden_dim]
            teacher_hidden_with_audio: [batch_size, seq_len, hidden_dim]
            teacher_hidden_without_audio: [batch_size, seq_len, hidden_dim]
            student_start_answer_positions: Starting positions for student answers
            teacher_start_answer_positions: Starting positions for teacher answers
            contrastive_beta: Weighting factor
            layer_key: Key to identify which projection layer to use (e.g., "s4_t4" for student layer 4 to teacher layer 4)

        Returns:
            loss: Scalar tensor representing the contrastive loss
        """
        total_loss = 0.0
        valid_samples = 0
        batch_size = student_hidden.size(0)

        for i in range(batch_size):
            if i >= len(student_start_answer_positions) or i >= len(teacher_start_answer_positions):
                continue

            student_start = student_start_answer_positions[i].item()
            teacher_start = teacher_start_answer_positions[i].item()

            # Get representations after answer positions
            student_repr = student_hidden[i][student_start:]
            teacher_repr_with_audio = teacher_hidden_with_audio[i][teacher_start:]
            teacher_repr_without_audio = teacher_hidden_without_audio[i][teacher_start:]

            # Check for empty sequences
            if student_repr.size(0) == 0 or teacher_repr_with_audio.size(0) == 0 or teacher_repr_without_audio.size(0) == 0:
                continue

            # Align dimensions if necessary (project teacher to student dimension)
            # Use layer-specific projection for multi-layer mode
            student_repr, teacher_repr_with_audio = self._align_hidden_dimensions(
                student_repr, teacher_repr_with_audio, layer_key=layer_key
            )

            # Compute similarity with positive pair (with audio)
            sim_pos = self._compute_aligned_similarity(student_repr, teacher_repr_with_audio)

            # OPTIMIZATION: When beta=0, skip projecting without_audio and computing sim_neg
            if contrastive_beta != 0:
                student_repr, teacher_repr_without_audio = self._align_hidden_dimensions(
                    student_repr, teacher_repr_without_audio, layer_key=layer_key
                )
                # Compute similarity with negative pair (without audio)
                sim_neg = self._compute_aligned_similarity(student_repr, teacher_repr_without_audio)
                # Direct contrastive loss: maximize (1+β)*sim_pos - β*sim_neg
                # Use negative sign because we minimize loss
                sample_loss = -((1 + contrastive_beta) * sim_pos - contrastive_beta * sim_neg)
            else:
                # FAST PATH: beta=0, loss = -1.0 * sim_pos - 0.0 * sim_neg = -sim_pos
                sample_loss = -sim_pos

            # Scale up to match magnitude with other losses (MSE, KL, etc.)
            # Without scaling, this loss is too small (-1.5 to +1.5) compared to other losses
            sample_loss = sample_loss * self.cross_attn_loss_scale

            total_loss += sample_loss
            valid_samples += 1

        # Average across valid samples
        if valid_samples > 0:
            return total_loss / valid_samples
        else:
            return torch.tensor(0.0, device=student_hidden.device, requires_grad=True)

    # ---------------- Cross-Attention Contrastive Loss (Direct Similarity Optimization) ----------------
    def cross_attention_contrastive_loss(
        self,
        student_hidden_states,
        teacher_hidden_states_with_audio,
        teacher_hidden_states_without_audio,
        student_start_answer_positions,
        teacher_start_answer_positions,
        contrastive_beta: float = 0.5
    ):
        """
        Cross-attention based contrastive loss using direct similarity optimization.
        Supports two modes:
          - "last": Only use final layer (original behavior)
          - "average": Use user-specified layers and average the loss

        Unlike MSE-based contrastive loss which operates on raw representations,
        this approach:
        1. Uses bidirectional cross-attention to compute aligned similarities
        2. Applies direct contrastive objective: -((1+β)*sim_pos - β*sim_neg)
        3. Simpler and more appropriate for 1-positive-1-negative setting

        Args:
            student_hidden_states: Student model hidden states (all layers)
            teacher_hidden_states_with_audio: Teacher hidden states WITH audio (positive)
            teacher_hidden_states_without_audio: Teacher hidden states WITHOUT audio (negative)
            student_start_answer_positions: Starting positions for student answers
            teacher_start_answer_positions: Starting positions for teacher answers
            contrastive_beta: Weighting factor for contrastive combination

        Returns:
            loss: Scalar tensor representing the contrastive loss

        Algorithm per sample:
            1. Extract student, teacher_pos, teacher_neg representations
            2. Compute sim_pos = similarity(student, teacher_pos) via cross-attention alignment
            3. Compute sim_neg = similarity(student, teacher_neg) via cross-attention alignment
            4. Loss = -((1+β)*sim_pos - β*sim_neg)
               Minimize loss → maximize (1+β)*sim_pos - β*sim_neg
               → Encourage high sim_pos, low sim_neg
        """
        if self.distill_layer_mode == "last":
            # Original behavior: only use final layer
            return self._compute_cross_attn_single_layer(
                student_hidden_states[-1],
                teacher_hidden_states_with_audio[-1],
                teacher_hidden_states_without_audio[-1],
                student_start_answer_positions,
                teacher_start_answer_positions,
                contrastive_beta
            )

        elif self.distill_layer_mode == "average":
            # Multi-layer mode: auto-sample layers using step approach (like CoD)
            student_hs_list = list(student_hidden_states)[::self.distill_student_step]  # Every 7th
            teacher_with_audio_list = list(teacher_hidden_states_with_audio)[::self.distill_teacher_step]  # Every 8th
            teacher_without_audio_list = list(teacher_hidden_states_without_audio)[::self.distill_teacher_step]

            # Align to same number of layers
            num_layers = min(len(student_hs_list), len(teacher_with_audio_list), len(teacher_without_audio_list))
            total_loss = 0.0

            for layer_idx in range(num_layers):
                student_hidden = student_hs_list[layer_idx]
                teacher_hidden_with_audio = teacher_with_audio_list[layer_idx]
                teacher_hidden_without_audio = teacher_without_audio_list[layer_idx]

                # Create layer key in CoD format (matches CoD exactly)
                teacher_dim = teacher_hidden_with_audio.size(-1)
                student_dim = student_hidden.size(-1)
                layer_key = f"layer_{layer_idx}_{teacher_dim}_to_{student_dim}"

                # Keep track of actual layer indices for logging
                s_idx = layer_idx * self.distill_student_step
                t_idx = layer_idx * self.distill_teacher_step

                layer_loss = self._compute_cross_attn_single_layer(
                    student_hidden,
                    teacher_hidden_with_audio,
                    teacher_hidden_without_audio,
                    student_start_answer_positions,
                    teacher_start_answer_positions,
                    contrastive_beta,
                    layer_key=layer_key  # Pass layer-specific key for projection
                )

                total_loss += layer_loss

                # Log individual layer losses for monitoring
                if self.training:
                    # hidden_states[i] corresponds to layer (i-1) for i>0, or embedding for i=0
                    if s_idx == 0:
                        layer_name = "embedding"
                    else:
                        layer_name = str(s_idx - 1)  # hidden_states[7] = layer 6
                    self.log(f"train/cross_attn_layer_{layer_name}", layer_loss.item(),
                             sync_dist=True, batch_size=student_hidden.size(0))

            # Return average loss
            return total_loss / num_layers

        else:
            raise ValueError(f"Unknown distill_layer_mode: {self.distill_layer_mode}")

    # ---------------- Entropy Calculation Methods ----------------
    def _compute_and_log_entropy(
        self,
        teacher_logits_with_audio,
        teacher_logits_without_audio,
        teacher_start_answer_positions,
        batch_size
    ):
        """
        Compute entropy for answer regions in the batch.

        For each sample in batch:
          1. Extract answer region (from answer_start to end)
          2. Compute entropy for each position in answer region
          3. Average across answer region → one entropy value per sample
          4. Store for epoch-level statistics

        Only runs during first epoch since teacher is frozen.

        Args:
            teacher_logits_with_audio: [batch, seq_len, vocab_size]
            teacher_logits_without_audio: [batch, seq_len, vocab_size]
            teacher_start_answer_positions: [batch] - Starting positions for answers
            batch_size: int - For logging
        """
        # Compute contrastive logits (same formula as in logit_contrastive_loss)
        if self.contrastive_beta != 0:
            # Apply contrastive formula: (1+β)*with_audio - β*without_audio
            contrastive_logits = (1 + self.contrastive_beta) * teacher_logits_with_audio - \
                                self.contrastive_beta * teacher_logits_without_audio
        else:
            # When β=0, contrastive = with_audio
            contrastive_logits = teacher_logits_with_audio

        # Process each sample in the batch
        batch_size_actual = teacher_logits_with_audio.size(0)
        for i in range(batch_size_actual):
            if i >= len(teacher_start_answer_positions):
                continue

            answer_start = teacher_start_answer_positions[i].item()

            # Extract answer region: [answer_length, vocab_size]
            answer_logits_with = teacher_logits_with_audio[i, answer_start:, :]
            answer_logits_contrast = contrastive_logits[i, answer_start:, :]

            # Skip if answer region is empty
            if answer_logits_with.size(0) == 0:
                continue

            # Compute average entropy across answer region
            entropy_with = self._compute_region_entropy(answer_logits_with)
            entropy_contrast = self._compute_region_entropy(answer_logits_contrast)

            # Store for epoch-level statistics (will be saved to file at epoch end)
            self.entropy_with_audio_samples.append(entropy_with)
            self.entropy_contrastive_samples.append(entropy_contrast)

    def _compute_region_entropy(self, logits):
        """
        Compute average entropy across sequence positions in a region using K-cand and K-agg.
        Computes for ALL K-agg values (1, 10, 20, all) simultaneously.

        Uses SAME K-cand and K-agg strategy as LogTokU for fair comparison:
        - K-cand: Compute entropy using only top-K logits (not full vocab)
        - K-agg: Select worst-K tokens (highest entropy) and average them

        Args:
            logits: [seq_len, vocab_size] - Logits for a region

        Returns:
            dict: Entropy values for each K-agg, keys {1, 10, 20, "all"}
        """
        seq_len, vocab_size = logits.shape
        entropy_per_position = []

        # Step 1: Compute entropy for EACH token position using K-cand
        for t in range(seq_len):
            position_logits = logits[t]  # [vocab_size]

            # Extract top-K-cand logits (same K as LogTokU uses for evidence)
            topk_logits, topk_indices = torch.topk(position_logits, k=min(self.topk_cand, vocab_size))

            # Compute entropy using only top-K-cand logits
            # Convert to probabilities (renormalize over top-K only)
            probs = F.softmax(topk_logits, dim=-1)

            # Compute entropy: H = -Σ p(x) * log(p(x))
            # Add small epsilon for numerical stability
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum().item()

            entropy_per_position.append(entropy)

        # Step 2: Aggregate using MULTIPLE K-agg values (same as LogTokU)
        # "use the most uncertain tokens in a sentence to represent the overall reliability"
        entropy_dict = {}

        for k_agg in self.kagg_values:
            if k_agg == "all":
                # Use all tokens
                avg_entropy = sum(entropy_per_position) / len(entropy_per_position) if entropy_per_position else 0.0
            else:
                # Select K-agg tokens with HIGHEST entropy (worst/most uncertain tokens)
                k = min(int(k_agg), len(entropy_per_position))

                if k == 0:
                    avg_entropy = 0.0
                else:
                    # Get indices of K-agg tokens with highest entropy
                    # Sort by entropy descending, take top K-agg
                    sorted_indices = sorted(range(len(entropy_per_position)),
                                          key=lambda i: entropy_per_position[i],
                                          reverse=True)[:k]

                    # Average entropy over only the K-agg worst tokens
                    avg_entropy = sum(entropy_per_position[i] for i in sorted_indices) / k

            # Store result for this K-agg value
            entropy_dict[k_agg] = avg_entropy

        return entropy_dict

    def _compute_eval_logtoku_per_sample(self, logits, answer_start_positions):
        """
        Compute AU (Aleatoric Uncertainty), EU (Epistemic Uncertainty), and
        combined uncertainty (AU*EU) for answer regions during EVALUATION using LogTokU framework.

        This computes uncertainty on the GROUND TRUTH answer region, not generated tokens.
        Implements the LogTokU framework from "Estimating LLM Uncertainty with Evidence".

        Important:
        - AU/EU: Single values (average over ALL tokens, only affected by K-cand)
        - Uncertainty: Multiple K-agg values (K-cand for per-token + K-agg for aggregation)
        - AU/EU sequences: Full per-token values for detailed analysis

        Args:
            logits: [batch, seq_len, vocab_size] - Student model logits
            answer_start_positions: [batch] - Starting positions for answers

        Returns:
            Tuple of (au_list, eu_list, uncertainty_list, au_seq_list, eu_seq_list, uncertainty_seq_list):
                - au_list: List of single AU values (one per sample)
                - eu_list: List of single EU values (one per sample)
                - uncertainty_list: List of dicts with keys {1, 10, 20, "all"} (one per sample)
                - au_seq_list: List of AU sequences (per-token values)
                - eu_seq_list: List of EU sequences (per-token values)
                - uncertainty_seq_list: List of uncertainty sequences (per-token values)
        """
        batch_size = logits.size(0)
        au_list = []
        eu_list = []
        uncertainty_list = []
        au_seq_list = []
        eu_seq_list = []
        uncertainty_seq_list = []

        for i in range(batch_size):
            if i >= len(answer_start_positions):
                au_list.append(None)
                eu_list.append(None)
                uncertainty_list.append(None)
                au_seq_list.append(None)
                eu_seq_list.append(None)
                uncertainty_seq_list.append(None)
                continue

            answer_start = answer_start_positions[i].item()

            # Extract answer region: [answer_length, vocab_size]
            answer_logits = logits[i, answer_start:, :]

            # Skip if answer region is empty
            if answer_logits.size(0) == 0:
                au_list.append(None)
                eu_list.append(None)
                uncertainty_list.append(None)
                au_seq_list.append(None)
                eu_seq_list.append(None)
                uncertainty_seq_list.append(None)
                continue

            # Compute AU (single), EU (single), uncertainty (multiple K-agg), and per-token sequences
            au_val, eu_val, uncertainty_dict, au_seq, eu_seq, uncertainty_seq = self._compute_region_au_eu(answer_logits)
            au_list.append(au_val)
            eu_list.append(eu_val)
            uncertainty_list.append(uncertainty_dict)
            au_seq_list.append(au_seq)
            eu_seq_list.append(eu_seq)
            uncertainty_seq_list.append(uncertainty_seq)

        return au_list, eu_list, uncertainty_list, au_seq_list, eu_seq_list, uncertainty_seq_list

    def _compute_region_au_eu(self, logits):
        """
        Compute AU (Aleatoric Uncertainty), EU (Epistemic Uncertainty), and
        combined uncertainty across sequence positions using LogTokU framework.

        Implements the evidence-based uncertainty estimation from:
        "Estimating LLM Uncertainty with Evidence" (LogTokU)

        Key formulas from the paper:
        Section 3.4 - Per-token AU/EU computation (using K-cand):
        - αk = M(τk|q, at-1)  # Top-K-cand logits as evidence
        - α0 = Σ αk           # Total evidence
        - AU(at) = -Σ(k=1 to K-cand) [αk/α0 · (ψ(αk + 1) - ψ(α0 + 1))]
        - EU(at) = K-cand / Σ(k=1 to K-cand) (αk + 1)

        Section 5.1 - Response-level aggregation (using K-agg, Equation 7):
        - R_response = (1/K-agg) Σ_{t∈T_{K-agg}} R(at)
        - where R(at) = -AU(at) * EU(at)
        - T_{K-agg} = set of K-agg tokens with HIGHEST uncertainty

        Important:
        - AU/EU: Averaged over ALL tokens (no K-agg selection)
        - Uncertainty: Aggregated using K-agg (select worst-K tokens)
        - Sequences: Full per-token values for detailed analysis

        Args:
            logits: [seq_len, vocab_size] - Logits for a region

        Returns:
            Tuple of (au_val, eu_val, uncertainty_dict, au_sequence, eu_sequence, uncertainty_sequence):
                - au_val: Single AU value (average over ALL tokens)
                - eu_val: Single EU value (average over ALL tokens)
                - uncertainty_dict: Dict with keys {1, 10, 20, "all"} with aggregated uncertainty
                - au_sequence: List of per-token AU values
                - eu_sequence: List of per-token EU values
                - uncertainty_sequence: List of per-token uncertainty values (AU * EU)
        """
        seq_len, vocab_size = logits.shape
        au_per_position = []
        eu_per_position = []
        uncertainty_per_position = []

        # Step 1: Compute AU and EU for EACH token position (using K-cand)
        for t in range(seq_len):
            position_logits = logits[t]  # [vocab_size]

            # Extract top-K-cand logits (evidence parameters)
            # Select top K-cand largest logits to model Dirichlet distribution
            topk_logits, topk_indices = torch.topk(position_logits, k=min(self.topk_cand, vocab_size))

            # Use logits as evidence (alpha parameters)
            # Following paper: αk = M(τk|q, at-1)
            alphas = topk_logits  # [K-cand]
            alpha_0 = alphas.sum()  # Total evidence

            # Compute AU (Aleatoric Uncertainty)
            # Formula: AU(at) = -Σ(k=1 to K-cand) [αk/α0 · (ψ(αk + 1) - ψ(α0 + 1))]
            au = self._compute_aleatoric_uncertainty(alphas, alpha_0)

            # Compute EU (Epistemic Uncertainty)
            # Formula: EU(at) = K-cand / Σ(k=1 to K-cand) (αk + 1)
            eu = self._compute_epistemic_uncertainty(alphas)

            # Compute combined uncertainty for this token
            uncertainty = au * eu

            au_per_position.append(au)
            eu_per_position.append(eu)
            uncertainty_per_position.append(uncertainty)

        # Step 2: Report AU/EU as single values (average over ALL tokens)
        # No K-agg selection for AU/EU - they are just intermediate values
        avg_au = sum(au_per_position) / len(au_per_position) if au_per_position else 0.0
        avg_eu = sum(eu_per_position) / len(eu_per_position) if eu_per_position else 0.0

        # Step 3: Aggregate uncertainty using MULTIPLE K-agg values (Equation 7 from paper)
        # "use the most uncertain tokens in a sentence to represent the overall reliability"
        # Compute for K-agg = [1, 10, 20, all] simultaneously
        uncertainty_dict = {}

        for k_agg in self.kagg_values:
            if k_agg == "all":
                # Use all tokens
                avg_uncertainty = sum(uncertainty_per_position) / len(uncertainty_per_position) if uncertainty_per_position else 0.0
            else:
                # Select K-agg tokens with HIGHEST uncertainty (worst/most unreliable tokens)
                k = min(int(k_agg), len(uncertainty_per_position))

                if k == 0:
                    avg_uncertainty = 0.0
                else:
                    # Get indices of K-agg tokens with highest uncertainty
                    # Sort by uncertainty descending, take top K-agg
                    sorted_indices = sorted(range(len(uncertainty_per_position)),
                                          key=lambda i: uncertainty_per_position[i],
                                          reverse=True)[:k]

                    # Average uncertainty over only the K-agg worst tokens
                    avg_uncertainty = sum(uncertainty_per_position[i] for i in sorted_indices) / k

            # Store result for this K-agg value
            uncertainty_dict[k_agg] = avg_uncertainty

        # Return both averaged values AND full per-token sequences
        return avg_au, avg_eu, uncertainty_dict, au_per_position, eu_per_position, uncertainty_per_position

    def _compute_aleatoric_uncertainty(self, alphas, alpha_0):
        """
        Compute Aleatoric Uncertainty (AU) using evidential deep learning.

        Formula from paper (Equation 4):
        AU(at) = -Σ(k=1 to K) [αk/α0 · (ψ(αk + 1) - ψ(α0 + 1))]

        This measures the expected entropy of the data distribution.
        - Lower entropy: model concentrates on single class
        - Higher entropy: more uniform distribution (model is undecided)

        Args:
            alphas: [K] - Top-K logits (evidence parameters)
            alpha_0: scalar - Sum of all alphas (total evidence)

        Returns:
            float: Aleatoric uncertainty value
        """
        K = len(alphas)

        # Convert to float32 for digamma (BFloat16 not supported)
        alphas = alphas.float()
        alpha_0 = alpha_0.float()

        # Compute digamma function values
        # ψ(x) = d/dx log Γ(x)
        digamma_alphas = torch.digamma(alphas + 1)  # ψ(αk + 1)
        digamma_alpha_0 = torch.digamma(alpha_0 + 1)  # ψ(α0 + 1)

        # Compute AU: -Σ [αk/α0 · (ψ(αk + 1) - ψ(α0 + 1))]
        au = -(alphas / alpha_0 * (digamma_alphas - digamma_alpha_0)).sum()

        return au.item()

    def _compute_epistemic_uncertainty(self, alphas):
        """
        Compute Epistemic Uncertainty (EU) using evidential deep learning.

        Formula from paper (Equation 5):
        EU(at) = K / Σ(k=1 to K) (αk + 1)

        This measures the model's inherent uncertainty (strength of evidence).
        - Larger αk values: sharper density, higher confidence
        - Higher EU: model lacks knowledge (low evidence strength)
        - Lower EU: model has strong evidence (seen many similar examples)

        Args:
            alphas: [K] - Top-K logits (evidence parameters)

        Returns:
            float: Epistemic uncertainty value
        """
        K = len(alphas)

        # Convert to float32 for numerical stability
        alphas = alphas.float()

        # Compute EU: K / Σ(αk + 1)
        eu = K / (alphas + 1).sum()

        return eu.item()

    # ---------------- Helper methods ----------------
    def _build_projection_layer(self, teacher_dim, student_dim, device, layer_name="default"):
        """
        Build linear projection layer: teacher_dim → student_dim
        Uses Xavier uniform initialization for stable training.

        Args:
            teacher_dim: Teacher hidden dimension
            student_dim: Student hidden dimension
            device: Device to place the layer on
            layer_name: Name of the layer (for logging purposes)
        """
        # Use PyTorch default initialization (Kaiming uniform) - matches CoD
        projection = torch.nn.Linear(teacher_dim, student_dim).to(device)
        return projection

    def _align_hidden_dimensions(self, student_hidden, teacher_hidden, layer_key="default"):
        """
        Align hidden dimensions between student and teacher (CoD-style on-demand creation).

        If dimensions match: No projection needed
        If dimensions differ: Project teacher → student using learned linear layer

        Args:
            student_hidden: [seq_len, student_hidden_dim]
            teacher_hidden: [seq_len, teacher_hidden_dim]
            layer_key: Key to identify which projection layer to use (for multi-layer mode)

        Returns:
            student_hidden: [seq_len, student_hidden_dim] (unchanged)
            teacher_hidden_aligned: [seq_len, student_hidden_dim] (projected if needed)
        """
        student_dim = student_hidden.size(-1)
        teacher_dim = teacher_hidden.size(-1)

        # Case 1: Dimensions already match - no projection needed
        if student_dim == teacher_dim:
            return student_hidden, teacher_hidden

        # Case 2: Dimensions differ - create projection on-demand (matches CoD approach)
        if layer_key not in self.projection_layers:
            # Create projection layer on-the-fly (CoD-style)
            self.projection_layers[layer_key] = self._build_projection_layer(
                teacher_dim=teacher_dim,
                student_dim=student_dim,
                device=student_hidden.device,
                layer_name=layer_key
            )

        # Project teacher hidden states to student dimension using layer-specific projection
        teacher_hidden_projected = self.projection_layers[layer_key](teacher_hidden)

        return student_hidden, teacher_hidden_projected

    def _compute_aligned_similarity(self, seq1, seq2):
        """
        Compute similarity between two sequences using bidirectional cross-attention alignment.

        This implements soft alignment using scaled dot-product attention, followed by
        cosine similarity measurement. The bidirectional approach ensures robustness
        regardless of which sequence is longer.

        Args:
            seq1: [len1, hidden_dim] - First sequence (typically student)
            seq2: [len2, hidden_dim] - Second sequence (typically teacher)

        Returns:
            similarity: scalar tensor - Average bidirectional cosine similarity

        Algorithm:
            1. Forward direction (seq1 → seq2):
               - Compute attention scores: seq1 @ seq2.T / sqrt(hidden_dim)
               - Apply softmax to get alignment weights
               - Align seq2 to seq1: aligned_seq2 = weights @ seq2
               - Compute cosine similarity between seq1 and aligned_seq2

            2. Backward direction (seq2 → seq1):
               - Repeat process in reverse direction

            3. Average both directions for symmetric similarity
        """
        hidden_dim = seq1.size(-1)

        # Forward: seq1 → seq2 (seq1 queries seq2)
        scores_1to2 = torch.matmul(seq1, seq2.T) / math.sqrt(hidden_dim)
        weights_1to2 = F.softmax(scores_1to2, dim=-1)  # [len1, len2]
        aligned_seq2 = torch.matmul(weights_1to2, seq2)  # [len1, hidden_dim]
        sim_1to2 = F.cosine_similarity(seq1, aligned_seq2, dim=-1).mean()

        # Backward: seq2 → seq1 (seq2 queries seq1)
        scores_2to1 = torch.matmul(seq2, seq1.T) / math.sqrt(hidden_dim)
        weights_2to1 = F.softmax(scores_2to1, dim=-1)  # [len2, len1]
        aligned_seq1 = torch.matmul(weights_2to1, seq1)  # [len2, hidden_dim]
        sim_2to1 = F.cosine_similarity(seq2, aligned_seq1, dim=-1).mean()

        # Average both directions for symmetric similarity
        return (sim_1to2 + sim_2to1) / 2

    def state_dict(self):
        """
        Override parent state_dict to ensure projection layers are included in checkpoints (CoD-style).
        """
        trainable_state_dict = super().state_dict()

        # Explicitly ensure projection layers are included (matches CoD exactly)
        for name, param in self.projection_layers.named_parameters():
            full_name = f"projection_layers.{name}"
            if param.requires_grad:
                trainable_state_dict[full_name] = param.data.clone().detach()

        return trainable_state_dict


# ============================================================================
# Example Config
# ============================================================================
#
# model:
#   kd_alpha: 0.5                    # Weight for KD loss vs NTP loss (0-1)
#   contrastive_beta: 0.5            # β for LOGIT contrastive: (1+β)*with_audio - β*without_audio
#   repr_contrastive_beta: 0.5       # β for REPR contrastive (if omitted, uses contrastive_beta)
#                                    # Set to 0.0 for NON-contrastive mode!
#   kd_temperature: 2.0              # Temperature for KL divergence (for standard_kd and logit_contrastive)
#
#   # Loss Component Flags (at least one must be true)
#   use_standard_kd: false           # Standard KL divergence (with-audio teacher only)
#   use_logit_contrastive: true      # Logit-level contrastive loss (KL divergence)
#   use_repr_contrastive: true       # Representation-level contrastive loss
#
#   # Representation Contrastive Loss Configuration
#   repr_contrastive_loss_type: "mse"           # Loss type: "mse" or "cross_attention"
#   cross_attn_loss_scale: 10.0                 # Scaling factor for cross-attention loss (default: 10.0)
#   distill_layer_mode: "average"               # "last" or "average" (multi-layer with auto-sampling)
#   distill_student_step: 7                     # Sample every 7th student layer (default: 7)
#   distill_teacher_step: 8                     # Sample every 8th teacher layer (default: 8)
#
#   # Notes on repr_contrastive_loss_type:
#   # - "mse": MSE-based contrastive loss with formula (1+β)*with_audio - β*without_audio
#              Operates directly on representations
#              NOW SUPPORTS multi-layer distillation with layer-specific projections!
#   # - "cross_attention": Similarity-based contrastive loss with cross-attention alignment
#                           Uses bidirectional attention + direct optimization: -((1+β)*sim_pos - β*sim_neg)
#                           Scaled by cross_attn_loss_scale to match magnitude with other losses
#
# ============================================================================
# Loss Combinations:
# ============================================================================
#
# NOTE: You can now control contrastive behavior separately for logits and representations!
#   - contrastive_beta: Controls logit contrastive weight
#   - repr_contrastive_beta: Controls representation contrastive weight
#   - Setting repr_contrastive_beta=0 → NON-contrastive repr loss (teacher with audio only)
#
# LOSS WEIGHTING (matches CoD):
#   - Logit losses (standard_kd OR logit_contrastive): 70% weight
#   - Representation loss (repr_contrastive): 30% weight
#   Total KD loss = (0.7 * logit_loss + 0.3 * repr_loss) / (0.7 + 0.3)
#
# 1. Pure Contrastive (RECOMMENDED):
#    use_standard_kd: false
#    use_logit_contrastive: true
#    use_repr_contrastive: true
#    contrastive_beta: 0.5
#    repr_contrastive_beta: 0.5
#    → Logit contrastive (70%) + Repr contrastive (30%)
#
# 1b. Logit Contrastive + Repr NON-Contrastive (NEW!):
#    use_standard_kd: false
#    use_logit_contrastive: true
#    use_repr_contrastive: true
#    contrastive_beta: 0.5
#    repr_contrastive_beta: 0.0  # NON-contrastive!
#    → Logit contrastive (70%) + NON-contrastive repr (30%, like CoD's layer_level_loss)
#
# 2. Standard KD + Representation Contrastive:
#    use_standard_kd: true
#    use_logit_contrastive: false
#    use_repr_contrastive: true
#    → Standard KL on logits (70%) + Repr contrastive (30%)
#
# 3. Logit Contrastive Only:
#    use_standard_kd: false
#    use_logit_contrastive: true
#    use_repr_contrastive: false
#    → Only logit contrastive (100% weight, no repr loss)
#
# 4. Representation Contrastive Only:
#    use_standard_kd: false
#    use_logit_contrastive: false
#    use_repr_contrastive: true
#    → Only repr contrastive (100% weight, no logit loss)
#
# 5. All Three (experimental):
#    use_standard_kd: true
#    use_logit_contrastive: true
#    use_repr_contrastive: true
#    → Standard KD (70%) + Logit contrastive (70%) + Repr contrastive (30%)
#       Note: Both logit losses get 70% each, totaling (0.7 + 0.7 + 0.3) = 1.7
#       Final weights: standard_kd=41%, logit_contrastive=41%, repr=18%
#
# ============================================================================
# Representation Contrastive Loss Types:
# ============================================================================
#
# MSE-based (Default):
#   - Formula: MSE(student, (1+β)*teacher_with_audio - β*teacher_without_audio)
#   - Operates directly on raw representations
#   - Creates a combined teacher target representation
#   - Simple and stable
#   - Good for when student and teacher have similar architectures
#
# Cross-Attention (Similarity-based):
#   - Uses bidirectional cross-attention alignment + direct contrastive optimization
#   - Algorithm:
#     1. Align student with teacher_with_audio via cross-attention → sim_pos
#     2. Align student with teacher_without_audio via cross-attention → sim_neg
#     3. Loss = -((1+β)*sim_pos - β*sim_neg) * scale
#        → Maximize (1+β)*sim_pos - β*sim_neg
#        → High similarity with positive (with audio), low with negative (without audio)
#   - Advantages:
#     * Handles variable sequence lengths elegantly via soft alignment
#     * Operates in similarity space rather than representation space
#     * More appropriate for 1-positive-1-negative setting
#     * Learns soft alignment between student and teacher sequences
#     * Better for sequences with different lengths or structures
#   - Key difference from MSE:
#     * MSE: Direct regression on representations
#     * Cross-Attention: Optimization on aligned similarities
#   - Scaling factor (cross_attn_loss_scale):
#     * Default: 10.0
#     * Purpose: Cross-attention loss is small (-1.5 to +1.5) compared to MSE/KL (10-1000)
#     * Without scaling, cross-attention loss gets dominated by other losses
#     * Tune this to balance loss magnitudes during training
#     * Higher values = stronger influence of cross-attention loss
#
# Recommendation:
#   - Start with "mse" for simplicity and stability
#   - Try "cross_attention" if:
#     * Student and teacher have very different architectures
#     * Sequence lengths vary significantly between samples
#     * You want explicit handling of sequence alignment
#     * MSE-based approach plateaus in performance
#
# ============================================================================

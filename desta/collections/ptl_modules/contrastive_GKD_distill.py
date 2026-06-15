from desta.collections.ptl_modules.desta3_ptl import DeSTA3PTLModule
from desta.collections.desta3.models.modeling_desta3 import DeSTA3Model, DeSTA3Config
import torch
from typing import Optional, Tuple, Union, Literal
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
import math


### GKD + Contrastive Distillation Module
###
### Combines two powerful techniques:
### 1. GKD (Generalized Knowledge Distillation): On-policy learning where student generates
###    sequences and teacher provides feedback on those student-generated sequences
### 2. Contrastive Learning: Teacher provides feedback in two scenarios (with/without audio)
###    creating positive and negative pairs
###
### Key Innovation:
### - Student generates continuation tokens (on-policy)
### - Teacher evaluates student's generation with TWO forward passes:
###   * WITH audio features (positive pair) → logits_pos
###   * WITHOUT audio features (negative pair) → logits_neg
### - Contrastive target: (1+β)*logits_pos - β*logits_neg
### - Student learns to match this contrastive distribution using reverse KL or JSD
###
### Reference: "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes"
###            https://arxiv.org/abs/2306.13649

class ContrastiveGKDPTLModule(DeSTA3PTLModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Initialize teacher model
        teacher_model_config = DeSTA3Config(
            llm_model_id=self.cfg.teacher_model.llm.model_id,
            encoder_model_id=self.cfg.teacher_model.encoder.model_id,
            connector_mode=self.cfg.teacher_model.connector.mode,
            qformer_num_hidden_layers=self.cfg.teacher_model.connector.num_hidden_layers,
            prompt_size=self.cfg.teacher_model.connector.prompt_size,
            first_n_layers=self.cfg.teacher_model.llm.first_n_layers if hasattr(self.cfg.teacher_model.llm, "first_n_layers") else -1,
        )

        print("="*100)
        print("Initializing GKD + Contrastive Distillation Module")
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

        # ========== GKD Configuration ==========
        # Core GKD parameters
        self.gkd_lambda = getattr(self.cfg.model, "gkd_lambda", 1.0)  # λ: 1.0=pure on-policy, 0.0=pure ground-truth
        self.gkd_divergence = getattr(self.cfg.model, "gkd_divergence", "reverse_kl")  # "forward_kl", "reverse_kl", "jsd"

        # ========== Contrastive Configuration ==========
        self.kd_alpha = self.cfg.model.kd_alpha  # Weight for total KD loss vs NTP loss
        self.contrastive_beta = getattr(self.cfg.model, "contrastive_beta", 0.5)  # β in contrastive formula
        self.kd_temperature = getattr(self.cfg.model, "kd_temperature", 2.0)

        # Validation
        if not 0.0 <= self.gkd_lambda <= 1.0:
            raise ValueError(f"gkd_lambda must be in [0.0, 1.0], got {self.gkd_lambda}")
        if self.gkd_divergence not in ["forward_kl", "reverse_kl", "jsd"]:
            raise ValueError(f"gkd_divergence must be 'forward_kl', 'reverse_kl', or 'jsd', got '{self.gkd_divergence}'")

        print(f"\nGKD + Contrastive Distillation Config:")
        print(f"  === GKD Parameters ===")
        print(f"    gkd_lambda: {self.gkd_lambda} (1.0=pure on-policy, 0.0=pure ground-truth)")
        print(f"    gkd_divergence: {self.gkd_divergence}")
        print(f"  === Contrastive Parameters ===")
        print(f"    kd_alpha: {self.kd_alpha}")
        print(f"    contrastive_beta: {self.contrastive_beta}")
        print(f"    kd_temperature: {self.kd_temperature}")
        print(f"  === Generation Parameters (training & validation) ===")
        gen_kwargs = self.cfg.model.generation_kwargs
        print(f"    max_new_tokens: {gen_kwargs.get('max_new_tokens', 'N/A')}")
        print(f"    do_sample: {gen_kwargs.get('do_sample', 'N/A')}")
        print(f"    temperature: {gen_kwargs.get('temperature', 'N/A')}")
        print(f"    top_p: {gen_kwargs.get('top_p', 'N/A')}")
        print()

    # ============================================================================
    # Core Forward Pass: GKD + Contrastive Learning
    # ============================================================================
    def forward(self, batch):
        """
        Forward pass implementing GKD + Contrastive Learning.

        Algorithm:
        1. Student forward on ground-truth (compute NTP loss)
        2. [If on-policy] Student generates continuation tokens
        3. Teacher evaluates student's output with TWO forward passes:
           - WITH audio features (positive pair) → logits_pos
           - WITHOUT audio features (negative pair) → logits_neg
        4. Compute contrastive target: (1+β)*logits_pos - β*logits_neg
        5. Compute GKD loss using specified divergence (reverse KL, JSD, etc.)
        6. Combine NTP loss and GKD loss
        """
        # === Unpack batch ===
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        batch_features = batch["batch_features"]
        batch_transcription_ids = batch["batch_transcription_ids"]
        batch_start_positions = batch["batch_start_positions"]
        labels = batch.get("labels", None)

        # ========== Step 1: Student forward on ground-truth ==========
        # This provides the NTP (Next Token Prediction) loss
        student_input_embeds = self.model._prepare_inputs_for_llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            batch_features=batch_features,
            batch_transcription_ids=batch_transcription_ids,
            batch_start_positions=batch_start_positions,
        )

        student_output = self.model.llm_model(
            inputs_embeds=student_input_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=False,
            use_cache=False,
        )

        if self.training:
            self.log("train/ntp_loss", student_output.loss.item(),
                     sync_dist=True, batch_size=input_ids.size(0))

        # ========== Step 2-6: GKD + Contrastive Learning ==========
        if self.training:
            # Decide whether to use on-policy or ground-truth based on lambda
            use_on_policy = (self.gkd_lambda > 0.0) and (torch.rand(1).item() < self.gkd_lambda)

            if use_on_policy:
                # ========== GKD: On-Policy Generation ==========
                # Student generates continuation, teacher evaluates student's output
                gkd_loss = self._compute_gkd_contrastive_loss_on_policy(
                    batch=batch,
                    student_output_logits=student_output.logits
                )
            else:
                # ========== Standard: Ground-truth Evaluation ==========
                # Teacher evaluates ground-truth continuation (original contrastive distillation)
                gkd_loss = self._compute_gkd_contrastive_loss_ground_truth(
                    batch=batch,
                    student_output_logits=student_output.logits
                )

            self.log("train/gkd_contrastive_loss", gkd_loss.item(),
                     sync_dist=True, batch_size=input_ids.size(0))

            # Final loss: NTP + GKD Contrastive
            student_output.loss = (1 - self.kd_alpha) * student_output.loss + self.kd_alpha * gkd_loss

        return student_output

    # ============================================================================
    # GKD Contrastive Loss: On-Policy (Student Generates)
    # ============================================================================
    def _compute_gkd_contrastive_loss_on_policy(self, batch, student_output_logits):
        """
        Compute GKD contrastive loss using ON-POLICY student-generated sequences.

        This is the core GKD innovation:
        1. Student generates continuation tokens autoregressively
        2. Teacher evaluates student's generation with contrastive pairs (with/without audio)
        3. Student learns from teacher's feedback on its own output distribution

        Args:
            batch: Input batch
            student_output_logits: Student's logits on ground-truth (for reference)

        Returns:
            loss: GKD contrastive loss (scalar)
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        batch_features = batch["batch_features"]
        batch_transcription_ids = batch["batch_transcription_ids"]
        batch_start_positions = batch["batch_start_positions"]
        student_start_answer_positions = batch["audio_start_answer_positions"]

        batch_size = input_ids.size(0)

        # ========== Step 1: Student generates continuation ==========
        # Generate from the answer start position (after the question/prompt)
        with torch.no_grad():
            # Prepare student input embeddings
            student_input_embeds = self.model._prepare_inputs_for_llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                batch_features=batch_features,
                batch_transcription_ids=batch_transcription_ids,
                batch_start_positions=batch_start_positions,
            )

            # Generate student sequences using generation_kwargs from config
            gen_kwargs = self.cfg.model.generation_kwargs
            generated_outputs = self.model.llm_model.generate(
                inputs_embeds=student_input_embeds,
                attention_mask=attention_mask,
                max_new_tokens=gen_kwargs.get("max_new_tokens", 128),
                do_sample=gen_kwargs.get("do_sample", True),
                temperature=gen_kwargs.get("temperature", 1.0),
                top_p=gen_kwargs.get("top_p", 1.0),
                top_k=gen_kwargs.get("top_k", 0),
                pad_token_id=self.tokenizer.eos_token_id,  # Use tokenizer's eos as pad (set in parent class)
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=False,  # Just return sequences (logits not needed)
                use_cache=True,
            )

            # generated_outputs shape: [batch_size, original_seq_len + new_tokens]
            # We need to extract the generated part only
            original_seq_len = input_ids.size(1)
            student_generated_ids = generated_outputs[:, original_seq_len:]  # [batch_size, new_tokens]

        # ========== Step 2: Prepare evaluation sequence ==========
        # Create full sequence: prefix + student_generated_continuation
        # For each sample, we need to evaluate teacher on student's generation

        # Use input_ids as prefix up to answer start position
        # Then concatenate student's generated tokens
        evaluation_sequences = []
        evaluation_attention_masks = []

        for i in range(batch_size):
            # Get answer start position for this sample
            answer_start = student_start_answer_positions[i].item()

            # Prefix: original input up to answer start
            prefix = input_ids[i, :answer_start]

            # Student's generated continuation
            continuation = student_generated_ids[i]

            # Concatenate: prefix + continuation
            eval_seq = torch.cat([prefix, continuation], dim=0)

            # Create attention mask (all 1s for now, can refine if needed)
            eval_attn = torch.ones_like(eval_seq)

            evaluation_sequences.append(eval_seq)
            evaluation_attention_masks.append(eval_attn)

        # Pad sequences to same length
        max_eval_len = max(seq.size(0) for seq in evaluation_sequences)
        eval_input_ids = torch.full(
            (batch_size, max_eval_len),
            self.tokenizer.eos_token_id,  # Use tokenizer's eos as pad (set in parent class)
            dtype=input_ids.dtype,
            device=input_ids.device
        )
        eval_attention_mask = torch.zeros(
            (batch_size, max_eval_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )

        for i, (seq, attn) in enumerate(zip(evaluation_sequences, evaluation_attention_masks)):
            eval_input_ids[i, :seq.size(0)] = seq
            eval_attention_mask[i, :attn.size(0)] = attn

        # ========== Step 3: Teacher evaluates with contrastive pairs ==========
        with torch.no_grad():
            # Teacher forward WITH audio (positive pair)
            teacher_input_embeds_with_audio = self.teacher_model._prepare_inputs_for_llm(
                input_ids=eval_input_ids,
                attention_mask=eval_attention_mask,
                batch_features=batch_features,
                batch_transcription_ids=batch_transcription_ids,
                batch_start_positions=batch_start_positions,
            )

            teacher_output_with_audio = self.teacher_model.llm_model(
                inputs_embeds=teacher_input_embeds_with_audio,
                attention_mask=eval_attention_mask,
                output_hidden_states=False,
                use_cache=False,
            )

            # Teacher forward WITHOUT audio (negative pair)
            # Use same preparation pipeline but with zero audio features to ensure sequence length matches
            teacher_input_embeds_without_audio = self.teacher_model._prepare_inputs_for_llm(
                input_ids=eval_input_ids,
                attention_mask=eval_attention_mask,
                batch_features=torch.zeros_like(batch_features),  # Zero audio features
                batch_transcription_ids=batch_transcription_ids,
                batch_start_positions=batch_start_positions,
            )

            teacher_output_without_audio = self.teacher_model.llm_model(
                inputs_embeds=teacher_input_embeds_without_audio,
                attention_mask=eval_attention_mask,
                output_hidden_states=False,
                use_cache=False,
            )

        # ========== Step 4: Compute contrastive target logits ==========
        # Formula: contrastive_logits = (1 + β) * logits_with_audio - β * logits_without_audio
        contrastive_target_logits = (1 + self.contrastive_beta) * teacher_output_with_audio.logits - \
                                     self.contrastive_beta * teacher_output_without_audio.logits

        # ========== Step 5: Get student logits on the SAME evaluation sequence ==========
        # We need student's logits on the evaluation sequence (student's own generation)
        # IMPORTANT: This forward pass is necessary for gradient flow!
        # The logits from .generate() are inside no_grad and can't backprop.
        # This forward pass creates a differentiable path for training the student.
        student_eval_embeds = self.model._prepare_inputs_for_llm(
            input_ids=eval_input_ids,
            attention_mask=eval_attention_mask,
            batch_features=batch_features,
            batch_transcription_ids=batch_transcription_ids,
            batch_start_positions=batch_start_positions,
        )

        student_eval_output = self.model.llm_model(
            inputs_embeds=student_eval_embeds,
            attention_mask=eval_attention_mask,
            output_hidden_states=False,
            use_cache=False,
        )

        # ========== Step 6: Compute GKD divergence loss ==========
        gkd_loss = self._compute_divergence_loss(
            student_logits=student_eval_output.logits,  # Differentiable logits with gradient!
            teacher_logits=contrastive_target_logits,
            student_start_answer_positions=student_start_answer_positions,
            attention_mask=eval_attention_mask,
            divergence_type=self.gkd_divergence,
            temperature=self.kd_temperature
        )

        return gkd_loss

    # ============================================================================
    # GKD Contrastive Loss: Ground-Truth (Standard Evaluation)
    # ============================================================================
    def _compute_gkd_contrastive_loss_ground_truth(self, batch, student_output_logits):
        """
        Compute GKD contrastive loss using GROUND-TRUTH sequences.

        This is the standard contrastive distillation (non-on-policy):
        1. Use ground-truth continuation from the dataset
        2. Teacher evaluates ground-truth with contrastive pairs (with/without audio)
        3. Student learns from teacher's feedback on ground-truth distribution

        This provides a baseline and allows mixing with on-policy via lambda parameter.

        Args:
            batch: Input batch with ground-truth labels
            student_output_logits: Student's logits on ground-truth

        Returns:
            loss: GKD contrastive loss on ground-truth (scalar)
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        batch_features = batch["batch_features"]
        batch_transcription_ids = batch["batch_transcription_ids"]
        batch_start_positions = batch["batch_start_positions"]
        student_start_answer_positions = batch["audio_start_answer_positions"]
        teacher_start_answer_positions = batch["audio_start_answer_positions"]

        # Teacher evaluates ground-truth sequence with contrastive pairs
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
                output_hidden_states=False,
                use_cache=False,
            )

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
                output_hidden_states=False,
                use_cache=False,
            )

        # Compute contrastive target logits
        contrastive_target_logits = (1 + self.contrastive_beta) * teacher_output_with_audio.logits - \
                                     self.contrastive_beta * teacher_output_without_audio.logits

        # Compute divergence loss
        gkd_loss = self._compute_divergence_loss(
            student_logits=student_output_logits,
            teacher_logits=contrastive_target_logits,
            student_start_answer_positions=student_start_answer_positions,
            attention_mask=attention_mask,
            divergence_type=self.gkd_divergence,
            temperature=self.kd_temperature
        )

        return gkd_loss

    # ============================================================================
    # Divergence Loss Computation (Forward KL, Reverse KL, JSD)
    # ============================================================================
    def _compute_divergence_loss(
        self,
        student_logits,
        teacher_logits,
        student_start_answer_positions,
        attention_mask,
        divergence_type: Literal["forward_kl", "reverse_kl", "jsd"] = "reverse_kl",
        temperature: float = 2.0
    ):
        """
        Compute divergence loss between student and teacher logits.

        Supports three divergence types (from GKD paper):
        1. Forward KL: KL(teacher || student) - Standard knowledge distillation
        2. Reverse KL: KL(student || teacher) - GKD default, encourages mode-seeking
        3. JSD: Jensen-Shannon Divergence - Symmetric, balanced approach

        Args:
            student_logits: [batch_size, seq_len, vocab_size]
            teacher_logits: [batch_size, seq_len, vocab_size]
            student_start_answer_positions: Starting positions for computing loss
            attention_mask: Attention mask to ignore padding
            divergence_type: Type of divergence to compute
            temperature: Temperature for softmax (default: 2.0)

        Returns:
            loss: Scalar divergence loss
        """
        total_loss = 0.0
        valid_samples = 0

        batch_size = student_logits.size(0)

        for i in range(batch_size):
            if i >= len(student_start_answer_positions):
                continue

            answer_start = student_start_answer_positions[i].item()

            # Get logits after answer start position
            student_output_logits = student_logits[i][answer_start:]
            teacher_output_logits = teacher_logits[i][answer_start:]
            sample_attention = attention_mask[i][answer_start:]

            # Filter out padding positions
            valid_positions = sample_attention.bool()
            if valid_positions.sum() == 0:
                continue

            student_output_logits = student_output_logits[valid_positions]
            teacher_output_logits = teacher_output_logits[valid_positions]

            if student_output_logits.size(0) == 0:
                continue

            # Compute distributions
            student_probs = F.softmax(student_output_logits / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_output_logits / temperature, dim=-1)
            student_log_probs = F.log_softmax(student_output_logits / temperature, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_output_logits / temperature, dim=-1)

            # Compute divergence based on type
            if divergence_type == "forward_kl":
                # Forward KL: KL(teacher || student) = sum(teacher * log(teacher / student))
                # This is standard knowledge distillation
                sample_loss = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction='batchmean'
                ) * (temperature ** 2)

            elif divergence_type == "reverse_kl":
                # Reverse KL: KL(student || teacher) = sum(student * log(student / teacher))
                # GKD's default choice - encourages mode-seeking behavior
                sample_loss = F.kl_div(
                    teacher_log_probs,
                    student_probs,
                    reduction='batchmean'
                ) * (temperature ** 2)

            elif divergence_type == "jsd":
                # Jensen-Shannon Divergence: JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
                # where M = 0.5 * (P + Q)
                # Symmetric and bounded divergence
                mixture_probs = 0.5 * (student_probs + teacher_probs)
                mixture_log_probs = torch.log(mixture_probs + 1e-10)

                kl_student_mixture = F.kl_div(
                    mixture_log_probs,
                    student_probs,
                    reduction='batchmean'
                )
                kl_teacher_mixture = F.kl_div(
                    mixture_log_probs,
                    teacher_probs,
                    reduction='batchmean'
                )

                sample_loss = 0.5 * (kl_student_mixture + kl_teacher_mixture) * (temperature ** 2)

            else:
                raise ValueError(f"Unknown divergence type: {divergence_type}")

            total_loss += sample_loss
            valid_samples += 1

        # Average across valid samples
        if valid_samples > 0:
            return total_loss / valid_samples
        else:
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)

    # ============================================================================
    # PTL Hooks
    # ============================================================================
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


# ============================================================================
# Configuration Example
# ============================================================================
#
# model:
#   # === GKD Parameters ===
#   gkd_lambda: 1.0                  # λ: Probability of using on-policy vs ground-truth
#                                    #    1.0 = pure on-policy (student generates all continuations)
#                                    #    0.5 = mixed (50% student-generated, 50% ground-truth)
#                                    #    0.0 = pure ground-truth (standard distillation)
#
#   gkd_divergence: "reverse_kl"     # Divergence type: "forward_kl", "reverse_kl", "jsd"
#                                    #    - forward_kl: Standard KD (KL(teacher || student))
#                                    #    - reverse_kl: GKD default (KL(student || teacher)), mode-seeking
#                                    #    - jsd: Jensen-Shannon Divergence, symmetric
#
#   # Student generation parameters (for on-policy sampling)
#   generation_max_new_tokens: 128   # Maximum tokens to generate
#   generation_temperature: 1.0      # Sampling temperature (higher = more random)
#   generation_top_p: 0.95           # Nucleus sampling threshold
#   generation_top_k: 50             # Top-k sampling (0 = disabled)
#   generation_do_sample: true       # Use sampling (true) or greedy (false)
#
#   # === Contrastive Parameters ===
#   kd_alpha: 0.5                    # Weight for KD loss vs NTP loss (0-1)
#   contrastive_beta: 0.5            # β in contrastive formula: (1+β)*with_audio - β*without_audio
#   kd_temperature: 2.0              # Temperature for divergence computation
#
# ============================================================================
# How GKD + Contrastive Works:
# ============================================================================
#
# Traditional Contrastive Distillation:
#   - Teacher evaluates ground-truth continuations
#   - Student learns to mimic teacher on fixed dataset
#   - Issue: Distribution mismatch - student trained on teacher's distribution,
#            but at inference sees its own distribution
#
# GKD + Contrastive (This Implementation):
#   - Student GENERATES its own continuations (on-policy)
#   - Teacher evaluates student's generations with contrastive pairs (with/without audio)
#   - Student learns from teacher's feedback on its OWN output distribution
#   - Benefits:
#     * Reduces distribution mismatch
#     * Student learns from its own mistakes
#     * Contrastive signal helps student leverage audio information
#     * Can mix on-policy and ground-truth via lambda parameter
#
# Example Training Iteration:
#   1. Student sees prompt: "Describe the audio: [AUDIO]"
#   2. Student generates: "Describe the audio: [AUDIO] This is a dog barking loudly"
#   3. Teacher evaluates student's generation:
#      - WITH audio: "Yes, this matches the audio" → logits_pos (high probability)
#      - WITHOUT audio: "Hmm, could be anything" → logits_neg (low probability)
#   4. Contrastive target: (1+β)*logits_pos - β*logits_neg
#      → Encourages student to rely on audio, not just language priors
#   5. Student adjusts to match contrastive target using reverse KL
#      → Mode-seeking behavior: student focuses on teacher's high-probability modes
#
# ============================================================================
# Key Differences from Standard Implementations:
# ============================================================================
#
# vs. Standard KD:
#   - Standard KD: Teacher evaluates ground-truth, student mimics (forward KL)
#   - GKD+Contrastive: Teacher evaluates student's generations with contrastive pairs
#                      (reverse KL + contrastive audio grounding)
#
# vs. Original Contrastive Distillation:
#   - Original: Teacher evaluates ground-truth with/without audio
#   - GKD+Contrastive: Teacher evaluates STUDENT-GENERATED text with/without audio
#                      → Student learns what happens when IT makes mistakes
#
# vs. Pure GKD:
#   - Pure GKD: Single teacher evaluation (no contrastive)
#   - GKD+Contrastive: Dual teacher evaluation (with/without audio)
#                      → Student learns to ground language in audio perception
#
# ============================================================================

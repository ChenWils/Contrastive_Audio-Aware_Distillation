import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from apex.optimizers import FusedAdam
from desta.collections.desta3.models.modeling_desta3 import DeSTA3Model, DeSTA3Config
from transformers import get_cosine_schedule_with_warmup
import logging
from desta.collections.desta3.data.simple_dataset import BaseAudioTextDataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoFeatureExtractor
from whisper_normalizer.basic import BasicTextNormalizer
import os
from typing import Dict, List
import json
from omegaconf import OmegaConf
from collections import OrderedDict, defaultdict
from pathlib import Path
from lulutils import get_unique_filepath
from desta.collections.utils.metrics import ConsecutiveWordsAccuracyMetric
from desta.collections.desta3.data.sampler import DistributedMaxLengthBatchSampler, MaxLengthBatchSampler, LengthBasedBatchSampler, DistributedLengthBasedBatchSampler
import traceback
class DeSTA3PTLModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        
        # PTL
        self.automatic_optimization = True

        model_config = DeSTA3Config(
            llm_model_id=self.cfg.model.llm.model_id,
            encoder_model_id=self.cfg.model.encoder.model_id,
            connector_mode=self.cfg.model.connector.mode,
            qformer_num_hidden_layers=self.cfg.model.connector.num_hidden_layers,
            prompt_size=self.cfg.model.connector.prompt_size,
            first_n_layers=self.cfg.model.llm.first_n_layers if hasattr(self.cfg.model.llm, "first_n_layers") else -1,
        )

        print("="*100)
        self.model = DeSTA3Model(model_config)

        # remove whisper decoder during PTL training (we only use Whisper decoder during inference)
        del self.model.perception.whisper.model.decoder
        del self.model.perception.whisper.proj_out

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.llm.model_id, cache_dir=os.getenv("HF_HOME"))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.tokenizer.add_tokens([self.cfg.dataset.audio_locator])
        
        self.processor = AutoFeatureExtractor.from_pretrained(self.cfg.model.encoder.model_id, cache_dir=os.getenv("HF_HOME"))

        self.metrics = ConsecutiveWordsAccuracyMetric()
        
        self.prediction_step_outputs = []

    def forward(self, batch):
        return self.model(**batch)

    
    def on_train_epoch_begin(self):
        logging.info(self.model.dtype)

    # Training / Validation / Prediction
    def training_step(self, batch, batch_idx):
        self.model.train()
        try:
            outputs = self(batch)
            loss = outputs.loss
            
        except Exception as e:
            logging.error(f"Error in training step: {e}")
            logging.error(traceback.format_exc())
            logging.error(f"Batch: {batch}")
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        perplexity = torch.exp(loss)
        batch_size = batch["input_ids"].size(0)
        self.log("train/loss", loss, prog_bar=True, rank_zero_only=True, sync_dist=True, batch_size=batch_size)
        self.log("train/ppl", perplexity, prog_bar=True, rank_zero_only=True, sync_dist=True, batch_size=batch_size)
        
        return loss


    
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        loss = 0
        perplexity = 0
        predictions = []

        outputs = self(batch)
        loss = outputs.loss
        perplexity = torch.exp(loss)

        predictions = self.predict_step(batch, batch_idx)

        batch_size = batch["input_ids"].size(0)
        
        self.log("val/loss", loss.item(), sync_dist=True, batch_size=batch_size)
        self.log("val/ppl", perplexity.item(), sync_dist=True, batch_size=batch_size)


        return {"val/loss": loss, "val/ppl": perplexity, "predictions": predictions}


    def predict_step(self, batch, batch_idx):
        self.model.eval()

        # Check if contrastive decoding is enabled
        use_contrastive = getattr(self.cfg.model, 'use_contrastive_decoding', False)

        if use_contrastive:
            contrastive_alpha = getattr(self.cfg.model, 'contrastive_alpha', 1.0)
            generated_ids = self.model._generate_step_contrastive(
                batch,
                pad_token_id=self.tokenizer.eos_token_id,
                generation_kwargs=self.cfg.model.generation_kwargs,
                contrastive_alpha=contrastive_alpha
            )
        else:
            generated_ids = self.model._generate_step(
                batch,
                pad_token_id=self.tokenizer.eos_token_id,
                generation_kwargs=self.cfg.model.generation_kwargs
            )


        batch["context_input_ids"][batch["context_input_ids"] == -100] = self.tokenizer.eos_token_id
        batch["labels"][batch["labels"] == -100] = self.tokenizer.eos_token_id
        generated_ids[generated_ids == -100] = self.tokenizer.eos_token_id

        contexts = self.tokenizer.batch_decode(batch["context_input_ids"], skip_special_tokens=False)
        labels = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
        preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Record predictions
        for context, label, pred, metadata in zip(contexts, labels, preds, batch["metadata"]):
            metadata.update({
                "context": context,
                "prediction": pred,
                "label": label,
            })
            self.prediction_step_outputs.append(metadata)

        return {"loss": 0}
    
    def on_validation_epoch_begin(self):
        pass

    def on_validation_epoch_end(self):
        dataset_name = "val"
        os.makedirs(f"{self.cfg.exp_dir}/results/{dataset_name}", exist_ok=True)
        output_path = f"{self.cfg.exp_dir}/results/{dataset_name}/val@ep={self.trainer.current_epoch}-{self.trainer.global_step}-rank={self.trainer.global_rank}.jsonl"

        report = self.write_to_file(
            results=self.prediction_step_outputs,
            filepath=output_path, 
            cfg=self.cfg,
            ckpt=f"ep={self.trainer.current_epoch}-{self.trainer.global_step}",
            write_report=True
        )

        self.log("val/accuracy", report["accuracy_by_sample"], sync_dist=True)
        self.prediction_step_outputs.clear()



    # Dataloader
    def _build_dataloader(self, data_cfg):
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
                    batch_size=data_cfg.batch_size,
                    num_replicas=self.trainer.world_size,
                    rank=self.trainer.global_rank,
                    shuffle=data_cfg.shuffle,
                )
            else:
                batch_sampler = LengthBasedBatchSampler(
                    dataset,
                    batch_size=data_cfg.batch_size,
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
        elif data_cfg.get("batch_sampler", None) == "max_length":
            if self.trainer.world_size > 1:
                # Distributed training
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=self.trainer.world_size,
                    rank=self.trainer.global_rank,
                    shuffle=True
                )
            else:
                # Single GPU or CPU training
                sampler = None
            
            batch_sampler = MaxLengthBatchSampler(
                data_source=dataset,
                max_batch_length=data_cfg.max_batch_length,  # Your desired max length
                drop_last=True,
                shuffle=True if sampler is None else False,  # Don't shuffle if using DistributedSampler
                seed=42,
                sampler=sampler if sampler else None,
                world_size=self.trainer.world_size
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


    def train_dataloader(self):
        data_cfg = self.cfg.dataset.train_ds
        logging.info("\n********************* Training dataset *********************\n")
        
        dataloader = self._build_dataloader(data_cfg)
        
        logging.info("\n***************** End of Training dataset *****************\n")
        return dataloader

    def val_dataloader(self):
        data_cfg = self.cfg.dataset.validation_ds
        logging.info("\n******************** Validation dataset ********************\n")
        dataloader = self._build_dataloader(data_cfg)
        logging.info("\n**************** End of Validation dataset ****************\n")
        return dataloader

        

    # Optimizer and scheduler
    def configure_optimizers(self):
        
        trainable_parameters = []
        for name, params in self.model.named_parameters():
            if name in self.model.trainable_parameter_names:
                trainable_parameters.append(params)

        optimizer = FusedAdam(trainable_parameters, 
                              lr=self.cfg.optim.lr,
                              betas=(self.cfg.optim.betas),
                              weight_decay=self.cfg.optim.weight_decay,
                              )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.optim.sched.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )

        for name in self.model.trainable_parameter_names:
            logging.info(f"Training parameter: {name}")
        
        logging.info(f"Optimizer: {optimizer}")
        logging.info(f"Scheduler: {scheduler}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val/loss",
            },
        }
    
    def state_dict(self):
        """
        save only trainable parameters. 
        Note: There are two types of state_dict()
        - This state_dict() will be called by PTL Trainer.save_checkpoint() for saving checkpoints
        - DeSTA3Model.state_dict() will be called by PreTrainedModel.save_pretrained() for saving model weights and can be easily loaded by DeSTA3Model.from_pretrained()
        
        The only difference is the prefix(model.) in the state_dict keys.
        """
        trainable_state_dict = OrderedDict()
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_state_dict[name] = param.data.clone().detach()

        return trainable_state_dict
    

    def write_to_file(self, results, filepath, cfg=None, ckpt=None, write_report=True):
        filepath = Path(filepath)
        
        categories_accuracy = defaultdict(list)
        
        jsonl_path = Path(get_unique_filepath(filepath.parent / "preds" / filepath.name))
        os.makedirs(jsonl_path.parent, exist_ok=True)

        with open(jsonl_path, "w") as f:
            for i, result in enumerate(results):
                result["correct"] = self.metrics(result["prediction"], result["label"])
                result["index"] = i
                f.write(json.dumps(result) + "\n")
                categories_accuracy[result.get("category", "all")].append(result["correct"])

        if write_report:
            # Report
            report_path = jsonl_path.parent.parent / jsonl_path.name.replace(".jsonl", "-report.json")

            reported_results = []
            for i, result in enumerate(results):
                # remove context and audio_context for better readability (too long!)
                del result["context"]
                del result["audio_context"]
                reported_results.append(result)

            report = {
                "metric": self.metrics.metric_name,
                "preds_path": str(jsonl_path),
                "accuracy_by_sample": sum([reported_results[i]["correct"] for i in range(len(reported_results))]) / len(reported_results),
                "avg_accuracy_by_category": sum([sum(v) / len(v) for v in categories_accuracy.values()]),
                "categories_accuracy": dict([ (k, sum(v) / len(v)) for k, v in categories_accuracy.items()]),
                "config": OmegaConf.to_container(cfg, resolve=True),
                "ckpt": str(ckpt),
                "results": reported_results,
            }
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logging.info(f"Report saved to\n{report_path}\n")
            print(f"Report saved to\n{report_path}\n")

            return report

# ### Using Dataloader, validation step, optimizer config and state dict from DeSTA3PTLModule

# class CoD_DistillPTLModule(DeSTA3PTLModule):
#     def __init__(self, cfg):
#         super().__init__(cfg)
        
#         teacher_model_config = DeSTA3Config(
#             llm_model_id=self.cfg.teacher_model.llm.model_id,
#             encoder_model_id=self.cfg.teacher_model.encoder.model_id,
#             connector_mode=self.cfg.teacher_model.connector.mode,
#             qformer_num_hidden_layers=self.cfg.teacher_model.connector.num_hidden_layers,
#             prompt_size=self.cfg.teacher_model.connector.prompt_size,
#             first_n_layers=self.cfg.teacher_model.llm.first_n_layers if hasattr(self.cfg.teacher_model.llm, "first_n_layers") else -1,
#         )

#         print("="*100)
#         self.teacher_model = DeSTA3Model(teacher_model_config)
        
#         ckpt_path = getattr(cfg.teacher_model, "ckpt_path", None)
#         if ckpt_path is not None and ckpt_path.lower() != "null":
#             state = torch.load(ckpt_path, map_location="cpu")

#             # 如果是 Lightning .ckpt → 先拿掉 "state_dict" 再過濾 key
#             if "state_dict" in state:
#                 state = state["state_dict"]
#             # 保險起見：只保留屬於 teacher_model 的權重或直接用 strict=False
#             missing, unexpected = self.teacher_model.load_state_dict(state, strict=False)
#             logging.info(f"Teacher ckpt loaded: {ckpt_path}")
#             logging.info(f"  ▸ missing keys:     {len(missing)}")
#             logging.info(f"  ▸ unexpected keys: {len(unexpected)}")

#         # remove whisper decoder during PTL training (we only use Whisper decoder during inference)
#         del self.teacher_model.perception.whisper.model.decoder
#         del self.teacher_model.perception.whisper.proj_out
        
#         for p in self.teacher_model.parameters():   # 凍結參數
#             p.requires_grad_(False)
            
#         self.teacher_model.to("cuda")

#         # # tokenizer
#         # self.teacher_tokenizer = AutoTokenizer.from_pretrained(self.cfg.teacher_model.llm.model_id, cache_dir=os.getenv("HF_HOME"))
#         # self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token
#         # self.teacher_tokenizer.pad_token_id = self.teacher_tokenizer.eos_token_id
#         # self.teacher_tokenizer.padding_side = "left"
#         # self.teacher_tokenizer.add_tokens([self.cfg.dataset.audio_locator]) #warning cfg.dataset.audio_locator nee adding teacher?
        
        
#         self.add_noise = self.cfg.model.add_noise
#         self.kd_alpha  = self.cfg.model.kd_alpha
#         self.strategy  = getattr(self.cfg.model, "strategy", "wise")   # average | shallow | deep | wise
#         self.kd_temperature = getattr(self.cfg.model, "kd_temperature", 2.0)
#         # 保留 MSE criterion 以備不時之需
#         self.criterion = torch.nn.MSELoss(reduction="mean")

#     # -------- core forward (student + teacher + KD) ----------
#     def forward(self, batch):
#         # === unpack batch ===
#         input_ids              = batch["input_ids"]
#         attention_mask         = batch["attention_mask"]
#         batch_features         = batch["batch_features"]
#         batch_transcription_ids= batch["batch_transcription_ids"]
#         batch_start_positions  = batch["batch_start_positions"]
#         labels                 = batch.get("labels", None)

#         # 1) encoder
#         batch_speech_features, batch_speech_feature_lengths = self.model.perception(batch_features)

#         # 2) prepare llm embeds for student(audio)
#         student_input_embeds = self.model._prepare_inputs_for_llm(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             batch_speech_features=batch_speech_features,
#             batch_speech_feature_lengths=batch_speech_feature_lengths,
#             batch_transcription_ids=batch_transcription_ids,
#             batch_start_positions=batch_start_positions,
#         )

#         # If training, also build teacher(teacher) path
#         if self.training:
#             student_start_answer_positions = batch["audio_start_answer_positions"]
#             teacher_start_answer_positions  = batch["seed_start_answer_positions"]

#             teacher_input_ids      = batch["seed_input_ids"]
#             # teacher_attention_mask = batch["seed_attention_mask"]
            
#             # teacher_input_embeds   = self.model.llm_model.model.embed_tokens(teacher_input_ids)
#             teacher_input_embeds = self.teacher_model._prepare_inputs_for_llm(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             batch_speech_features=batch_speech_features,
#             batch_speech_feature_lengths=batch_speech_feature_lengths,
#             batch_transcription_ids=batch_transcription_ids,
#             batch_start_positions=batch_start_positions,
#             )

#             if self.add_noise:
#                 student_input_embeds, teacher_input_embeds = self._add_noise(
#                     student_input_embeds, teacher_input_embeds,
#                     student_start_answer_positions, teacher_start_answer_positions
#                 )

#         # 3) student forward
#         student_output = self.model.llm_model(
#             inputs_embeds        = student_input_embeds,
#             attention_mask       = attention_mask,
#             labels               = labels,
#             output_hidden_states = True,
#             use_cache            = False,
#         )
#         if self.training:
#             self.log("train/ntp_loss", student_output.loss.item(),
#                      sync_dist=True, batch_size=input_ids.size(0))

#         # 4) teacher forward + KD
#         if self.training:
#             with torch.no_grad():
#                 teacher_output = self.model.llm_model(
#                     inputs_embeds        = teach_input_embeds,
#                     attention_mask       = teach_attention_mask,
#                     output_hidden_states = True,
#                     use_cache            = False,
#                 )

#             kd_loss = self.layer_level_loss(
#                 student_hidden_states = student_output.hidden_states,
#                 teach_hidden_states  = teach_output.hidden_states,
#                 student_start_answer_positions = student_start_answer_positions,
#                 teach_start_answer_positions  = teach_start_answer_positions,
#                 temperature = self.kd_temperature,
#                 strategy    = self.strategy,
#                 shallow_k   = self.shallow_k,
#             )

#             student_output.loss = (1 - self.kd_alpha) * student_output.loss + self.kd_alpha * kd_loss
#             self.log("train/kd_loss", kd_loss.item(),
#                      sync_dist=True, batch_size=input_ids.size(0))

#         return student_output

#     # ---------------- PTL hooks ----------------
#     def training_step(self, batch, batch_idx):
#         try:
#             outputs = self(batch)
#             loss = outputs.loss
            
#         except Exception as e:
#             logging.error(f"Error in training step: {e}")
#             logging.error(traceback.format_exc())
#             logging.error(f"Batch: {batch}")
#             loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
#         ppl = torch.exp(loss)
#         bs  = batch["input_ids"].size(0)
#         self.log("train/loss_total", outputs.loss, prog_bar=True,
#                  sync_dist=True, batch_size=bs)
#         self.log("train/ppl", ppl, prog_bar=True,
#                  sync_dist=True, batch_size=bs)
#         return outputs.loss

#     def validation_step(self, batch, batch_idx):
#         # distill val 不一定要算 KD，直接用 predict
#         self.model.eval()
#         preds = self.predict_step(batch, batch_idx)
#         loss = torch.tensor(0.0, device=self.device)
#         ppl  = torch.tensor(0.0, device=self.device)
#         bs   = batch["input_ids"].size(0)
#         self.log("val_loss", loss.item(), sync_dist=True, batch_size=bs)
#         self.log("val_ppl",  ppl.item(),  sync_dist=True, batch_size=bs)
#         return {"val_loss": loss, "val_ppl": ppl, "predictions": preds}



#     # ---------------- layer-wise KD ----------------
#     def layer_level_loss(
#         self,
#         student_hidden_states,
#         teach_hidden_states,
#         student_start_answer_positions,
#         teach_start_answer_positions,
#         temperature: float = 2.0,
#         strategy: str = "wise",
#         shallow_k: int = 4,
#     ):
#         """
#         strategy:
#           - wise: 逐層 KL，平均 (你原本的做法)
#           - shallow: 只取前 K 層做 wise
#           - deep: 只取後 K 層做 wise
#           - average: 把所有層 concat 起來一次算 KL
#         """
#         # handle layer selection
#         s_hs = list(student_hidden_states)
#         t_hs = list(teach_hidden_states)

#         # usually hidden_states[0] 是 embeddings，若不想算把它丟掉
#         # s_hs, t_hs = s_hs[1:], t_hs[1:]

        
#         ###Need match for different llm model###
#         if strategy == "shallow":
#             target_layers_s = [4, 8, 12, 16]                  
#             target_layers_t = [6, 12, 18, 32]
#             idx_s = [l - 1 for l in target_layers_s if 1 <= l <= num_layers]
#             idx_s = [l - 1 for l in target_layers_t if 1 <= l <= num_layers]
#             # 真的去拿層
#             a_sh = [s_hs[i] for i in idx_s]
#             s_sh = [t_hs[i] for i in idx_t]
#         elif strategy == "deep":
#             s_hs, t_hs = s_hs[-shallow_k:], t_hs[-shallow_k:]
#         elif strategy == "average":
#             step_s = 4
#             step_t = 8
#             a_sampled = s_hs[::step_s]
#             s_sampled = t_hs[::step_t]
            

#         # wise: per-layer average
#         total_loss = 0.0
#         for layer_student_hidden_states, layer_teach_hidden_states in zip(s_hs, t_hs):
#             # layer wise
#             for i, (student_hidden_state, teach_hidden_state) in enumerate(zip(layer_student_hidden_states, layer_teach_hidden_states)):
#                 # in batch

#                 student_start = student_start_answer_positions[i].item()
#                 teach_start = teach_start_answer_positions[i].item()

                
#                 # Only compute the loss after answer positions
#                 student_hidden_state = student_hidden_state[student_start:]
#                 teach_hidden_state = teach_hidden_state[teach_start:]

#                 # truncate the length of student_hidden_state and teach_hidden_state to the same length
#                 max_length = min(student_hidden_state.size(0), teach_hidden_state.size(0))
#                 student_hidden_state = student_hidden_state[:max_length]
#                 teach_hidden_state = teach_hidden_state[:max_length]

#                 student_probs = F.log_softmax(student_hidden_state / temperature, dim=-1)
#                 teach_probs = F.softmax(teach_hidden_state / temperature, dim=-1)

#                 loss_kd = F.kl_div(
#                     student_probs,
#                     teach_probs,
#                     reduction='batchmean'
#                 ) * (temperature ** 2)

#                 total_loss += loss_kd
            
#         return total_loss / len(student_hidden_states)



#     def _add_noise(self, student_input_embeds, teach_input_embeds, student_start_answer_positions, teach_start_answer_positions):
#             # Create new copies of the input tensors

#             # Add same noise only to the portions after answer positions
#             for i in range(len(student_start_answer_positions)):
#                 student_start = student_start_answer_positions[i].item()
#                 teach_start = teach_start_answer_positions[i].item()

#                 student_answer_embeds = student_input_embeds[i, student_start:]
#                 teach_answer_embeds = teach_input_embeds[i, teach_start:]
                
#                 max_length = min(student_answer_embeds.size(0), teach_answer_embeds.size(0))
#                 noise = torch.randn_like(student_answer_embeds[:max_length])
#                 # replace the answer embeddings with noise
#                 # USE noise
#                 student_answer_embeds[:max_length] = noise
#                 teach_answer_embeds[:max_length] = noise

#                 student_input_embeds[i, student_start:student_start+max_length] = student_answer_embeds[:max_length]
#                 teach_input_embeds[i, teach_start:teach_start+max_length] = teach_answer_embeds[:max_length]
            

#             return student_input_embeds, teach_input_embeds
        
        
#     def _build_dataloader(self, data_cfg):
#             return_seed = data_cfg.get("return_seed", False)
#             if return_seed:
#                 dataset = DistillDataset(
#                     cfg=self.cfg,
#                     data_cfg=data_cfg,
#                     tokenizer=self.tokenizer,
#                     processor=self.processor,
#                     return_seed=True
#                 )
#             else:
#                 dataset = BaseAudioTextDataset(
#                     cfg=self.cfg,
#                     data_cfg=data_cfg,
#                     tokenizer=self.tokenizer,
#                     processor=self.processor
#                 )
#             logging.info(dataset[0])

#             if data_cfg.get("batch_sampler", None) == "length_based":
#                 if self.trainer.world_size > 1:
#                     batch_sampler = DistributedLengthBasedBatchSampler(
#                         dataset,
#                         max_batch_length=data_cfg.max_batch_length,
#                         num_replicas=self.trainer.world_size,
#                         rank=self.trainer.global_rank,
#                         shuffle=data_cfg.shuffle,
#                     )
#                 else:
#                     batch_sampler = LengthBasedBatchSampler(
#                         dataset,
#                         max_batch_length=data_cfg.max_batch_length,
#                         shuffle=data_cfg.shuffle,
#                         drop_last=data_cfg.drop_last,
#                     )
#                 dataloader = DataLoader(
#                     dataset,
#                     batch_sampler=batch_sampler,
#                     num_workers=data_cfg.num_workers,
#                     pin_memory=data_cfg.pin_memory,
#                     collate_fn=dataset.collate_fn,
#                 )
#             else:
#                 dataloader = DataLoader(
#                     dataset,
#                     batch_size=data_cfg.batch_size,
#                     collate_fn=dataset.collate_fn,
#                     shuffle=data_cfg.shuffle,
#                     num_workers=data_cfg.num_workers,
#                     pin_memory=data_cfg.pin_memory,
#                     drop_last=data_cfg.drop_last,
#                 )
#             return dataloader
        


    # model:
    #   kd_alpha: 0.5
    #   kd_temperature: 2.0
    #   strategy: wise   # or average / shallow / deep
    #   shallow_k: 4

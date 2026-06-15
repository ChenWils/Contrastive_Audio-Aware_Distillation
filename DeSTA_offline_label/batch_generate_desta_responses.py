#!/usr/bin/env python3
"""
Batch generate DeSTA responses using PyTorch Lightning infrastructure.
This script uses the eval_desta3.py batch processing infrastructure but
outputs training-ready format instead of evaluation results.

Input format (your seed data):
{
  "audio_filepath": "VoxCeleb1/dev/wav/id11006/PpUmNsKnClc/00008.wav",
  "seed_transcript": "[00:00:00 - 00:00:05] Roshon_Fegan: ...",
  "transcription": "the real lockdown...",
  "dataset": "VoxCeleb1",
  "duration": 5.2400625,
  "messages": [...],
  "system_prompt": "Imagine you can **hear**...",
  "input": "Based on the expression...",
  "target": "..."  # This will be replaced by model generation
}

Output format (training-ready):
{
  "audios": [{"audio_filepath": "...", "transcription": "..."}],
  "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "<start_audio><|AUDIO|><end_audio> ..."}],
  "target": "model generated response"
}
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from omegaconf import DictConfig, OmegaConf, open_dict
from pathlib import Path
import argparse
import torch
import json
from tqdm import tqdm
import logging

from desta.collections.ptl_modules.desta3_ptl import DeSTA3PTLModule
from desta.collections.ptl_modules.wilz_contrastive_distill import ContrastiveDistillPTLModule


class BatchWriterCallback(Callback):
    """
    PyTorch Lightning callback that writes predictions batch-by-batch.
    This ensures incremental saving and lower memory usage.
    """
    def __init__(self, output_file, eval_data_list, skip_errors=False):
        super().__init__()
        self.output_file = output_file
        self.eval_data_list = eval_data_list
        self.skip_errors = skip_errors
        self.last_written_idx = 0  # Track how many predictions we've written
        self.success_count = 0
        self.error_count = 0
        self.file_handle = None

    def on_predict_start(self, trainer, pl_module):
        """Open output file when prediction starts."""
        self.file_handle = open(self.output_file, 'w', encoding='utf-8')
        print(f"\n✓ Output file opened: {self.output_file}")
        print(f"  Writing results batch-by-batch...\n")

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Write predictions immediately after each batch completes.

        Note: The model's predict_step stores predictions in pl_module.prediction_step_outputs,
        not in the 'outputs' parameter. We need to read new entries from that list.
        """
        # Get new predictions from model's prediction_step_outputs list
        current_predictions = pl_module.prediction_step_outputs
        new_predictions = current_predictions[self.last_written_idx:]

        # Write each new prediction
        for i, result in enumerate(new_predictions):
            global_idx = self.last_written_idx + i

            if global_idx >= len(self.eval_data_list):
                break

            try:
                eval_data = self.eval_data_list[global_idx]

                # Extract generated response from result
                # The result is a metadata dict with "prediction" key
                generated_response = result.get("prediction", result.get("pred", ""))

                # Transform to training format
                training_data = transform_eval_to_training_format(
                    eval_data,
                    generated_response
                )

                # Write immediately
                self.file_handle.write(json.dumps(training_data, ensure_ascii=False) + '\n')
                self.file_handle.flush()  # Ensure data is written to disk

                self.success_count += 1

            except Exception as e:
                self.error_count += 1
                error_msg = f"Error processing sample {global_idx}: {str(e)}"

                if self.skip_errors:
                    print(f"⚠️  {error_msg} - Skipping...")
                    # Write with error marker
                    training_data = transform_eval_to_training_format(eval_data, "")
                    training_data["generation_error"] = str(e)
                    self.file_handle.write(json.dumps(training_data, ensure_ascii=False) + '\n')
                    self.file_handle.flush()
                else:
                    print(f"❌ {error_msg}")
                    raise

        # Update tracking counter
        self.last_written_idx = len(current_predictions)

    def on_predict_end(self, trainer, pl_module):
        """Close output file when prediction ends."""
        if self.file_handle:
            self.file_handle.close()
            print(f"\n✓ Output file closed: {self.output_file}")
            print(f"  Total written: {self.success_count} samples")
            if self.error_count > 0:
                print(f"  Errors: {self.error_count} samples")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch generate DeSTA responses for training data"
    )
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Experiment directory containing checkpoint")
    parser.add_argument("--epoch", type=int, required=True,
                        help="Epoch number to load")
    parser.add_argument("--config_file", type=str, default="",
                        help="Config file path (optional)")
    parser.add_argument("--input_manifest", type=str, required=True,
                        help="Input JSONL manifest in seed format")
    parser.add_argument("--output_manifest", type=str, required=True,
                        help="Output JSONL manifest in training format")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing audio files")
    parser.add_argument("--batch_size", type=int, default=24,
                        help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument("--skip_errors", action="store_true",
                        help="Skip samples with errors instead of stopping")
    parser.add_argument("--use_contrastive_decoding", action="store_true",
                        help="Enable contrastive decoding (default: False, normal generation)")
    parser.add_argument("--contrastive_alpha", type=float, default=1.0,
                        help="Contrastive alpha weight when --use_contrastive_decoding is enabled (default=1.0)")
    return parser.parse_args()


def transform_seed_to_eval_format(seed_data):
    """
    Transform seed format to eval format expected by DeSTA3 dataloader.

    Input (seed): {"audio_filepath", "transcription", "messages", "system_prompt", "input", ...}
    Output (eval): {"audios": [...], "messages": [...], "target": ...}
    """
    # Extract audio info
    audios = [{
        "audio_filepath": seed_data["audio_filepath"],
        "transcription": seed_data.get("transcription", "")
    }]

    # Build messages for evaluation
    messages = []

    # Add system message if exists
    if seed_data.get("system_prompt"):
        messages.append({
            "role": "system",
            "content": seed_data["system_prompt"]
        })

    # Add user message with <|AUDIO|> token
    # Note: The dataloader will wrap it with <start_audio><end_audio> automatically
    user_content = f"<|AUDIO|>\n\n{seed_data.get('input', '')}"
    messages.append({
        "role": "user",
        "content": user_content
    })

    eval_data = {
        "audios": audios,
        "messages": messages,
        "target": seed_data.get("target", "")  # Will be replaced by model generation
    }

    return eval_data


def transform_eval_to_training_format(eval_data, generated_response):
    """
    Transform eval format + generated response to training format.

    Input: eval format + generated text
    Output: training format with wrapped <start_audio><|AUDIO|><end_audio>
    """
    # Build training messages (system + user only, with wrapped audio token)
    training_messages = []

    for msg in eval_data["messages"]:
        training_msg = {"role": msg["role"], "content": msg["content"]}

        # Replace <|AUDIO|> with wrapped format for training
        if "<|AUDIO|>" in training_msg["content"]:
            training_msg["content"] = training_msg["content"].replace(
                "<|AUDIO|>",
                "<start_audio><|AUDIO|><end_audio>"
            )

        training_messages.append(training_msg)

    # Create output in training format
    training_data = {
        "audios": eval_data["audios"],
        "messages": training_messages,
        "target": generated_response
    }

    return training_data


def create_temp_manifest(input_manifest, data_root, temp_manifest_path):
    """
    Create temporary manifest in eval format for batch processing.
    Returns list of original seed data for later reconstruction.
    """
    print(f"\n[Preparing Data] Transforming seed data to eval format...")
    print(f"  Input:  {input_manifest}")
    print(f"  Temp:   {temp_manifest_path}")

    seed_data_list = []
    eval_data_list = []

    with open(input_manifest, 'r') as f:
        for line in tqdm(f, desc="Reading seed data"):
            if line.strip():
                seed_data = json.loads(line)
                seed_data_list.append(seed_data)
                eval_data = transform_seed_to_eval_format(seed_data)
                eval_data_list.append(eval_data)

    # Write temporary eval manifest
    temp_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_manifest_path, 'w') as f:
        for eval_data in eval_data_list:
            f.write(json.dumps(eval_data, ensure_ascii=False) + '\n')

    print(f"  ✓ Transformed {len(seed_data_list)} samples")

    return seed_data_list, eval_data_list


def main(args):
    pl.seed_everything(42)

    print("="*80)
    print("Batch DeSTA Response Generation")
    print("="*80)

    # Find checkpoint
    print(f"\n[1/4] Finding checkpoint...")
    ckpt = None
    for ckpt_path in Path(args.exp_dir).glob("**/*.ckpt"):
        if f"epoch={args.epoch}" in str(ckpt_path):
            ckpt = ckpt_path
            break

    if ckpt is None:
        raise FileNotFoundError(
            f"Checkpoint not found for epoch={args.epoch} in {args.exp_dir}"
        )

    logging.info(f"Loading checkpoint: {ckpt}")

    # Load model
    print(f"\n[2/4] Loading model...")
    try:
        import os
        os.environ["DESTA_EVAL_ONLY"] = "1"
        model = ContrastiveDistillPTLModule.load_from_checkpoint(
            ckpt, map_location="cpu", strict=False
        )
        print("  ✓ Loaded as ContrastiveDistillPTLModule")

        # Clean up projection layers
        if hasattr(model, 'projection_layers'):
            del model.projection_layers
        torch.cuda.empty_cache()

        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        print(f"  Failed to load as ContrastiveDistillPTLModule: {e}")
        print("  Falling back to DeSTA3PTLModule")
        model = DeSTA3PTLModule.load_from_checkpoint(ckpt, strict=False)

    # Configure model
    if args.config_file and args.config_file != "None":
        model.cfg = OmegaConf.create(OmegaConf.load(args.config_file))
    model.cfg.exp_dir = args.exp_dir
    model.cfg.trainer.devices = 1

    # Prepare temporary manifest in eval format
    temp_manifest_path = Path(args.output_manifest).parent / "_temp_eval_manifest.jsonl"
    seed_data_list, eval_data_list = create_temp_manifest(
        args.input_manifest,
        args.data_root,
        temp_manifest_path
    )

    # Configure trainer with batch writer callback
    print(f"\n[3/5] Setting up batch processing...")
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    devices = [0] if torch.cuda.is_available() else 1

    # Prepare output path
    output_path = Path(args.output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create callback for batch-by-batch writing
    writer_callback = BatchWriterCallback(
        output_file=output_path,
        eval_data_list=eval_data_list,
        skip_errors=args.skip_errors
    )

    trainer = pl.Trainer(
        logger=False,
        accelerator=accelerator,
        devices=devices,
        callbacks=[writer_callback]
    )

    # Configure dataset
    with open_dict(model.cfg):
        model.cfg.dataset.test_ds = model.cfg.dataset.validation_ds
        model.cfg.dataset.test_ds.manifest_filepaths = str(temp_manifest_path)
        model.cfg.dataset.test_ds.data_root = args.data_root
        model.cfg.dataset.test_ds.batch_size = args.batch_size
        model.cfg.dataset.test_ds.max_seq_length = args.max_seq_length
        model.cfg.model.generation_kwargs.max_new_tokens = args.max_new_tokens

        # Contrastive decoding parameters
        model.cfg.model.use_contrastive_decoding = args.use_contrastive_decoding
        model.cfg.model.contrastive_alpha = args.contrastive_alpha

    test_dataloader = model._build_dataloader(model.cfg.dataset.test_ds)

    # Run batch inference (results written by callback)
    print(f"\n[4/4] Running batch inference with incremental saving...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total samples: {len(seed_data_list)}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Mode: BATCH-BY-BATCH (incremental saving)")
    if args.use_contrastive_decoding:
        print(f"  Decoding: CONTRASTIVE (alpha={args.contrastive_alpha})")
    else:
        print(f"  Decoding: NORMAL")
    trainer.predict(model, dataloaders=test_dataloader)

    # Get statistics from callback
    success_count = writer_callback.success_count
    error_count = writer_callback.error_count

    # Clean up temp manifest
    if temp_manifest_path.exists():
        temp_manifest_path.unlink()

    # Summary
    print("\n" + "="*80)
    print("Generation Complete!")
    print("="*80)
    print(f"✓ Success: {success_count}/{len(seed_data_list)}")
    if error_count > 0:
        print(f"⚠️  Errors: {error_count}/{len(seed_data_list)}")
    print(f"\nOutput saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    args = parse_args()
    main(args)

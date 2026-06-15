import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf, open_dict
from lulutils import get_unique_filepath
from pytorch_lightning.callbacks import RichModelSummary, ModelCheckpoint
import argparse
from pathlib import Path
from omegaconf import DictConfig, OmegaConf, open_dict
from desta.collections.ptl_modules.desta3_ptl import DeSTA3PTLModule
from desta.collections.ptl_modules.wilz_contrastive_distill import ContrastiveDistillPTLModule
import logging
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--manifest_filepaths", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--enable_eval_entropy", action="store_true", default=False,
                        help="Enable entropy calculation during evaluation")
    parser.add_argument("--enable_logtoku", action="store_true", default=False,
                        help="Enable LogTokU (AU/EU) calculation during evaluation")
    parser.add_argument("--topk_cand", type=int, default=20,
                        help="K-cand: Top-K logits for computing per-token uncertainty (default: 20)")
    parser.add_argument("--use_contrastive_decoding", action="store_true", default=False,
                        help="Enable contrastive decoding (default: False, normal generation)")
    parser.add_argument("--contrastive_alpha", type=float, default=1.0,
                        help="Contrastive alpha weight when --use_contrastive_decoding is enabled (default=1.0)")

    return parser.parse_args()

def main(args):
    pl.seed_everything(42)

    for ckpt in Path(args.exp_dir).glob("**/*.ckpt"):
        if f"epoch={args.epoch}" in str(ckpt):
            break
    else:
        assert False, f"Checkpoint not found for epoch={args.epoch} in {args.exp_dir}"

    logging.info(f"Loading checkpoint: \n\n\n{ckpt}\n\n\n")

    # Try loading as ContrastiveDistillPTLModule first, fallback to DeSTA3PTLModule
    if args.enable_eval_entropy or args.enable_logtoku:
        try:
            # Set environment variable to skip teacher model initialization (saves VRAM)
            import os
            os.environ["DESTA_EVAL_ONLY"] = "1"
            print("Set DESTA_EVAL_ONLY=1 to skip teacher model initialization")

            # Load checkpoint - teacher model won't be initialized
            print("Loading checkpoint (teacher model will be skipped)...")
            model = ContrastiveDistillPTLModule.load_from_checkpoint(ckpt, map_location="cpu", strict=False)
            print("Loaded model as ContrastiveDistillPTLModule (teacher model not initialized)")

            # Clean up projection layers if they exist
            if hasattr(model, 'projection_layers'):
                print("Deleting projection layers...")
                del model.projection_layers
                print("Projection layers deleted")

            # Free up memory
            torch.cuda.empty_cache()
            print("Memory freed")

            # Now move only student model to GPU
            print("Moving student model to GPU...")
            model = model.to("cuda")
            print("Model moved to GPU")
        except Exception as e:
            print(f"Failed to load as ContrastiveDistillPTLModule: {e}")
            print("Falling back to DeSTA3PTLModule")
            model = DeSTA3PTLModule.load_from_checkpoint(ckpt, strict=False)
    else:
        print("Loading model as DeSTA3PTLModule")
        model = DeSTA3PTLModule.load_from_checkpoint(ckpt, strict=False)

    # Save model and tokenizer (optional)
    try:
        epoch = getattr(model, 'epoch', args.epoch)
        global_step = getattr(model, 'global_step', 'unknown')
        model.model.save_pretrained(f"{args.exp_dir}/hf_models/epoch-{epoch}-{global_step}")
        model.tokenizer.save_pretrained(f"{args.exp_dir}/hf_models/epoch-{epoch}-{global_step}")
        print(f"Model saved to {args.exp_dir}/hf_models/epoch-{epoch}-{global_step}")
    except Exception as e:
        print(f"Could not save model: {e}")
        print("Continuing with evaluation...")

    
    if args.config_file and args.config_file != "None":
        model.cfg = OmegaConf.create(OmegaConf.load(args.config_file))
    model.cfg.exp_dir = args.exp_dir
    model.cfg.trainer.devices = 1 # use one GPU

    # Auto-detect available accelerator
    if torch.cuda.is_available():
        accelerator = "cuda"
        devices = [0]
        print("Using CUDA accelerator")
    else:
        accelerator = "cpu"
        devices = 1
        print("WARNING: CUDA not available, falling back to CPU")

    trainer = pl.Trainer(
        logger=False,
        accelerator=accelerator,
        devices=devices,
    )

    # Overwrite dataset config
    with open_dict(model.cfg):
        model.cfg.dataset.test_ds = model.cfg.dataset.validation_ds
        model.cfg.dataset.test_ds.manifest_filepaths = args.manifest_filepaths
        model.cfg.dataset.test_ds.data_root = args.data_root
        model.cfg.dataset.test_ds.batch_size = 12 # origin: 8
        model.cfg.dataset.test_ds.max_seq_length = 1024
        model.cfg.model.generation_kwargs.max_new_tokens = 256

        # Contrastive decoding parameters
        model.cfg.model.use_contrastive_decoding = args.use_contrastive_decoding
        model.cfg.model.contrastive_alpha = args.contrastive_alpha

    # Enable entropy calculation if requested (for ContrastiveDistillPTLModule only)
    if args.enable_eval_entropy:
        if isinstance(model, ContrastiveDistillPTLModule):
            print("\n" + "="*80)
            print("ENABLING EVALUATION ENTROPY CALCULATION")
            print("="*80 + "\n")
            model.enable_eval_entropy = True
            # Initialize storage list if it doesn't exist
            if not hasattr(model, 'eval_entropy_samples'):
                model.eval_entropy_samples = []
        else:
            print(f"\nWARNING: --enable_eval_entropy only works with ContrastiveDistillPTLModule")
            print(f"Current model type: {type(model).__name__}")
            print(f"Entropy calculation will be DISABLED\n")

    # Enable LogTokU (AU/EU) calculation if requested (for ContrastiveDistillPTLModule only)
    if args.enable_logtoku:
        if isinstance(model, ContrastiveDistillPTLModule):
            print("\n" + "="*80)
            print("ENABLING LOGTOKU (AU/EU) CALCULATION")
            print(f"K-cand (top-K logits): {args.topk_cand}")
            print("="*80 + "\n")
            model.enable_logtoku = True
            # Initialize storage lists if they don't exist
            if not hasattr(model, 'eval_au_samples'):
                model.eval_au_samples = []
            if not hasattr(model, 'eval_eu_samples'):
                model.eval_eu_samples = []
            if not hasattr(model, 'eval_uncertainty_samples'):
                model.eval_uncertainty_samples = []
        else:
            print(f"\nWARNING: --enable_logtoku only works with ContrastiveDistillPTLModule")
            print(f"Current model type: {type(model).__name__}")
            print(f"LogTokU calculation will be DISABLED\n")

    # Set topk_cand parameter if provided (affects both entropy and LogTokU)
    if args.topk_cand and isinstance(model, ContrastiveDistillPTLModule):
        print(f"Setting topk_cand (K-cand) to {args.topk_cand}")
        model.topk_cand = args.topk_cand

    # Display contrastive decoding status
    if args.use_contrastive_decoding:
        print("\n" + "="*80)
        print("CONTRASTIVE DECODING ENABLED")
        print(f"Alpha: {args.contrastive_alpha}")
        print("="*80 + "\n")
    else:
        print("\n" + "="*80)
        print("NORMAL DECODING (Contrastive disabled)")
        print("="*80 + "\n")

    test_dataloader = model._build_dataloader(model.cfg.dataset.test_ds)

    trainer.predict(model, dataloaders=test_dataloader)

    results = model.prediction_step_outputs
    report_path = model.write_to_file(
        results,
        f"{args.exp_dir}/results/test@{args.dataset_name}/epoch={args.epoch}.jsonl",
        cfg=model.cfg,
        ckpt=f"ep={args.epoch}",
        write_report=True
    )
    print("\n\n", report_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
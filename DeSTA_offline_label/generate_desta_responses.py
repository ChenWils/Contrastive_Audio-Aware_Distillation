#!/usr/bin/env python3
"""
Generate DeSTA responses using actual audio files.
Reads a JSONL manifest, runs DeSTA inference on real audio,
and replaces the target field with model-generated responses.
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel
from lulutils import resolve_filepath, get_unique_filepath
from desta.collections.desta3.data.simple_dataset import _resolve_audio_filepath


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate DeSTA responses from audio files"
    )
    parser.add_argument(
        "--input_manifest",
        "-i",
        type=str,
        required=True,
        help="Path to input JSONL manifest file"
    )
    parser.add_argument(
        "--output_manifest",
        "-o",
        type=str,
        required=True,
        help="Path to output JSONL manifest file"
    )
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        default="DeSTA-ntu/DeSTA2-8B-beta",
        help="HuggingFace model ID or local path to DeSTA model"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing audio files (e.g., /path/to/data/audios)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["desta2", "desta3"],
        default="desta2",
        help="DeSTA model type (desta2 or desta3)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (currently only supports 1)"
    )
    parser.add_argument(
        "--skip_errors",
        action="store_true",
        help="Skip samples with errors instead of stopping"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file (skip already processed samples)"
    )

    return parser.parse_args()


def construct_desta2_messages(data, audio_filepath):
    """
    Construct messages for DeSTA2 model format.
    DeSTA2 uses separate "audio" role.
    """
    messages = []

    # Add system message if exists
    if data.get("system_prompt"):
        messages.append({
            "role": "system",
            "content": data["system_prompt"]
        })

    # Add audio role
    messages.append({
        "role": "audio",
        "content": audio_filepath
    })

    # Add user instruction (without seed_transcript)
    user_content = data.get("input", "")
    messages.append({
        "role": "user",
        "content": user_content
    })

    return messages


def construct_desta3_messages(data, audio_filepath):
    """
    Construct messages for DeSTA3 model format.
    DeSTA3 embeds audio in user message with <|AUDIO|> token.

    Format:
      - System: prompt text
      - User: "<|AUDIO|>\n\ninstruction"
      - Audios: [{audio: path, transcription: text}]

    Model will replace <|AUDIO|> with <start_audio><|AUDIO|><end_audio> internally.
    """
    messages = []

    # Add system message if exists
    if data.get("system_prompt"):
        messages.append({
            "role": "system",
            "content": data["system_prompt"]
        })

    # Add user message with audio
    # Use <|AUDIO|> token - model will wrap it with <start_audio><end_audio>
    user_content = f"<|AUDIO|>\n\n{data.get('input', '')}"
    messages.append({
        "role": "user",
        "content": user_content,
        "audios": [
            {
                "audio": audio_filepath,
                "transcription": data.get("transcription")  # Use transcription from JSONL (or None for ASR)
            }
        ]
    })

    return messages


def generate_desta2_response(model, messages, generation_kwargs):
    """Generate response using DeSTA2 model."""
    generated_ids = model.chat(
        messages,
        max_new_tokens=generation_kwargs["max_new_tokens"],
        do_sample=True,
        temperature=generation_kwargs["temperature"],
        top_p=generation_kwargs["top_p"]
    )
    response = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def generate_desta3_response(model, messages, generation_kwargs):
    """Generate response using DeSTA3 model."""
    generated_ids = model.generate(
        messages,
        generation_kwargs={
            "max_new_tokens": generation_kwargs["max_new_tokens"],
            "do_sample": True,
            "temperature": generation_kwargs["temperature"],
            "top_p": generation_kwargs["top_p"]
        }
    )
    response = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def main(args):
    print("="*80)
    print("DeSTA Audio Response Generation")
    print("="*80)

    # Load model
    print(f"\n[1/4] Loading {args.model_type.upper()} model from: {args.model_path}")
    import torch

    if args.model_type == "desta2":
        model = AutoModel.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN"),
            torch_dtype=torch.bfloat16,
            device_map=args.device
        )
    else:  # desta3
        from desta.collections.desta3.models.modeling_desta3 import DeSTA3Model, DeSTA3Config

        print("  Loading from HuggingFace format...")
        print("  Loading config first...")

        # Load config
        config = DeSTA3Config.from_pretrained(
            args.model_path,
            token=os.getenv("HF_TOKEN")
        )

        print("  Creating model (LLM and Whisper will be loaded)...")
        # Create model instance - this loads LLM and Whisper
        model = DeSTA3Model(
            config=config,
            cache_dir=os.getenv("HF_HOME"),
            token=os.getenv("HF_TOKEN")
        )

        print("  Loading trained connector weights from checkpoint...")
        # Now load the trained weights (only connector) from the safetensors
        import safetensors.torch
        state_dict = safetensors.torch.load_file(
            str(Path(args.model_path) / "model.safetensors")
        )
        model.load_state_dict(state_dict, strict=False)

        # Add missing method that generate() needs
        from desta.collections.desta3.data.simple_dataset import _prepare_audio_context_and_start_positions
        model._prepare_audio_context_and_start_positions = _prepare_audio_context_and_start_positions

        print(f"  Moving model to {args.device}...")
        model.to(args.device)

    model.eval()
    print(f"✓ Model loaded successfully on {args.device}")

    # Prepare paths
    print(f"\n[2/4] Preparing paths")
    input_manifest_path = resolve_filepath(args.input_manifest)
    output_manifest_path = Path(args.output_manifest)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Make unique output path if file exists
    if output_manifest_path.exists():
        output_manifest_path = get_unique_filepath(str(output_manifest_path))

    data_root = Path(args.data_root)

    print(f"  Input:  {input_manifest_path}")
    print(f"  Output: {output_manifest_path}")
    print(f"  Data root: {data_root}")

    # Count total samples
    print(f"\n[3/4] Counting samples...")
    with open(input_manifest_path, 'r') as f:
        total_samples = sum(1 for _ in f)
    print(f"  Total samples: {total_samples}")

    # Process samples
    print(f"\n[4/4] Generating responses...")
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p
    }

    error_count = 0
    success_count = 0

    with open(input_manifest_path, 'r') as fin, \
         open(output_manifest_path, 'w') as fout:

        for i, line in enumerate(tqdm(fin, total=total_samples, desc="Processing")):
            try:
                data = json.loads(line)

                # Resolve audio filepath
                audio_rel_path = data["audio_filepath"]
                audio_full_path = data_root / audio_rel_path
                audio_full_path = _resolve_audio_filepath(str(audio_full_path))

                # Construct messages based on model type
                if args.model_type == "desta2":
                    messages = construct_desta2_messages(data, audio_full_path)
                    response = generate_desta2_response(model, messages, generation_kwargs)
                else:  # desta3
                    messages = construct_desta3_messages(data, audio_full_path)
                    response = generate_desta3_response(model, messages, generation_kwargs)

                # Update data with generated response
                data["target"] = response

                # Replace messages with DeSTA input format (for training data generation)
                # Convert <|AUDIO|> to <start_audio><|AUDIO|><end_audio> for training format
                training_messages = []
                for msg in messages:
                    training_msg = {"role": msg["role"], "content": msg["content"]}

                    # Replace <|AUDIO|> with pre-wrapped format for training
                    if "<|AUDIO|>" in training_msg["content"]:
                        training_msg["content"] = training_msg["content"].replace(
                            "<|AUDIO|>",
                            "<start_audio><|AUDIO|><end_audio>"
                        )

                    training_messages.append(training_msg)

                # Add assistant response to messages
                training_messages.append({
                    "role": "assistant",
                    "content": response
                })

                # Update messages field with training-ready format
                data["messages"] = training_messages

                # Write to output
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                fout.flush()  # Ensure data is written even if interrupted

                success_count += 1

            except Exception as e:
                error_count += 1
                error_msg = f"Error processing sample {i}: {str(e)}"

                if args.skip_errors:
                    tqdm.write(f"⚠️  {error_msg} - Skipping...")
                    # Write original data with error marker
                    data["generation_error"] = str(e)
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                    fout.flush()
                else:
                    tqdm.write(f"❌ {error_msg}")
                    raise

    # Summary
    print("\n" + "="*80)
    print("Generation Complete!")
    print("="*80)
    print(f"✓ Success: {success_count}/{total_samples}")
    if error_count > 0:
        print(f"⚠️  Errors: {error_count}/{total_samples}")
    print(f"\nOutput saved to: {output_manifest_path}")
    print("="*80)


if __name__ == "__main__":
    args = parse_args()
    main(args)

#!/bin/bash

export TRANSFORMERS_OFFLINE=0
export HF_TOKEN="your_hf_token_here"
export ROOT_DIR="/home/chenwils/DeSTA3-dev-main"
export DATA_ROOT="/home/chenwils/DeSTA3-dev-main/data/desta2/desta2_audios"
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH=/NeMo:$ROOT_DIR:$PYTHONPATH

# # # # # # # # # # # # # # # # # # # #
# Configuration
# # # # # # # # # # # # # # # # # # # #

# Checkpoint configuration
exp_dir="/home/chenwils/DeSTA3-dev-main/data/teacher_ckpt/llama_31_8B/250306-13@rdesta2/pytorch_lightning/3b88q8kb"
epoch=10

# Input/Output files
input_manifest="/groups/chenwils/Distill_Qwen2.5-omni/manifest/241226_respond2@0811_seed_transcript.jsonl"  # Your seed format
output_manifest="$ROOT_DIR/DeSTA_offline_label/AUDIO_OUTPUTS/Contrastive_training_ready_$(date +%Y%m%d_%H%M%S).jsonl"

# Data root (where audio files are located)
data_root="$DATA_ROOT"

# Batch processing parameters
batch_size=24
max_new_tokens=256
max_seq_length=1024

# Contrastive decoding parameters (set to enable)
use_contrastive_decoding=true  # Set to true to enable contrastive decoding
contrastive_alpha=0.5           # Alpha weight (0.1-1.5, typical: 1.0)

# GPU device
gpu_device=${1:-0}  # Default to GPU 0, override with first argument

# # # # # # # # # # # # # # # # # # # #
# Run batch generation
# # # # # # # # # # # # # # # # # # # #

echo "================================================"
echo "Batch DeSTA Response Generation"
echo "================================================"
echo "Experiment dir:   $exp_dir"
echo "Epoch:            $epoch"
echo "Input manifest:   $input_manifest"
echo "Output manifest:  $output_manifest"
echo "Data root:        $data_root"
echo "Batch size:       $batch_size"
echo "Max new tokens:   $max_new_tokens"
echo "GPU device:       $gpu_device"
echo "Contrastive:      $use_contrastive_decoding"
if [ "$use_contrastive_decoding" = true ]; then
    echo "  Alpha:          $contrastive_alpha"
fi
echo "================================================"

pip install --user --upgrade transformers==4.49.0 typing_extensions peft==0.17.0 
pip install --user git+https://github.com/kehanlu/lulutils.git
pip install --user whisper-normalizer 
pip install --user numpy==1.24


# Build contrastive decoding arguments
contrastive_args=""
if [ "$use_contrastive_decoding" = true ]; then
    contrastive_args="--use_contrastive_decoding --contrastive_alpha $contrastive_alpha"
fi

CUDA_VISIBLE_DEVICES=$gpu_device python $ROOT_DIR/DeSTA_offline_label/batch_generate_desta_responses.py \
    --exp_dir "$exp_dir" \
    --epoch $epoch \
    --config_file "" \
    --input_manifest "$input_manifest" \
    --output_manifest "$output_manifest" \
    --data_root "$data_root" \
    --batch_size $batch_size \
    --max_new_tokens $max_new_tokens \
    --max_seq_length $max_seq_length \
    --skip_errors \
    $contrastive_args

echo ""
echo "================================================"
echo "Done! Output saved to:"
echo "$output_manifest"
echo "================================================"

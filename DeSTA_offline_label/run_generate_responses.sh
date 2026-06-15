#!/bin/bash
# Convenience script for running DeSTA response generation

# Default configuration
INPUT_MANIFEST="/groups/chenwils/Distill_Qwen2.5-omni/manifest/241226_respond2@0811_seed_transcript_sub.jsonl"
OUTPUT_DIR="/home/chenwils/DeSTA3-dev-main/DeSTA_offline_label/AUDIO_OUTPUTS"
DATA_ROOT="/home/chenwils/DeSTA3-dev-main/data/desta2/desta2_audios"
MODEL_PATH="/home/chenwils/DeSTA3-dev-main/data/teacher_ckpt/llama_31_8B/250306-13@rdesta2/pytorch_lightning/3b88q8kb/hf_models/epoch-10-0"
MODEL_TYPE="desta3"
DEVICE="cuda"

# Generation parameters
MAX_NEW_TOKENS=128
TEMPERATURE=0.6
TOP_P=0.9

# Parse command line arguments (optional overrides)
while [[ $# -gt 0 ]]; do
    case $1 in
        --input|-i)
            INPUT_MANIFEST="$2"
            shift 2
            ;;
        --output_dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --model_path|-m)
            MODEL_PATH="$2"
            shift 2
            ;;
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --input, -i        Input JSONL manifest path"
            echo "  --output_dir, -o   Output directory"
            echo "  --data_root        Root directory containing audio files"
            echo "  --model_path, -m   HuggingFace model ID or local path"
            echo "  --model_type       Model type (desta2 or desta3)"
            echo "  --device           Device (cuda or cpu)"
            echo "  --help, -h         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done


pip install --user --upgrade transformers==4.49.0 typing_extensions peft==0.17.0 
pip install --user git+https://github.com/kehanlu/lulutils.git
pip install --user whisper-normalizer 
pip install --user numpy==1.24

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate output filename based on input and model
INPUT_BASENAME=$(basename "$INPUT_MANIFEST" .jsonl)
MODEL_NAME=$(basename "$MODEL_PATH")
OUTPUT_MANIFEST="${OUTPUT_DIR}/${INPUT_BASENAME}_${MODEL_NAME}_generated.jsonl"

echo "========================================"
echo "DeSTA Response Generation"
echo "========================================"
echo "Input:      $INPUT_MANIFEST"
echo "Output:     $OUTPUT_MANIFEST"
echo "Data Root:  $DATA_ROOT"
echo "Model:      $MODEL_PATH ($MODEL_TYPE)"
echo "Device:     $DEVICE"
echo "========================================"
echo ""

# Run the generation script (add parent dir to PYTHONPATH)
cd /home/chenwils/DeSTA3-dev-main
PYTHONPATH=/home/chenwils/DeSTA3-dev-main:$PYTHONPATH python3 /home/chenwils/DeSTA3-dev-main/DeSTA_offline_label/generate_desta_responses.py \
    --input_manifest "$INPUT_MANIFEST" \
    --output_manifest "$OUTPUT_MANIFEST" \
    --model_path "$MODEL_PATH" \
    --model_type "$MODEL_TYPE" \
    --data_root "$DATA_ROOT" \
    --device "$DEVICE" \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --skip_errors

echo ""
echo "Done! Check output at: $OUTPUT_MANIFEST"

export TRANSFORMERS_OFFLINE=0
export HF_TOKEN="your_hf_token_here"
export ROOT_DIR="/home/chenwils/Contrastive_Audio-Aware_Distillation"
# export HF_HOME="/home/chenwils/.cache/huggingface"
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH=/NeMo:$ROOT_DIR:$PYTHONPATH

pip install --user --upgrade transformers==4.49.0 typing_extensions peft==0.17.0 
pip install --user git+https://github.com/kehanlu/lulutils.git
pip install --user whisper-normalizer 
pip install --user numpy==1.24
# # # # # # # # # # # # # # # # # # # # 
# If you want to pull audios from Morioh/livingroom to data_root

# shell:
# desta-pull-audios --data_root /root/lab/DeSTA3-dev/my_data --repo_id Morioh/livingroom -p dynamic-superb-test --revision eval -t extract download

# desta-pull-audios --data_root /root/lab/DeSTA3-dev/my_data --repo_id Morioh/livingroom -p MMAU-test-mini --revision eval -t extract download

# # # # # # # # # # # # # # # # # # # # 

# exp_dir="/home/chenwils/DeSTA3-dev-main/data/teacher_ckpt/llama_31_8B/250306-13@rdesta2/pytorch_lightning/3b88q8kb"
# exp_dir="/home/chenwils/DeSTA3-dev-main/my_exps/distill_desta2_llama32_8B_llama32_3B/251004@bsA6000"
# exp_dir='/home/chenwils/DeSTA3-dev-main/my_exps/contrastive_logits_distill_desta2_llama32_8B_llama32_3B/251227@bsA6000'
# exp_dir="/home/chenwils/DeSTA3-dev-main/my_exps/contractive_logits_distill_desta2_llama32_8B_llama32_3B/260108@bsA6000"
# exp_dir="/home/chenwils/DeSTA3-dev-main/my_exps/desta2_base_3B/250915@bs3090"
exp_dir="/home/chenwils/DeSTA3-dev-main/my_exps/RB_contrastive_logits_distill_desta2_llama32_8B_llama32_3B/260303@bsA6000"
epoch=9

# test_name="MMAU-test-mini"
test_name="MCR-BENCH-SER"
# test_name="dynamic-superb"
# # # # # # # # # # # # # # # # # # # # 

if [ "$test_name" == "val" ]; then
    manifest_filepaths="/root/project_kira/data/0720_val.jsonl"  # list of filepaths
    data_root="/NeMo/data/audios/dynamic-superb-train"
elif [ "$test_name" == "dynamic-superb" ]; then
    # manifest_filepaths="https://huggingface.co/datasets/Morioh/shelf/resolve/manifest/eval_manifest/dynamic-superb-test.val.jsonl"
    manifest_filepaths="https://huggingface.co/datasets/Morioh/shelf/resolve/manifest/eval_manifest/dynamic-superb-test.test.jsonl"
    data_root="/home/chenwils/DeSTA3-dev-main/data/eval"
elif [ "$test_name" == "MCR-BENCH-SER" ]; then
    manifest_filepaths="/groups/chenwils/Distill_Qwen2.5-omni/manifest/MCR-BENCH-SER.jsonl"
    data_root="/groups/chenwils/MCR-BENCH/MCR-Bench"
elif [ "$test_name" == "MMAU-test-mini" ]; then
    manifest_filepaths="https://huggingface.co/datasets/Morioh/shelf/resolve/manifest/eval_manifest/MMAU-test-mini.jsonl"
    data_root="/home/chenwils/DeSTA3-dev-main/data/eval"
fi


echo ================================================
echo exp_dir: $exp_dir
echo epoch: $epoch
echo manifest_filepaths: $manifest_filepaths
echo test_name: $test_name
echo Contrastive decoding: $USE_CONTRASTIVE_DECODING
if [ "$USE_CONTRASTIVE_DECODING" == "true" ]; then
    echo "  Alpha: $CONTRASTIVE_ALPHA"
fi
echo ================================================


# Enable entropy calculation (add --enable_eval_entropy flag)
ENABLE_ENTROPY=false  # Set to false to disable

if [ "$ENABLE_ENTROPY" == "true" ]; then
    ENTROPY_FLAG="--enable_eval_entropy"
else
    ENTROPY_FLAG=""
fi

# Enable LogTokU (AU/EU) calculation (add --enable_logtoku flag)
ENABLE_LOGTOKU=false  # Set to true to enable
TOPK_CAND=5  # K-cand: Top-K logits for computing per-token uncertainty (default: 20)

if [ "$ENABLE_LOGTOKU" == "true" ]; then
    LOGTOKU_FLAG="--enable_logtoku --topk_cand $TOPK_CAND"
else
    LOGTOKU_FLAG=""
fi

# Contrastive decoding parameters (set to enable)
USE_CONTRASTIVE_DECODING=false  # Set to true to enable contrastive decoding
CONTRASTIVE_ALPHA=1.0           # Alpha weight (0.1-1.5, typical: 0.3-0.5)

if [ "$USE_CONTRASTIVE_DECODING" == "true" ]; then
    CONTRASTIVE_FLAG="--use_contrastive_decoding --contrastive_alpha $CONTRASTIVE_ALPHA"
else
    CONTRASTIVE_FLAG=""
fi

CUDA_VISIBLE_DEVICES=$1 python $ROOT_DIR/examples/train/desta3_ptl/eval_desta3.py \
    --config_file "" \
    --exp_dir $exp_dir \
    --manifest_filepaths "$manifest_filepaths" \
    --data_root "$data_root" \
    --dataset_name $test_name \
    --epoch $epoch \
    $ENTROPY_FLAG \
    $LOGTOKU_FLAG \
    $CONTRASTIVE_FLAG
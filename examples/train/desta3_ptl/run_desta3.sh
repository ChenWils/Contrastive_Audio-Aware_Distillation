# export TRANSFORMERS_OFFLINE=0
# export HF_TOKEN="..."

# export CUDA_VISIBLE_DEVICES=$1
# export PYTHONPATH=/NeMo:$PYTHONPATH
# export HF_HOME="/NeMo/.cache"
export ROOT_DIR="/home/chenwils/Contrastive_Audio-Aware_Distillation"
export HF_TOKEN=your_hf_token_here
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH
# export NUMBA_CACHE_DIR=/tmp/numba_cache

#project="pytorch_lightning"
#name="rdesta2"
project="RB_AUDIO_contrastive_logits_distill_desta2_llama32_8B_llama32_3B"
name="260426@bsA6000"

#name=$(date +%y%m%d-%H)@${name}
exp_dir="/home/chenwils/DeSTA3-dev-main/my_exps/${project}/${name}"

# resume_from_checkpoint=/home/jovyan/shared/kehanluu/workspace/DeSTA3-dev/my_exps/a100/250327-04@ds-8a100-syn5/checkpoints/epoch\=5-step\=10548.ckpt
# init_from_pretrained_weights=null
pip install --user --upgrade transformers==4.49.0 typing_extensions peft==0.17.0 
pip install --user git+https://github.com/kehanlu/lulutils.git
pip install --user whisper-normalizer 
pip install --user numpy==1.24

python examples/train/desta3_ptl/run_desta3_ptl.py \
    +exp_dir=${exp_dir} \
    project=${project} \
    name=${name} \
    # +resume_from_checkpoint=${resume_from_checkpoint} \
    # +init_from_pretrained_weights=${init_from_pretrained_weights}

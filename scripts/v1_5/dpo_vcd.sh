#!/bin/bash
# RLHF-V-Dataset_images
export WANDB_MODE=offline
export WANDB_API_KEY=""

# 设置环境变量
# 激活目标环境、
# 初始化 Conda
source /data/ruipeng.zhang/anaconda3/etc/profile.d/conda.sh
conda activate llava-dpo

OUTPUT_DIR="/data/ruipeng.zhang/dpo_on/output/llava_lora_r64_vcd"
mkdir -p $OUTPUT_DIR

exec 1> >(tee "${OUTPUT_DIR}/stdout.log" >&1) 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)

cp -f "$0" "${OUTPUT_DIR}/script.sh"

MASTER_PORT_START=10000
MASTER_PORT_END=65535
MASTER_PORT="$(
	comm -23 \
		<(seq "${MASTER_PORT_START}" "${MASTER_PORT_END}" | sort) \
		<(ss -Htan | awk '{ print $4 }' | awk -F ':' '{ print $NF }' | sort -u) |
		shuf | head -n 1
)"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,P2P

gpu_vis=7 # Change this to the GPUs you want to use, e.g., 0,1,2 for 3 GPUs

MODEL_PATH="/data/ruipeng.zhang/OPA-DPO/base_models/llava-v1.5-7b"
REF_MODEL_PATH="/data/ruipeng.zhang/OPA-DPO/base_models/llava-v1.5-7b"

deepspeed --include localhost:$gpu_vis --master_port $MASTER_PORT \
    --module llava_dpo.train.dpo_train \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $MODEL_PATH \
    --ref_model_name_or_path $REF_MODEL_PATH \
    --n_random_images 0 \
    --version v1 \
    --lora_enable True \
    --lora_r 32  \
    --lora_alpha  \
    --lora_dropout 0.05 \
    --scale_coeff 0.1 \
    --data_path /data/ruipeng.zhang/dpo_on/Dataset_vcd.json \
    --image_folder /data/ruipeng.zhang/dpo_on/RLHF-V-Dataset_images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 4 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --learning_rate 2e-5 \
    --weight_decay 0.05 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --log_project LLaVA-DPO-WL \
    --report_to wandb \
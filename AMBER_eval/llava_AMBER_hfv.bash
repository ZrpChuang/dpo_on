#!/bin/bash

# ==============================================================================
#  SSH脚本：用于运行LLaVA模型推理生成回答
#
#  功能:
#  1. 初始化并激活指定的Conda环境 (mdpo)。
#  2. 执行Python推理脚本 (generate_responses.py)。
#  3. 传入指定的模型路径和输出文件路径。
#
#  使用方法:
#  1. 将Python代码保存为 generate_responses.py。
#  2. 确保此脚本有执行权限: chmod +x run_inference.sh
#  3. 运行此脚本: ./run_inference.sh
# ==============================================================================

# 当任何命令失败时，立即退出脚本
set -e

echo "--- 开始执行模型推理任务 ---"

# --- 1. 环境设置 ---
echo "正在初始化Conda..."
# 根据您提供的路径加载Conda环境
source /data/ruipeng.zhang/anaconda3/etc/profile.d/conda.sh

echo "正在激活Conda环境: llava-dpo"
conda activate llava-dpo
echo "Conda环境已激活: $(which python)" # 打印Python解释器路径以确认

# --- 2. 参数配置 ---
# 将Python脚本的文件名放在这里
PYTHON_SCRIPT="AMBER_llava.py"

# 从外部传入或在此处硬编码的核心参数,就是为hfv服务的
MODEL_PATH="/data/ruipeng.zhang/dpo_on/output/llava_lora_r128_hfv"
OUTPUT_FILE="/data/ruipeng.zhang/dpo_on/AMBER_eval/AMBER_output/AMBER_llava_responses_hfv_r128.jsonl"

# 其他参数将使用Python脚本中定义的默认值。
# 如果也想在这里指定，可以取消下面几行的注释并修改：
MODEL_BASE="/data/ruipeng.zhang/OPA-DPO/base_models/llava-v1.5-7b"
export CUDA_VISIBLE_DEVICES="5"  # 设置可见的GPU设备

# --- 3. 执行Python脚本 ---
echo "即将执行Python推理脚本..."
echo "  - 模型路径: $MODEL_PATH"
echo "  - 输出文件: $OUTPUT_FILE"

# 使用 \ 符号将长命令分行，使其更易读
python "$PYTHON_SCRIPT" \
    --model-path "$MODEL_PATH" \
    --output-file "$OUTPUT_FILE"

# --- 如果需要覆盖所有参数，可以使用下面的完整版命令 ---
# python "$PYTHON_SCRIPT" \
#     --model-path "$MODEL_PATH" \
#     --model-base "$MODEL_BASE" \
#     --question-file "$QUESTION_FILE" \
#     --image-folder "$IMAGE_FOLDER" \
#     --output-file "$OUTPUT_FILE" \
#     --conv-mode "llava_v1" \
#     --temperature 0.2


echo "--- 任务执行完毕 ---"
echo "结果已成功保存到: $OUTPUT_FILE"
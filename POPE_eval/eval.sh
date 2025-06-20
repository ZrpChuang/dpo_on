# 激活目标环境、
# 初始化 Conda
source /data/ruipeng.zhang/anaconda3/etc/profile.d/conda.sh
conda activate llava-dpo

# 指定要使用的GPU，例如 "0" 或 "0,1"
export CUDA_VISIBLE_DEVICES="3"

# (可选) 为了调试CUDA错误，可以取消下面这行的注释
# export CUDA_LAUNCH_BLOCKING=1

# --- 2. 路径与文件名配置 (请根据你的项目修改) ---

# Python脚本的完整路径
PYTHON_SCRIPT_PATH="/data/ruipeng.zhang/dpo_on/POPE_eval/pope_llava.py" # <--- ‼️ 请务必修改为你自己的Python脚本路径

# 基础模型路径 (原始的、未微调的模型)
MODEL_BASE="/data/ruipeng.zhang/OPA-DPO/base_models/llava-v1.5-7b"



# 图像文件夹路径
IMAGE_FOLDER="/data/ruipeng.zhang/VCD/experiments/data/coco/val2014"

# 问题文件路径 (当前为POPE数据集)
QUESTION_FILE="/data/ruipeng.zhang/VCD/experiments/data/POPE/coco/coco_pope_adversarial.json"

# 输出文件路径
ANSWERS_FILE="/data/ruipeng.zhang/dpo_on/POPE_eval/llava_vcd_r64/coco_adversarial_output.jsonl"
# 你微调后的模型路径
MODEL_PATH="/data/ruipeng.zhang/dpo_on/output/llava_lora_r64_att"
# 日志文件路径
LOG_FILE="/data/ruipeng.zhang/dpo_on/POPE_eval/llava_vcd_r64/adversarial_$(date +'%Y%m%d_%H%M%S').log"


# --- 3. 激活环境与执行命令 ---

echo "==================================================="
echo "开始执行 LLaVA POPE 评测..."
echo "Conda 环境: $CONDA_ENV_NAME"
echo "使用GPU: $CUDA_VISIBLE_DEVICES"
echo "模型路径: $MODEL_PATH"
echo "基础模型: $MODEL_BASE"
echo "问题文件: $QUESTION_FILE"
echo "输出文件: $ANSWERS_FILE"
echo "日志将保存到: $LOG_FILE"
echo "==================================================="



# 使用 nohup 在后台运行 Python 脚本
# 2>&1 将标准错误重定向到标准输出
# | ts ... 为每行日志添加时间戳 (需要安装 moreutils: sudo apt-get install moreutils)
# & 表示后台运行
nohup python -u $PYTHON_SCRIPT_PATH \
    --model-path "$MODEL_PATH" \
    --model-base "$MODEL_BASE" \
    --image-folder "$IMAGE_FOLDER" \
    --question-file "$QUESTION_FILE" \
    --answers-file "$ANSWERS_FILE" \
    --conv-mode "llava_v1" \
    --temperature 0.0 \
    --top_p 1.0 \
    > "$LOG_FILE" 2>&1 &


# 打印后台任务的进程ID (PID)
echo "脚本已在后台启动，进程ID (PID) 为: $!"
echo "你可以使用 'tail -f $LOG_FILE' 命令实时查看日志。"
echo "要停止任务，请使用 'kill $!'"
export CUDA_VISIBLE_DEVICES=0  # 只使用 GPU 4
seed=${1:-55}
dataset_name=${2:-"coco"}
type=${3:-"adversarial"}
model_path=${4:-"/data/ruipeng.zhang/V-DPO/output/lora_4ep_vlfeedback_llava_10k"}
cd_alpha=${5:-1}
cd_beta=${6:-0.2}
noise_step=${7:-500}
if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
  image_folder=/data/ruipeng.zhang/VCD/experiments/data/coco/val2014
else
  image_folder=./data/gqa/images
fi

python /data/ruipeng.zhang/V-DPO/eval/AMBER_llava.py \
--model-path ${model_path} \
--question-file /data/ruipeng.zhang/VCD/experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./AMBER_output/checkpoint-11468_${dataset_name}_pope_${type}_answers_seed${seed}.jsonl \
--cd_alpha $cd_alpha \
--cd_beta $cd_beta \
--noise_step $noise_step \
--seed ${seed}



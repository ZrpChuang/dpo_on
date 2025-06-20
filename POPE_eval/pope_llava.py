import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava_dpo.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava_dpo.conversation import conv_templates, SeparatorStyle
from llava_dpo.model.builder import load_pretrained_model
from llava_dpo.utils import disable_torch_init
from llava_dpo.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava_dpo.model import *
from PIL import Image
import math

# import kornia
from transformers import set_seed


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)#能识别波浪线的函数
    model_name = get_model_name_from_path(model_path)#模型的名字，是从路径中直接挖出来的，所以一定要注意命名
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]  #导入问题
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")#打开回答文件
    for line in tqdm(questions):#遍历每一个问题 显示进度条
        idx = line["question_id"]  #问题
        image_file = line["image"]  #图片
        qs = line["text"]  # 问题
        cur_prompt = qs
        
        # qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        # conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)

        # 改成这样，看看能不能生成完整文本
        # conv.append_message(conv.roles[0], None)
        
        prompt = conv.get_prompt()
        # print("prompt=:",prompt)

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                #input改了
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=False,
                num_beams=1, # 明确指定使用贪心搜索
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=1024,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "model_id": model_name,
                                   "image": image_file,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/data/ruipeng.zhang/dpo_on/output/llava_lora_r16_att")
    parser.add_argument("--model-base", type=str, default="/data/ruipeng.zhang/OPA-DPO/base_models/llava-v1.5-7b")
    parser.add_argument("--image-folder", type=str, default="/data/ruipeng.zhang/VCD/experiments/data/coco/val2014")
    parser.add_argument("--question-file", type=str, default="/data/ruipeng.zhang/VCD/experiments/data/POPE/coco/coco_pope_adversarial.json")
    #我猜应该不是这个“问题文件”导致模型输出只有yes或者no
    parser.add_argument("--answers-file", type=str, default="/data/ruipeng.zhang/dpo_on/POPE_eval/llava_att_r16/coco_adversarial.jsonl") #这个是输出
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)

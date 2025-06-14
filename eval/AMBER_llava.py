import argparse
import torch
import os
import json
from tqdm import tqdm
from PIL import Image

# 导入你的LLaVA项目中的必要模块
# 注意：在原始代码中，sys.path.append使用了 'file'，这可能是一个拼写错误。
# 我已将其更正为 '__file__'，这是Python中引用当前文件路径的标准方式。
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava_dpo.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava_dpo.conversation import conv_templates, SeparatorStyle
from llava_dpo.model.builder import load_pretrained_model
from llava_dpo.utils import disable_torch_init
from llava_dpo.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

# 主生成函数
def generate_model_responses(args):
    """
    使用训练好的LLaVA模型加载问题文件，生成回答，并保存为JSON Lines (.jsonl)格式。
    """
    # 1. 模型加载
    print("开始加载模型...")
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )
    print(f"模型 '{model_name}' 加载完成。")

    # 2. 加载问题文件
    print(f"正在从 {args.question_file} 加载问题...")
    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)
    print(f"共加载 {len(questions)} 个问题。")

    # --- 修改点 1: 移除用于存储所有结果的列表 ---
    # all_responses = [] # 不再需要这个列表

    # --- 修改点 2: 在循环开始前打开输出文件 ---
    output_file_path = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # 使用 'with' 语句管理文件，将整个循环包裹起来
    with open(output_file_path, "w", encoding='utf-8') as ans_file:
        # 3. 遍历问题并生成回答，同时逐行写入文件
        for item in tqdm(questions, desc="正在生成回答"):
            # 从问题项中提取信息
            item_id = item["id"]
            image_file = item["image"]
            query_text = item["query"]
            
            # 构建prompt
            if model.config.mm_use_im_start_end:
                prompt_text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + query_text
            else:
                prompt_text = DEFAULT_IMAGE_TOKEN + '\n' + query_text

            # 使用对话模板
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], prompt_text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # 编码和图像处理
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            try:
                image_path = os.path.join(args.image_folder, image_file)
                image = Image.open(image_path).convert('RGB')
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            except FileNotFoundError:
                print(f"\n[警告] 图像文件未找到: {image_path}。跳过此问题。")
                continue

            # 定义停止标志
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            # 模型生成
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )

            # 解码和清理
            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            # --- 修改点 3: 将单条结果直接写入文件 ---
            # 构建当前结果的字典
            response_dict = {
                "id": item_id,
                "response": outputs
            }
            # 将字典转换为JSON字符串，并添加换行符，然后写入文件
            ans_file.write(json.dumps(response_dict, ensure_ascii=False) + "\n")
            # 如果需要立即看到结果，可以取消下面一行的注释，但会降低I/O性能
            # ans_file.flush()

    # --- 修改点 4: 移除最后的批量写入步骤 ---
    # 因为已经在循环中写入，这里不再需要任何操作

    print(f"\n生成完成！结果已以JSONL格式保存到: {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --- 核心参数 ---
    # 我已将默认值改为了你提供的路径，方便你直接运行测试
    parser.add_argument("--model-path", type=str, default="/data/ruipeng.zhang/OPA-DPO/base_models/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str,  default="/data/ruipeng.zhang/OPA-DPO/base_models/llava-v1.5-7b")
    parser.add_argument("--question-file", type=str, default="/data/ruipeng.zhang/AMBER/data/query/query_all.json")
    parser.add_argument("--image-folder", type=str, default="/data/ruipeng.zhang/V-DPO/playground/AMBER_image")
    # 建议将输出文件后缀改为 .jsonl 以表明其格式
    parser.add_argument("--output-file", type=str, default="/data/ruipeng.zhang/V-DPO/eval/AMBER_output/AMBER_llava_responses_org.jsonl")
    
    # --- 模型和生成参数 ---
    parser.add_argument("--conv-mode", type=str, default="llava_v1", help="对话模板模式。")
    parser.add_argument("--temperature", type=float, default=0.2, help="生成时的温度参数，值越小越确定。")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p (nucleus) sampling 参数。")
    
    args = parser.parse_args()
    
    generate_model_responses(args)
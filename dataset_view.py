import os
import json
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm  #

# 设置路径
dataset_path = "/data/ruipeng.zhang/V-DPO/RLHF-V-Dataset/train"  # 替换成你的保存路径
output_image_base_path = "./output_images"  # 所有图片的根目录（可自定义）
output_json_path = "./output_data.json"     # 最终保存的JSON文件路径

# 加载数据集
dataset = load_from_disk(dataset_path)

# 存储最终数据
json_data = []

# 遍历数据集

for item in tqdm(dataset, desc="Processing dataset"):
    # 保存图片
    rel_img_path = item["image_path"]
    save_img_path = os.path.join(output_image_base_path, rel_img_path)
    save_img_dir = os.path.dirname(save_img_path)
    os.makedirs(save_img_dir, exist_ok=True)

    # 统一转换为RGB模式以避免保存出错
    image = item["image"]
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(save_img_path)

    # 解析text字段
    text_data = json.loads(item["text"])

    # 构造JSON格式条目
    json_item = {
        "image": rel_img_path,  # 相对路径
        "conversations": [
            {"from": "human", "value": text_data["question"]},
            {"from": "gpt", "value": text_data["chosen"]}
        ],
        "rejected": text_data.get("rejected", ""),
        "origin_dataset": item["origin_dataset"],
        "origin_split": item["origin_split"],
        "idx": item["idx"],
    }
    json_data.append(json_item)

# 保存为JSON文件
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)

print(f"处理完成，共保存 {len(json_data)} 条数据。")
print(f"JSON 文件路径: {output_json_path}")
print(f"图像根目录: {output_image_base_path}")

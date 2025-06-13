import re
from collections import defaultdict

file_path = "/data/ruipeng.zhang/V-DPO/model_inspect_output.txt"

def remove_layer_number(name):
    """
    把 layers.数字 替换成 layers，保留前面完整路径
    例如：
    base_model.model.model.layers.24.self_attn.k_proj.lora_A.default.weight
    -> base_model.model.model.layers.self_attn.k_proj.lora_A.default.weight
    """
    return re.sub(r'(layers)\.\d+\.', r'\1.', name)

def extract_full_weight_name(line):
    """
    从一行中提取完整的权重名，去除 layers 中的数字层号
    """
    # 从行首到空格前为权重名部分
    weight_name = line.split()[0]
    weight_name_no_num = remove_layer_number(weight_name)
    return weight_name_no_num

def count_lora_weights(file_path):
    weight_counts = defaultdict(int)
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 只处理包含 lora_A 或 lora_B 的行
            if "lora_A" in line or "lora_B" in line:
                full_name = extract_full_weight_name(line)
                weight_counts[full_name] += 1
    return weight_counts

if __name__ == "__main__":
    counts = count_lora_weights(file_path)
    print("LoRA 权重完整命名（层号已去除）及数量统计：")
    for weight_name, cnt in sorted(counts.items()):
        print(f"{weight_name}: {cnt}")

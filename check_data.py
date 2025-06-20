import json
import os

# --- 1. 定义文件路径 ---
# 基础文件 (来源: mk)
base_file_path = '/data/ruipeng.zhang/dpo_on/RLHF-V-Dataset_4.json'
# 需要合并进来的文件 (来源: vcd)
vcd_file_path = '/data/ruipeng.zhang/dpo_on/Dataset_vcd.json'
# 输出文件
output_file_path = '/data/ruipeng.zhang/dpo_on/RLHF-V-Dataset_mix.json'

print("开始合并JSON文件...")
print(f"将使用 'idx' 作为唯一匹配索引。")

# --- 2. 加载两个JSON文件 ---
try:
    with open(base_file_path, 'r', encoding='utf-8') as f:
        base_data = json.load(f)
    print(f"成功加载基础文件: {base_file_path}，包含 {len(base_data)} 条记录。")

    with open(vcd_file_path, 'r', encoding='utf-8') as f:
        vcd_data = json.load(f)
    print(f"成功加载VCD文件: {vcd_file_path}，包含 {len(vcd_data)} 条记录。")

except FileNotFoundError as e:
    print(f"错误: 文件未找到 - {e}")
    exit()
except json.JSONDecodeError as e:
    print(f"错误: JSON解析失败 - {e}")
    exit()

# --- 3. 创建一个vcd数据的查找字典，以 'idx' 为键 ---
# 使用 idx 作为键，这比使用图像路径更可靠
vcd_lookup = {item['idx']: item for item in vcd_data}
print("已为VCD数据创建基于 'idx' 的快速查找索引。")

# --- 4. 遍历基础数据，进行合并和修改 ---
merged_count = 0
unmatched_count = 0
missing_idx_count = 0
new_data = [] # 创建一个新的列表来存储处理后的数据

for item in base_data:
    idx_key = item.get('idx')
    
    # 检查基础数据中是否存在 idx
    if idx_key is None:
        missing_idx_count += 1
        print(f"警告: 基础数据中的一条记录缺少 'idx' 键，将跳过。记录内容: {str(item)[:200]}...")
        new_data.append(item) # 即使无法匹配，也保留原始数据
        continue

    # 在vcd查找字典中找到对应的记录
    corresponding_vcd_item = vcd_lookup.get(idx_key)

    if corresponding_vcd_item:
        # --- 对 'conversations' 字段进行操作 ---
        # 假设GPT的回答总是在第二个位置 (index 1)
        if len(item['conversations']) > 1 and item['conversations'][1]['from'] == 'gpt':
            # 从base_data中获取value，并重命名key
            mk_value = item['conversations'][1].pop('value', '')
            item['conversations'][1]['mk_value'] = mk_value
            
            # 从vcd_data中获取value，并添加为新key
            if len(corresponding_vcd_item['conversations']) > 1 and corresponding_vcd_item['conversations'][1]['from'] == 'gpt':
                vcd_value = corresponding_vcd_item['conversations'][1].get('value', '')
                item['conversations'][1]['vcd_value'] = vcd_value

        # --- 对 'contrastive_conversations' 字段进行同样的操作 ---
        if 'contrastive_conversations' in item and 'contrastive_conversations' in corresponding_vcd_item:
            if len(item['contrastive_conversations']) > 1 and item['contrastive_conversations'][1]['from'] == 'gpt':
                # 从base_data中获取value，并重命名key
                mk_value_contrastive = item['contrastive_conversations'][1].pop('value', '')
                item['contrastive_conversations'][1]['mk_value'] = mk_value_contrastive
                
                # 从vcd_data中获取value，并添加为新key
                if len(corresponding_vcd_item['contrastive_conversations']) > 1 and corresponding_vcd_item['contrastive_conversations'][1]['from'] == 'gpt':
                    vcd_value_contrastive = corresponding_vcd_item['contrastive_conversations'][1].get('value', '')
                    item['contrastive_conversations'][1]['vcd_value'] = vcd_value_contrastive
        
        merged_count += 1
    else:
        unmatched_count += 1
        print(f"警告: 在VCD文件中未找到与 idx '{idx_key}' 匹配的记录。")
    
    new_data.append(item)


print("\n--- 处理结果摘要 ---")
print(f"成功合并 {merged_count} 条记录。")
if unmatched_count > 0:
    print(f"有 {unmatched_count} 条记录在VCD文件中未找到匹配项。")
if missing_idx_count > 0:
    print(f"有 {missing_idx_count} 条记录在基础文件中缺少 'idx' 键。")
print(f"总计生成 {len(new_data)} 条记录。")
print("---------------------\n")


# --- 5. 将结果写入新的JSON文件 ---
# 确保输出目录存在
output_dir = os.path.dirname(output_file_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(output_file_path, 'w', encoding='utf-8') as f:
    # 使用 indent=2 来格式化输出，使其更易读，类似你给的例子
    json.dump(new_data, f, indent=2, ensure_ascii=False)

print(f"合并后的文件已成功保存到: {output_file_path}")
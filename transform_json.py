import json
import os

# --- 配置 ---
# 输入文件路径
input_file_path = '/data/ruipeng.zhang/dpo_on/RLHF-V-Dataset_3.json'
# 输出文件路径
output_file_path = '/data/ruipeng.zhang/dpo_on/RLHF-V-Dataset_4.json'
# --- 配置结束 ---

def transform_dataset(input_path, output_path):
    """
    读取、转换并保存JSON数据集。
    """
    print(f"开始处理文件...")
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")

    try:
        # 1. 读取原始JSON文件
        with open(input_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        print(f"成功读取 {len(original_data)} 条记录。")

    except FileNotFoundError:
        print(f"错误：输入文件未找到 at '{input_path}'")
        return
    except json.JSONDecodeError:
        print(f"错误：输入文件 '{input_path}' 不是有效的JSON格式。")
        return
    
    transformed_data = []
    
    # 2. 遍历并转换每一条数据
    for i, item in enumerate(original_data):
        try:
            # 创建一个新字典，避免在迭代时修改原始字典
            new_item = item.copy()

            # 提取需要移动的值
            crop_answer = new_item.get('crop_answer_vicrop')
            ori_answer = new_item.get('ori_answer_vicrop')

            if crop_answer is None or ori_answer is None:
                print(f"警告: 第 {i+1} 条记录缺少 'crop_answer_vicrop' 或 'ori_answer_vicrop'，将跳过转换。")
                transformed_data.append(item) # 如果缺少关键字段，可以按原样添加
                continue

            # 3. 更新 'conversations' 和 'contrastive_conversations'
            # 假设'gpt'的回答总是在第二个位置 (index 1)
            if len(new_item['conversations']) > 1:
                new_item['conversations'][1]['value'] = crop_answer
            else:
                print(f"警告: 第 {i+1} 条记录的 'conversations' 结构异常。")

            if len(new_item['contrastive_conversations']) > 1:
                new_item['contrastive_conversations'][1]['value'] = ori_answer
            else:
                print(f"警告: 第 {i+1} 条记录的 'contrastive_conversations' 结构异常。")

            # 4. 删除不再需要的旧字段
            # 使用 pop(key, None) 方式可以避免当key不存在时报错
            new_item.pop('crop_answer_vicrop', None)
            new_item.pop('ori_answer_vicrop', None)
            new_item.pop('rejected', None) # 根据您的例子，也移除了 'rejected'

            transformed_data.append(new_item)

        except (KeyError, IndexError) as e:
            print(f"处理第 {i+1} 条记录时发生错误，格式可能不符: {e}")
            print(f"问题数据: {item}")
            # 如果某条数据格式有问题，可以选择跳过它
            continue
            
    print(f"数据转换完成，共处理 {len(transformed_data)} 条记录。")

    # 5. 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    # 6. 将转换后的数据写入新文件
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # indent=2 使输出的JSON文件格式化，易于阅读
            # ensure_ascii=False 确保中文字符能正确显示
            json.dump(transformed_data, f, indent=2, ensure_ascii=False)
        print(f"成功将转换后的数据保存到: {output_path}")
    except Exception as e:
        print(f"写入输出文件时发生错误: {e}")

# --- 运行脚本 ---
if __name__ == "__main__":
    transform_dataset(input_file_path, output_file_path)
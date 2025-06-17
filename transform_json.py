import json
import os

def convert_json_array_format(input_path, output_path):
    """
    读取一个标准的JSON数组文件，根据指定规则转换格式，并写入新文件。

    Args:
        input_path (str): 输入文件的路径 (必须是JSON数组格式)。
        output_path (str): 输出文件的路径。
    """
    print("🚀 开始处理文件...")
    print(f"    - 输入文件: {input_path}")
    print(f"    - 输出文件: {output_path}")

    try:
        # 1. 一次性读取整个JSON文件
        with open(input_path, 'r', encoding='utf-8') as infile:
            data_list = json.load(infile)

        # 检查是否为列表
        if not isinstance(data_list, list):
            print(f"❌ 错误: 输入文件 {input_path} 的内容不是一个JSON数组 (列表)。请检查文件格式。")
            return

        # 2. 遍历列表中的每一个对象并进行修改
        for i, item in enumerate(data_list):
            # --- 核心转换逻辑 ---
            # a. 获取源数据
            cd_answer = item.get('cd_answer')
            noisy_answer = item.get('noisy_answer')

            # b. 覆盖 conversations
            #    假设 'gpt' 的回答总是在列表的第二个位置 (索引为1)
            if len(item.get('conversations', [])) > 1 and cd_answer is not None:
                item['conversations'][1]['value'] = cd_answer
            else:
                # 给出警告，但继续处理，以防某些条目格式不同
                print(f"⚠️ 警告: 在索引为 {i} 的条目中找不到 'conversations' 的有效结构或 'cd_answer'。")

            # c. 覆盖 contrastive_conversations
            if len(item.get('contrastive_conversations', [])) > 1 and noisy_answer is not None:
                item['contrastive_conversations'][1]['value'] = noisy_answer
            else:
                print(f"⚠️ 警告: 在索引为 {i} 的条目中找不到 'contrastive_conversations' 的有效结构或 'noisy_answer'。")

            # d. (可选但推荐) 清理已使用的字段
            if 'cd_answer' in item:
                del item['cd_answer']
            if 'noisy_answer' in item:
                del item['noisy_answer']
            if 'normal_answer' in item:
                del item['normal_answer']

        # 3. 将修改后的整个列表写入新文件
        #    使用 indent=2 使输出的JSON文件格式化，易于阅读
        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(data_list, outfile, ensure_ascii=False, indent=2)

        print(f"\n✅ 处理完成！共处理 {len(data_list)} 条数据。")

    except FileNotFoundError:
        print(f"❌ 错误: 文件未找到 {input_path}")
    except json.JSONDecodeError:
        print(f"❌ 错误: 文件 {input_path} 不是有效的JSON格式。")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")

# --- 配置 ---
if __name__ == "__main__":
    # 1. 定义你的文件路径
    input_file_path = '/data/ruipeng.zhang/dpo_on/RLHF-V-Dataset_vcd.json'
    
    output_file_path = '/data/ruipeng.zhang/dpo_on/Dataset_vcd.json'
    
    # 3. 执行转换函数
    convert_json_array_format(input_file_path, output_file_path)
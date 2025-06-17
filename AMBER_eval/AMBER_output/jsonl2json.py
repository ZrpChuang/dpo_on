import json

# --- 配置 ---
# 输入的 JSONL 文件名
input_file_path = '/data/ruipeng.zhang/dpo_on/AMBER_eval/AMBER_output/AMBER_llava_responses_hfv_r128.jsonl'
# 输出的 JSON 文件名
output_file_path = '/data/ruipeng.zhang/dpo_on/AMBER_eval/AMBER_output/AMBER_llava_responses_hfv_r128.json'

# --- 脚本 ---

# 1. 创建一个空列表，用来存储从文件中读取的每一个 JSON 对象
all_objects = []

print(f"正在读取文件: {input_file_path}...")

try:
    # 2. 以只读模式 ('r') 打开 JSONL 文件
    # 使用 'with' 语句可以确保文件在使用后被自动关闭
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        # 3. 逐行读取文件
        for line in infile:
            # 去除每行末尾可能存在的换行符和空格
            line = line.strip()
            # 确保不是空行
            if line:
                try:
                    # 4. 将当前行的字符串解析成 Python 字典（JSON 对象）
                    obj = json.loads(line)
                    # 5. 将解析后的对象添加到列表中
                    all_objects.append(obj)
                except json.JSONDecodeError:
                    print(f"警告: 发现格式错误的行，已跳过: {line}")

    # 6. 以写入模式 ('w') 打开输出文件
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        # 7. 将整个列表转换成 JSON 格式的字符串并写入文件
        #    - indent=4: 使输出的 JSON 文件格式化，有 4 个空格的缩进，更易读
        #    - ensure_ascii=False: 确保像中文这样的非 ASCII 字符能被正确写入，而不是转义码
        json.dump(all_objects, outfile, indent=4, ensure_ascii=False)

    print(f"转换成功！结果已保存到: {output_file_path}")

except FileNotFoundError:
    print(f"错误: 输入文件 '{input_file_path}' 未找到。请检查文件名和路径是否正确。")
except Exception as e:
    print(f"发生了一个意外错误: {e}")
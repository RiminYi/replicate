import os
import csv
import json
from data_loader import preprocess_text # 从我们自己的模块中导入函数

def load_seed_bigrams(file_path):
    """
    从CSV文件中加载种子二元词组。

    参数:
    file_path (str): 种子词组CSV文件的路径。

    返回:
    set: 一个包含二元词组元组的集合，例如 {('climate', 'change'), ...}
    """
    seed_bigrams = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            # 每一行可能有多个词组，需要处理空值
            for item in row:
                item = item.strip() # 去除首尾空格
                # 确保非空且是两个词的组合
                if item:
                    words = item.split() # 分割成单词
                    if len(words) == 2:
                        seed_bigrams.add(tuple(words))
    return seed_bigrams

def contains_seed_bigram(sentence_tokens, seed_bigrams):
    """
    检查一个已分词的句子是否包含任何一个种子二元词组。

    参数:
    sentence_tokens (list): 代表一个句子的单词列表。
    seed_bigrams (set): 种子二元词组的集合。

    返回:
    bool: 如果包含则返回True，否则返回False。
    """
    # 确保句子至少有两个词，否则直接判断不合要求
    if len(sentence_tokens) < 2:
        return False
    # 从句子中生成二元词组并检查是否存在于种子集中
    sentence_bigrams = set(zip(sentence_tokens, sentence_tokens[1:])) # 使用zip生成二元词组，是一个idiom，快速高效
    # 检查是否有交集：isdisjoint() 方法检查两个集合是否没有共同元素
    return not sentence_bigrams.isdisjoint(seed_bigrams)

def build_sets(raw_data_path, seed_bigrams_path, output_path):
    """
    遍历原始数据，构建参照集R和搜索集S，并将其保存为JSON文件。

    参数:
    raw_data_path (str): 存放所有原始文本的根目录。
    seed_bigrams_path (str): 种子词组文件的路径。
    output_path (str): 处理后文件（R和S）的输出目录。
    """
    # 1. 加载种子词组
    seed_bigrams = load_seed_bigrams(seed_bigrams_path)
    if not seed_bigrams:
        print("Error: No seed bigrams loaded. Exiting.")
        return

    print(f"Loaded {len(seed_bigrams)} seed bigrams.")

    reference_set_R = []
    search_set_S = []

    # 2. 遍历所有原始文本文件
    for root, _, files in os.walk(raw_data_path): # _表示我们不需要子目录列表，忽略即可。
        # os.walk() 会遍历目录树，返回一个三元组 (root, dirs, files),root是当前目录路径，dirs是子目录列表，files是文件列表
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                
                # 3. 使用我们之前写的函数进行预处理
                processed_sentences = preprocess_text(file_path)
                
                if processed_sentences is None:
                    continue

                # 4. 对每个句子进行分类
                for sentence in processed_sentences:
                    if contains_seed_bigram(sentence, seed_bigrams):
                        reference_set_R.append(sentence)
                    else:
                        search_set_S.append(sentence)
    
    print("\n--- Set Building Complete ---")
    print(f"Reference Set (R) contains: {len(reference_set_R)} sentences.")
    print(f"Search Set (S) contains: {len(search_set_S)} sentences.")

    # 5. 保存结果
    os.makedirs(output_path, exist_ok=True) # 确保输出目录存在
    
    path_r = os.path.join(output_path, 'reference_set_R.json')
    with open(path_r, 'w', encoding='utf-8') as f:
        json.dump(reference_set_R, f, indent=2)
    print(f"Reference Set (R) saved to: {path_r}")

    path_s = os.path.join(output_path, 'search_set_S.json')
    with open(path_s, 'w', encoding='utf-8') as f:
        json.dump(search_set_S, f, indent=2)
    print(f"Search Set (S) saved to: {path_s}")


if __name__ == '__main__':
    # 定义文件路径
    RAW_DATA_PATH = os.path.join('data', 'raw')
    SEED_BIGRAMS_PATH = os.path.join(RAW_DATA_PATH, 'initial_seed_bigrams.csv')
    PROCESSED_DATA_PATH = os.path.join('data', 'processed')

    build_sets(RAW_DATA_PATH, SEED_BIGRAMS_PATH, PROCESSED_DATA_PATH)
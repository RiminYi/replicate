import os
import json
import csv
from collections import Counter

def get_bigram_document_frequency(list_of_sentences):
    """
    计算一个句子列表中，每个二元词组的文档频率（DF）。
    文档频率指的是包含该词组的句子数量。
    """
    # 使用Counter来高效计数
    df_counter = Counter()
    for sentence_tokens in list_of_sentences:
        if len(sentence_tokens) < 2:
            continue
        # 使用set确保每个句子中的同一个bigram只被计数一次
        sentence_bigrams = set(zip(sentence_tokens, sentence_tokens[1:]))
        df_counter.update(sentence_bigrams)
    return df_counter

def discover_new_bigrams(processed_data_path, seed_bigrams_path):
    """
    通过比较目标集T和非目标集(S\T)的词组频率来发现新词组。
    """
    # 1. 加载所需数据
    print("--- Loading T, S, and initial seed bigrams ---")
    with open(os.path.join(processed_data_path, 'target_set_T.json'), 'r') as f:
        target_set_T = [tuple(s) for s in json.load(f)]
    with open(os.path.join(processed_data_path, 'search_set_S.json'), 'r') as f:
        search_set_S = [tuple(s) for s in json.load(f)]
    
    with open(seed_bigrams_path, 'r') as f:
        initial_seeds = {tuple(row[0].split()) for row in csv.reader(f) if row}

    # 2. 创建非目标集 (S \ T)
    # 将列表转换为集合以高效地计算差集
    set_T = set(target_set_T)
    set_S = set(search_set_S)
    non_target_set_S_minus_T = list(set_S - set_T)
    
    print(f"Target Set (T) size: {len(target_set_T)}")
    print(f"Non-Target Set (S\\T) size: {len(non_target_set_S_minus_T)}")

    if not target_set_T:
        print("Warning: Target Set T is empty. No new bigrams can be discovered.")
        return list(initial_seeds) # 只返回初始种子

    # 3. 计算T和S\T中每个bigram的文档频率
    print("\n--- Calculating Document Frequencies ---")
    df_t = get_bigram_document_frequency(target_set_T)
    df_s_minus_t = get_bigram_document_frequency(non_target_set_S_minus_T)

    # 4. 发现新词组
    print("--- Discovering New Bigrams by comparing frequencies ---")
    newly_discovered_bigrams = set()
    
    # 遍历所有在T中出现过的词组作为候选
    for bigram, count_in_t in df_t.items():
        # 计算在T中的文档频率 (DF)
        freq_in_t = count_in_t / len(target_set_T)
        
        # 计算在S\T中的文档频率 (DF)
        count_in_s_minus_t = df_s_minus_t.get(bigram, 0) # 如果不存在，则计数为0
        
        # 防止除以零
        if len(non_target_set_S_minus_T) > 0:
            freq_in_s_minus_t = count_in_s_minus_t / len(non_target_set_S_minus_T)
        else:
            freq_in_s_minus_t = 0
            
        # 这就是论文中的核心辨别逻辑：如果一个词组在T中的频率更高，则保留
        if freq_in_t > freq_in_s_minus_t:
            newly_discovered_bigrams.add(bigram)

    print(f"Found {len(newly_discovered_bigrams)} new potential bigrams.")
    
    # 5. 合并初始种子词组和新发现的词组
    final_bigram_library = initial_seeds.union(newly_discovered_bigrams)
    
    # 注意: 为了简化，我们在这里只保留频率更高的词组。
    # 原论文中还会使用更复杂的似然度量进行排序和筛选，这是一个可以深入优化的方向。
    
    return [list(bg) for bg in final_bigram_library]


if __name__ == '__main__':
    PROCESSED_DATA_PATH = os.path.join('data', 'processed')
    RAW_DATA_PATH = os.path.join('data', 'raw')
    SEED_BIGRAMS_PATH = os.path.join(RAW_DATA_PATH, 'initial_seed_bigrams.csv')

    final_library = discover_new_bigrams(PROCESSED_DATA_PATH, SEED_BIGRAMS_PATH)
    
    print(f"\n--- Final Bigram Library Created ---")
    print(f"Total bigrams in the final library: {len(final_library)}")

    # 6. 保存最终的词库
    output_path = os.path.join(PROCESSED_DATA_PATH, 'final_bigram_library.csv')
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['bigram_word1', 'bigram_word2']) # 写入表头
        # 将 ('word1', 'word2') 格式的元组写入CSV
        for bigram_tuple in sorted(final_library):
            writer.writerow(bigram_tuple)
            
    print(f"Final bigram library saved to: {output_path}")
    print("\n--- Example new bigrams (if any) ---")
    print(final_library[:10])
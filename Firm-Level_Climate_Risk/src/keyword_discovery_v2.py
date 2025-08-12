import os
import json
import csv
import math
from collections import Counter

def get_bigram_document_frequency(list_of_sentences):
    """计算一个句子列表中，每个二元词组的文档频率（DF）。"""
    df_counter = Counter()
    for sentence_tokens in list_of_sentences:
        if len(sentence_tokens) < 2:
            continue
        sentence_bigrams = set(zip(sentence_tokens, sentence_tokens[1:]))
        df_counter.update(sentence_bigrams)
    return df_counter

# --- 新增的函数：计算似然度分数 ---
def calculate_likelihood_score(n_k_T, n_k_S_minus_T, N_T, N_S_minus_T):
    """
    根据 King et al. (2017) 的公式计算似然度分数。
    使用对数伽马函数以保证计算稳定性。
    """
    # 论文中设定 alpha 值为 1
    alpha_T = 1
    alpha_S_minus_T = 1
    
    # 派生其他变量
    n_not_k_T = N_T - n_k_T
    n_not_k_S_minus_T = N_S_minus_T - n_k_S_minus_T
    n_k_S = n_k_T + n_k_S_minus_T
    n_not_k_S = n_not_k_T + n_not_k_S_minus_T

    # --- 公式的第一部分 ---
    term1_num = math.lgamma(n_k_T + alpha_T) + math.lgamma(n_k_S_minus_T + alpha_S_minus_T)
    term1_den = math.lgamma(n_k_S + alpha_T + alpha_S_minus_T)
    term1 = term1_num - term1_den
    
    # --- 公式的第二部分 ---
    term2_num = math.lgamma(n_not_k_T + alpha_T) + math.lgamma(n_not_k_S_minus_T + alpha_S_minus_T)
    term2_den = math.lgamma(n_not_k_S + alpha_T + alpha_S_minus_T)
    term2 = term2_num - term2_den
    
    # 返回对数似然度
    return term1 + term2

# --- 修改后的主函数 ---
def discover_new_bigrams(processed_data_path, seed_bigrams_path):
    """
    通过比较目标集T和非目标集(S\T)的词组频率来发现新词组，
    并使用似然度分数进行排序和筛选。
    """
    # 1. 加载所需数据 (与之前相同)
    print("--- Loading T, S, and initial seed bigrams ---")
    with open(os.path.join(processed_data_path, 'target_set_T.json'), 'r') as f:
        target_set_T = [tuple(s) for s in json.load(f)]
    with open(os.path.join(processed_data_path, 'search_set_S.json'), 'r') as f:
        search_set_S = [tuple(s) for s in json.load(f)]
    with open(seed_bigrams_path, 'r') as f:
        initial_seeds = {tuple(row[0].split()) for row in csv.reader(f) if row}

    # 2. 创建非目标集 (S \ T) (与之前相同)
    set_T = set(target_set_T)
    set_S = set(search_set_S)
    non_target_set_S_minus_T = list(set_S - set_T)
    
    N_T = len(target_set_T)
    N_S_minus_T = len(non_target_set_S_minus_T)
    
    print(f"Target Set (T) size: {N_T}")
    print(f"Non-Target Set (S\\T) size: {N_S_minus_T}")

    if N_T == 0:
        print("Warning: Target Set T is empty. No new bigrams can be discovered.")
        return [list(bg) for bg in initial_seeds]

    # 3. 计算T和S\T中每个bigram的文档频率 (与之前相同)
    print("\n--- Calculating Document Frequencies ---")
    df_t = get_bigram_document_frequency(target_set_T)
    df_s_minus_t = get_bigram_document_frequency(non_target_set_S_minus_T)

    # 4. 筛选、计算分数并排序
    print("--- Calculating likelihood scores for candidate bigrams ---")
    scored_bigrams = []
    
    for bigram, n_k_T in df_t.items():
        freq_in_t = n_k_T / N_T
        n_k_S_minus_T = df_s_minus_t.get(bigram, 0)
        freq_in_s_minus_t = n_k_S_minus_T / N_S_minus_T if N_S_minus_T > 0 else 0
        
        if freq_in_t > freq_in_s_minus_t:
            score = calculate_likelihood_score(n_k_T, n_k_S_minus_T, N_T, N_S_minus_T)
            scored_bigrams.append((bigram, score))

    # 按分数降序排序
    scored_bigrams.sort(key=lambda x: x[1], reverse=True)
    
    # 5. 应用 "top 5%" 规则
    num_to_keep = int(len(scored_bigrams) * 0.05)
    
    # 确保至少保留几个，以防列表太短
    if len(scored_bigrams) > 0 and num_to_keep == 0:
        num_to_keep = 1
        
    top_bigrams = {item[0] for item in scored_bigrams[:num_to_keep]}
    
    print(f"Kept top 5% of candidates: {num_to_keep} new bigrams.")
    
    # 6. 合并初始种子词组和新发现的词组
    final_bigram_library = initial_seeds.union(top_bigrams)
    
    return [list(bg) for bg in final_bigram_library]

# if __name__ == '__main__': 部分保持不变
if __name__ == '__main__':
    PROCESSED_DATA_PATH = os.path.join('data', 'processed')
    RAW_DATA_PATH = os.path.join('data', 'raw')
    SEED_BIGRAMS_PATH = os.path.join(RAW_DATA_PATH, 'initial_seed_bigrams.csv')

    final_library = discover_new_bigrams(PROCESSED_DATA_PATH, SEED_BIGRAMS_PATH)
    
    print(f"\n--- Final Bigram Library Created ---")
    print(f"Total bigrams in the final library: {len(final_library)}")

    output_path = os.path.join(PROCESSED_DATA_PATH, 'final_bigram_library.csv')
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['bigram_word1', 'bigram_word2'])
        for bigram_tuple in sorted([tuple(bg) for bg in final_library]):
            writer.writerow(bigram_tuple)
            
    print(f"Final bigram library saved to: {output_path}")
    print("\n--- Example new bigrams (if any) ---")
    # 为了能打印，转换回元组列表
    print([tuple(bg) for bg in sorted(final_library)][:10])
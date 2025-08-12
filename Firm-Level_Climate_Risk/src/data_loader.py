import os
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk


def preprocess_text(file_path):
    """
    加载并预处理单个文本文件。

    步骤:
    1. 读取文件内容。
    2. 分句。
    3. 对每个句子进行分词。
    4. 转换为小写。
    5. 移除标点符号和数字。
    6. 移除停用词。
    7. 对单词进行词形还原。

    参数:
    file_path (str): 原始文本文件的路径。

    返回:
    list: 一个列表，其中每个元素是代表一个句子的单词列表。
          例如: [['this', 'be', 'a', 'sentence'], ['this', 'be', 'another']]
    """
    print(f"Preprocessing file: {file_path}")
    
    # 初始化词形还原器和停用词列表
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    processed_sentences = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 1. 分句
        sentences = sent_tokenize(text)

        for sentence in sentences:
            # 2. 分词
            words = word_tokenize(sentence)
            
            # 3. 转换为小写并处理
            lemmatized_words = []
            for word in words:
                word_lower = word.lower()
                # 4. 移除标点和数字，并确保是纯字母
                if word_lower.isalpha() and word_lower not in stop_words:
                    # 5. 词形还原
                    lemmatized_word = lemmatizer.lemmatize(word_lower)
                    lemmatized_words.append(lemmatized_word)
            
            if lemmatized_words: # 仅添加非空句子
                processed_sentences.append(lemmatized_words)

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
        
    return processed_sentences

# --- 用于直接运行和测试此脚本的模块 ---
if __name__ == '__main__':
    print("--- Testing data_loader.py ---")
    
    # 构建到我们模拟数据文件的路径
    # 注意: 请确保从项目根目录 (climate_keyword_discovery/) 运行此脚本
    mock_file = os.path.join('data', 'raw', 'transcripts', 'energy_co_q3_2025.txt')

    # 调用预处理函数
    processed_data = preprocess_text(mock_file)

    if processed_data:
        print("\n--- Preprocessing Successful ---")
        print(f"Original file path: {mock_file}")
        print(f"Total sentences processed: {len(processed_data)}")
        print("\n--- Example Output (first 3 sentences) ---")
        for i, sentence in enumerate(processed_data[:3]):
            print(f"Sentence {i+1}: {sentence}")
import os
import json
import joblib

def load_assets(models_path):
    """加载所有必需的模型和向量化器。"""
    vectorizer_path = os.path.join(models_path, 'tfidf_vectorizer.joblib')
    nb_path = os.path.join(models_path, 'naivebayes_classifier.joblib')
    svc_path = os.path.join(models_path, 'svc_classifier.joblib')
    rf_path = os.path.join(models_path, 'randomforest_classifier.joblib')

    try:
        vectorizer = joblib.load(vectorizer_path)
        nb_model = joblib.load(nb_path)
        svc_model = joblib.load(svc_path)
        rf_model = joblib.load(rf_path)
        
        classifiers = [nb_model, svc_model, rf_model]
        print("Successfully loaded vectorizer and all 3 models.")
        return vectorizer, classifiers
    except FileNotFoundError as e:
        print(f"Error loading model or vectorizer: {e}")
        print("Please ensure you have run the model_trainer.py script successfully.")
        return None, None

def create_target_set(search_set, vectorizer, classifiers, threshold=0.15):
    """
    使用训练好的模型扫描搜索集S，创建目标集T。

    参数:
    search_set (list): 搜索集S，每个元素是一个句子的单词列表。
    vectorizer: 已经fit过的TfidfVectorizer。
    classifiers (list): 包含三个训练好的分类器模型的列表。
    threshold (float): 将句子归入目标集T的概率阈值。

    返回:
    list: 目标集T，结构与搜索集S相同。
    """
    target_set_T = []
    
    total_sentences = len(search_set)
    print(f"Scanning {total_sentences} sentences in the Search Set (S)...")

    for i, sentence_tokens in enumerate(search_set):
        # 为了显示进度
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{total_sentences} sentences...")

        # 必须将分好词的句子用空格连接起来，以匹配vectorizer的输入格式
        sentence_text = " ".join(sentence_tokens)
        
        # 使用加载的vectorizer进行transform，注意不能用fit_transform，这是因为现在只需要在测试中转换（transform），而不是训练（fit）
        # vectorizer.transform需要一个可迭代对象，所以传入列表
        vectorized_sentence = vectorizer.transform([sentence_text])
        
        # 检查是否有任何一个分类器给出的概率超过阈值
        is_target = False
        for model in classifiers:
            # predict_proba返回一个数组，[[prob_class_0, prob_class_1]]
            # 我们需要的是类别1（气候相关）的概率
            probability_class_1 = model.predict_proba(vectorized_sentence)[0, 1] # 获取类别1的概率。[0,1]表示第一个样本的第二个类别的概率
            
            if probability_class_1 > threshold:
                is_target = True
                break # 只要有一个模型满足条件，就无需再检查其他模型
        
        if is_target:
            target_set_T.append(sentence_tokens)
            
    return target_set_T

if __name__ == '__main__':
    PROCESSED_DATA_PATH = os.path.join('data', 'processed')
    MODELS_PATH = 'models'

    # 1. 加载模型和向量化器
    print("--- Loading Models and Vectorizer ---")
    vectorizer, models = load_assets(MODELS_PATH)

    if vectorizer and models:
        # 2. 加载搜索集S
        print("\n--- Loading Search Set (S) ---")
        path_s = os.path.join(PROCESSED_DATA_PATH, 'search_set_S.json')
        with open(path_s, 'r', encoding='utf-8') as f:
            search_set_S = json.load(f)
        
        # 3. 创建目标集T
        target_set_T = create_target_set(search_set_S, vectorizer, models)

        # 4. 保存目标集T
        print(f"\n--- Prediction Complete ---")
        print(f"Found {len(target_set_T)} sentences for the Target Set (T).")
        
        output_path_t = os.path.join(PROCESSED_DATA_PATH, 'target_set_T.json')
        with open(output_path_t, 'w', encoding='utf-8') as f:
            json.dump(target_set_T, f, indent=2)
        
        print(f"Target Set (T) saved to: {output_path_t}")
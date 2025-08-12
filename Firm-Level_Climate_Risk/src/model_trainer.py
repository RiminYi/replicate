import os
import json
import random
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def load_data(processed_data_path):
    """加载处理好的R集和S集。"""
    path_r = os.path.join(processed_data_path, 'reference_set_R.json')
    path_s = os.path.join(processed_data_path, 'search_set_S.json')
    
    with open(path_r, 'r', encoding='utf-8') as f:
        reference_set_R = json.load(f)
    with open(path_s, 'r', encoding='utf-8') as f:
        search_set_S = json.load(f)
        
    return reference_set_R, search_set_S

def prepare_training_data(reference_set, search_set, num_samples_from_s):
    """
    生成训练集，
    准备训练用的X（文本）和y（标签）。
    论文中从S集随机抽样100,000句。
    """
    # 正样本 (标签为1)
    positive_samples = [" ".join(sentence) for sentence in reference_set]
    positive_labels = [1] * len(positive_samples) # 输出的结果是1的标签，确保与正样本一一对应
    
    # 从S集中随机抽样作为负样本 (标签为0)
    # 使用 min() 防止样本数超过S集本身大小
    num_samples = min(num_samples_from_s, len(search_set))
    negative_samples_sentences = random.sample(search_set, num_samples)
    negative_samples = [" ".join(sentence) for sentence in negative_samples_sentences] # 将分词后的句子重新连接成字符串
    negative_labels = [0] * len(negative_samples) # 输出的结果是0的标签，确保与负样本一一对应
    
    # 合并数据
    X_train_text = positive_samples + negative_samples
    y_train = positive_labels + negative_labels
    
    return X_train_text, y_train

def train_and_save_models(X_train_text, y_train, models_output_path):
    """
    向量化文本，训练、调优并保存三个模型。
    """
    # 1. TF-IDF 向量化
    print("Vectorizing text data with TF-IDF...")
    # 注意: 我们将lemmatized的词用空格连接起来，TfidfVectorizer会处理它
    # TF-IDF向量化器会自动处理文本数据
    # 这里的fit_transform会将文本转换为TF-IDF矩阵
    # 注意: 这里的X_train_text是一个列表，每个元素是一个句子的文本
    # 例如: ["this is a sentence", "this is another"]
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    
    # 保存vectorizer，因为后续预测新数据时需要用同一个
    os.makedirs(models_output_path, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(models_output_path, 'tfidf_vectorizer.joblib'))
    print("Vectorizer saved.")

    # 2. 定义模型和参数网格
    # 注意: 这里的参数网格为了快速演示已大幅简化
    classifiers = {
        'NaiveBayes': (MultinomialNB(), {'alpha': [1.0, 0.5, 0.1, 0.05, 0.01]}),
        'SVC': (SVC(probability=True), {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 0.1, 0.01, 0.001]}),
        'RandomForest': (RandomForestClassifier(), {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]})
    }

    # 3. 循环训练、调优和保存每个模型
    for name, (model, params) in classifiers.items():
        print(f"\n--- Training {name} ---")
        grid_search = GridSearchCV(model, params, cv=3, n_jobs=-1, verbose=1) # cv=3 为了快速，意思是三折交叉验证；n_jobs=-1 使用所有可用的CPU核心; verbose=1 显示进度
        grid_search.fit(X_train_tfidf, y_train)
        
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        
        # 获取最佳模型并保存
        best_model = grid_search.best_estimator_
        model_path = os.path.join(models_output_path, f'{name.lower()}_classifier.joblib')
        joblib.dump(best_model, model_path)
        print(f"{name} model saved to: {model_path}")

if __name__ == '__main__':
    random.seed(42) # 为了可重复性，设置一个随机种子
    PROCESSED_DATA_PATH = os.path.join('data', 'processed')
    MODELS_OUTPUT_PATH = 'models'
    NUM_S_SAMPLES = 500 # 在我们的模拟数据上设为500即可

    print("--- Loading Data ---")
    R, S = load_data(PROCESSED_DATA_PATH)
    print(f"Loaded {len(R)} sentences for Reference Set (R).")
    print(f"Loaded {len(S)} sentences for Search Set (S).")

    print("\n--- Preparing Training Data ---")
    X_text, y = prepare_training_data(R, S, NUM_S_SAMPLES)
    print(f"Created training set with {len(X_text)} samples.")

    train_and_save_models(X_text, y, MODELS_OUTPUT_PATH)
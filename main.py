from data_loading import load_data
from data_preprocessing import preprocess_data
from feature_engineering import create_features
from model_training import train_model
from evaluation import evaluate_model
from utils import memory_usage

def main():
    # Загрузка данных
    smm_data, zvuk_data = load_data()

    # Предобработка данных
    smm_data, zvuk_data = preprocess_data(smm_data, zvuk_data)

    # Создание признаков
    smm_features, zvuk_features = create_features(smm_data, zvuk_data)

    # Обучение модели
    features = ['product_id', 'time_diff', 'song_id', 'average_rating', 'mean', 'median', 'std', 'min', 'max', 'unique_users']
    target = 'rating'
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_at': [10],
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbosity': -1
    }

    model = train_model(smm_features, features, target, params)

    # Оценка модели
    evaluate_model(model, zvuk_features, features, target)

    # Проверка использования памяти
    memory_usage()

if __name__ == "__main__":
    main()
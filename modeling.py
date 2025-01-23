import lightgbm as lgb
import numpy as np

# Настройки LightGBM
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

# Оценка NDCG@10
def ndcg_at_k(y_true, y_score, k=10):
    y_true = y_true.reshape(-1)
    y_score = y_score.reshape(-1)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gain = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts) / np.sum(2 ** np.sort(y_true)[::-1] / discounts)

# Функция для предсказания и формирования ответа
def predict_and_generate_submission(model, test_data, features, output_file):
    test_data['predicted_score'] = model.predict(test_data[features])
    recommendations = (
        test_data
        .sort_values(by=['user_id', 'predicted_score'], ascending=[True, False])
        .groupby('user_id')
        .agg(item_ids=('product_id', lambda x: x.head(10).tolist()))
        .reset_index()
    )
    recommendations['index'] = np.arange(len(recommendations))
    recommendations = recommendations[['index', 'user_id', 'item_ids']]
    recommendations.to_parquet(output_file, engine='pyarrow', index=False)
    print(f"Файл {output_file} успешно сохранен!")

# Функция для обучения модели
def train_model(train_data, group):
    dataset = lgb.Dataset(train_data, group=group)
    model = lgb.train(params, dataset, num_boost_round=100)
    return model

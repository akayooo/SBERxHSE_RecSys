import pandas as pd
from data_processing import type_preparation_user_data, type_preparation_items_data, limit_group_size, calculate_stats
from modeling import train_model, predict_and_generate_submission

# Загрузка данных
zvuk_data = pd.read_parquet('train_zvuk.parquet')
smm_data = pd.read_parquet('train_smm.parquet')

# Подготовка временных признаков
max_timestamp_zvuk = zvuk_data['timestamp'].max()
max_timestamp_smm = smm_data['timestamp'].max()
zvuk_data['time_diff'] = max_timestamp_zvuk - zvuk_data['timestamp']
smm_data['time_diff'] = max_timestamp_smm - smm_data['timestamp']

# Преобразование типов
smm_data = type_preparation_user_data(smm_data)
zvuk_data = type_preparation_user_data(zvuk_data)

# Объединение данных
merged_data = pd.merge(smm_data, zvuk_data, on='user_id', suffixes=('_smm', '_zvuk'))

# Группировка и вычисление средних оценок
average_rating_products_for_songs = merged_data.groupby(['item_id_smm', 'item_id_zvuk'])['rating_zvuk'].mean().reset_index()
average_rating_products_for_songs.rename(columns={'item_id_smm': 'product_id', 'item_id_zvuk': 'song_id', 'rating_zvuk': 'average_rating'}, inplace=True)

average_rating_songs_for_products = merged_data.groupby(['item_id_zvuk', 'item_id_smm'])['rating_smm'].mean().reset_index()
average_rating_songs_for_products.rename(columns={'item_id_zvuk': 'song_id', 'item_id_smm': 'product_id', 'rating_smm': 'average_rating'}, inplace=True)

# Расчёт статистических признаков
stats_products_for_songs = merged_data.groupby('item_id_zvuk')['rating_smm'].apply(calculate_stats).unstack()
stats_products_for_songs.reset_index(inplace=True)
stats_products_for_songs.rename(columns={'item_id_zvuk': 'song_id'}, inplace=True)

stats_songs_for_products = merged_data.groupby('item_id_smm')['rating_zvuk'].apply(calculate_stats).unstack()
stats_songs_for_products.reset_index(inplace=True)
stats_songs_for_products.rename(columns={'item_id_smm': 'product_id'}, inplace=True)

# Добавление статистик
average_rating_products_for_songs = pd.merge(average_rating_products_for_songs, stats_products_for_songs, on='song_id')
average_rating_songs_for_products = pd.merge(average_rating_songs_for_products, stats_songs_for_products, on='product_id')

# Уникальные пользователи
unique_users_per_product = smm_data.groupby('item_id')['user_id'].nunique().reset_index()
unique_users_per_product.rename(columns={'item_id': 'product_id', 'user_id': 'unique_users'}, inplace=True)

unique_users_per_song = zvuk_data.groupby('item_id')['user_id'].nunique().reset_index()
unique_users_per_song.rename(columns={'item_id': 'song_id', 'user_id': 'unique_users'}, inplace=True)

# Финальное объединение
smm_features = pd.merge(average_rating_products_for_songs, unique_users_per_song, on='song_id')
zvuk_features = pd.merge(average_rating_songs_for_products, unique_users_per_product, on='product_id')

smm_features = type_preparation_items_data(smm_features)
zvuk_features = type_preparation_items_data(zvuk_features)

# Приведение рейтингов к диапазону 0-10
smm_data['rating'] = ((smm_data['rating'] - smm_data['rating'].min()) / (smm_data['rating'].max() - smm_data['rating'].min()) * 20).round().astype(int)
zvuk_data['rating'] = ((zvuk_data['rating'] - zvuk_data['rating'].min()) / (zvuk_data['rating'].max() - zvuk_data['rating'].min()) * 20).round().astype(int)

# Ограничение групп до 10,000 записей
train_data_smm = smm_data.groupby('user_id', group_keys=False).apply(lambda x: limit_group_size(x, max_size=10000))
train_data_zvuk = zvuk_data.groupby('user_id', group_keys=False).apply(lambda x: limit_group_size(x, max_size=10000))

# Пересчёт групп
train_group_smm = train_data_smm.groupby('user_id').size().tolist()
train_group_zvuk = train_data_zvuk.groupby('user_id').size().tolist()

# Обучение модели
model_smm = train_model(train_data_smm, train_group_smm)
model_zvuk = train_model(train_data_zvuk, train_group_zvuk)

# Предсказание и сохранение результатов
predict_and_generate_submission(model_smm, train_data_smm, ['product_id'], 'smm_output')
predict_and_generate_submission(model_zvuk, train_data_zvuk, ['song_id'], 'zvuk_output')

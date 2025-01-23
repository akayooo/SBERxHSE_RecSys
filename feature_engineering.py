import pandas as pd
from utils import calculate_stats
from data_preprocessing import type_preparation_items_data, type_preparation_user_data

def create_features(smm_data, zvuk_data):
    merged_data = pd.merge(smm_data, zvuk_data, on='user_id', suffixes=('_smm', '_zvuk'))
    average_rating_products_for_songs = merged_data.groupby(['item_id_smm', 'item_id_zvuk'])['rating_zvuk'].mean().reset_index()
    average_rating_products_for_songs.rename(columns={'item_id_smm': 'product_id', 'item_id_zvuk': 'song_id', 'rating_zvuk': 'average_rating'}, inplace=True)

    merged_data = pd.merge(zvuk_data, smm_data, on='user_id', suffixes=('_zvuk', '_smm'))
    average_rating_songs_for_products = merged_data.groupby(['item_id_zvuk', 'item_id_smm'])['rating_smm'].mean().reset_index()
    average_rating_songs_for_products.rename(columns={'item_id_zvuk': 'song_id', 'item_id_smm': 'product_id', 'rating_smm': 'average_rating'}, inplace=True)

    stats_products_for_songs = merged_data.groupby('item_id_zvuk')['rating_smm'].apply(calculate_stats).unstack()
    stats_products_for_songs.reset_index(inplace=True)
    stats_products_for_songs.rename(columns={'item_id_zvuk': 'song_id'}, inplace=True)

    stats_songs_for_products = merged_data.groupby('item_id_smm')['rating_zvuk'].apply(calculate_stats).unstack()
    stats_songs_for_products.reset_index(inplace=True)
    stats_songs_for_products.rename(columns={'item_id_smm': 'product_id'}, inplace=True)

    average_rating_products_for_songs = pd.merge(average_rating_products_for_songs, stats_products_for_songs, on='song_id')
    average_rating_songs_for_products = pd.merge(average_rating_songs_for_products, stats_songs_for_products, on='product_id')
    del stats_products_for_songs, stats_songs_for_products

    unique_users_per_product = smm_data.groupby('item_id')['user_id'].nunique().reset_index()
    unique_users_per_product.rename(columns={'item_id': 'product_id', 'user_id': 'unique_users'}, inplace=True)

    unique_users_per_song = zvuk_data.groupby('item_id')['user_id'].nunique().reset_index()
    unique_users_per_song.rename(columns={'item_id': 'song_id', 'user_id': 'unique_users'}, inplace=True)

    smm_features = pd.merge(average_rating_products_for_songs, unique_users_per_song, on='song_id')
    zvuk_features = pd.merge(average_rating_songs_for_products, unique_users_per_product, on='product_id')
    del average_rating_products_for_songs, average_rating_songs_for_products

    smm_features = type_preparation_items_data(smm_features)
    zvuk_features = type_preparation_items_data(zvuk_features)

    return smm_features, zvuk_features
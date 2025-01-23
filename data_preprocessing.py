import pandas as pd

def type_preparation_user_data(data):
    data['user_id'] = data['user_id'].astype('uint32')
    data['timestamp'] = data['timestamp'].astype('uint64')
    data['item_id'] = data['item_id'].astype('uint16')
    data['rating'] = data['rating'].astype('uint16')
    data['time_diff'] = data['time_diff'].astype('uint64')
    return data

def type_preparation_items_data(data):
    data['song_id'] = data['song_id'].astype('uint16')
    data['product_id'] = data['product_id'].astype('uint32')
    data['average_rating'] = data['average_rating'].astype('float32')
    data['mean'] = data['mean'].astype('float32')
    data['median'] = data['median'].astype('float32')
    data['std'] = data['std'].astype('float32')
    data['min'] = data['min'].astype('float32')
    data['max'] = data['max'].astype('float32')
    data['unique_users'] = data['unique_users'].astype('uint16')
    return data

def preprocess_data(smm_data, zvuk_data):
    max_timestamp_zvuk = zvuk_data['timestamp'].max()
    max_timestamp_smm = smm_data['timestamp'].max()
    zvuk_data['time_diff'] = max_timestamp_zvuk - zvuk_data['timestamp']
    smm_data['time_diff'] = max_timestamp_smm - smm_data['timestamp']
    del max_timestamp_zvuk, max_timestamp_smm

    smm_data = type_preparation_user_data(smm_data)
    zvuk_data = type_preparation_user_data(zvuk_data)

    return smm_data, zvuk_data
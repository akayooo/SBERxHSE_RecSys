import pandas as pd

# Функция для подготовки данных пользователей
def type_preparation_user_data(df):
    df['user_id'] = df['user_id'].astype(int)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Функция для подготовки данных о товарах
def type_preparation_items_data(df):
    df['product_id'] = df['product_id'].astype(int)
    return df

# Функция для расчёта статистик
def calculate_stats(series):
    return {
        'mean': series.mean(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max()
    }

# Функция для ограничения групп
def limit_group_size(group, max_size=10000):
    if len(group) > max_size:
        return group.sample(n=max_size, random_state=42)
    return group

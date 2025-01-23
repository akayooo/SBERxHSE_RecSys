import psutil
import numpy as np
import pandas as pd

def memory_usage():   
    process = psutil.Process()
    memory_info = process.memory_info().rss
    memory_usage_mb = memory_info / (1024 * 1024 * 1024)
    print(f"Использование оперативной памяти: {memory_usage_mb:.2f} ГБ")

def calculate_stats(group):
    return pd.Series({
        'mean': group.mean(),
        'median': group.median(),
        'std': group.std(),
        'min': group.min(),
        'max': group.max()
    })

def limit_group_size(group, max_size=10000):
    if len(group) > max_size:
        return group.sample(n=max_size, random_state=42)
    return group

def ndcg_at_k(y_true, y_score, k=10):
    y_true = y_true.reshape(-1)
    y_score = y_score.reshape(-1)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gain = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts) / np.sum(2 ** np.sort(y_true)[::-1] / discounts)
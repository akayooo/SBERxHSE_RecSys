import pandas as pd
import kagglehub

def load_data():
    
    smm_data = pd.read_parquet('train_smm.parquet')
    zvuk_data = pd.read_parquet('train_zvuk.parquet')

    # Ограничение данных для тестирования
    smm_data = smm_data.iloc[0:10**6]
    zvuk_data = zvuk_data.iloc[0:10**6]

    return smm_data, zvuk_data
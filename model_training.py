import lightgbm as lgb
from sklearn.model_selection import train_test_split

def train_model(train_data, features, target, params):
    train_dataset = lgb.Dataset(train_data[features], label=train_data[target])
    model = lgb.train(params, train_dataset, num_boost_round=100)
    return model
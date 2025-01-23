from utils import ndcg_at_k

def evaluate_model(model, test_data, features, target):
    predictions = model.predict(test_data[features])
    ndcg_at_10 = ndcg_at_k(test_data[target].values, predictions, k=10)
    print(f"NDCG@10: {ndcg_at_10}")
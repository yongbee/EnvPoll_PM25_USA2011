import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from model.utils import cluster_train_test_index, data_drop_na, normalize_train_test

class TrainTest:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def define_model(self):
        if self.model_name == 'RF':
            model = RandomForestRegressor(max_features=12, n_estimators=10, random_state=1000)
        return model

    def train(self, train_dataset: dict):
        self.all_models = {}
        for cluster_id in train_dataset.keys():
            print(f"Cluster{cluster_id} Train")
            input_dt = train_dataset[cluster_id]["input"]
            label_dt = train_dataset[cluster_id]["label"]
            model = self.define_model()
            model.fit(input_dt, label_dt)
            self.all_models[cluster_id] = model

    def pred(self, pred_dataset: dict):
        all_pred_vals = {}
        for cluster_id in pred_dataset.keys():
            input_dt = pred_dataset[cluster_id]["input"]
            pred_val = self.all_models[cluster_id].predict(input_dt)
            all_pred_vals[f"cluster{cluster_id}"] = pred_val
        return all_pred_vals

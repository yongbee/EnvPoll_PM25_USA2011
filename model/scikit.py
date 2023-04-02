import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def _compute_accuracy(cluster_data_pair, cluster_all_pred):
    all_label, all_pred = [], []
    for cluster in cluster_data_pair.keys():
        cluster_label = cluster_data_pair[cluster]["label"]
        clsuter_pred = cluster_all_pred[f"cluster{cluster}"]
        all_label.append(cluster_label)
        all_pred.append(clsuter_pred)
    all_label = np.hstack(all_label)
    all_pred = np.hstack(all_pred)
    mae = mean_absolute_error(all_label, all_pred)
    r2 = r2_score(all_label, all_pred)
    return mae, r2

class TrainTest:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.hyperparameter = None

    def define_model(self):
        if self.model_name == 'RF':
            if self.hyperparameter is None:
                model = RandomForestRegressor(max_features=12, n_estimators=1000, random_state=1000)
            else:
                model = RandomForestRegressor(random_state=1000, **self.hyperparameter)
        elif self.model_name == "GBM":
            if self.hyperparameter is None:
                model = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=6, random_state=1000)
            else:
                model = GradientBoostingRegressor(random_state=1000, **self.hyperparameter)
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

    def predict(self, pred_dataset: dict):
        all_pred_vals, all_labels = {}, {}
        for cluster_id in pred_dataset.keys():
            input_dt = pred_dataset[cluster_id]["input"]
            label_dt = pred_dataset[cluster_id]["label"]
            pred_val = self.all_models[cluster_id].predict(input_dt)
            all_pred_vals[f"cluster{cluster_id}"] = pred_val
            all_labels[f"cluster{cluster_id}"] = np.array(label_dt)
            mse_val = mean_squared_error(np.array(label_dt), pred_val)
            print(f"MSE value: {mse_val}")
        return all_pred_vals, all_labels

class HyperparameterTest(TrainTest):
    def __init__(self, model_name: str, parameters):
        super().__init__(model_name)
        self.parameters = parameters

    def hyperparameter_test(self, train_dataset, pred_dataset, test_name):
        keys, values = zip(*self.parameters.items())
        hyperparameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        all_accuracy = pd.DataFrame(columns=keys + ("MAE", "R2"), index=range(len(hyperparameter_combinations)))
        for comb_id, comb in enumerate(hyperparameter_combinations):
            self.hyperparameter = comb
            self.train(train_dataset)
            comb_preds = self.predict(pred_dataset)
            mae, r2 = _compute_accuracy(pred_dataset, comb_preds)
            all_accuracy.loc[comb_id] = list(comb.values()) + [mae,r2]
        all_accuracy.to_csv(f"result/{self.model_name}_{test_name}_hyperparameters_accuracy.csv", index=False)

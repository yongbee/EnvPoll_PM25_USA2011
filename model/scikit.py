import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from model.utils import cluster_train_test_index, data_drop_na, normalize_train_test

class TrainTest:
    def __init__(self, model_name: str, normalize=False):
        if model_name == 'RF':
            self.model = RandomForestRegressor(max_features=12, n_estimators=10, random_state=10)
        self.model_name = model_name
        self.normalize = normalize

    def train_predict(self, input_dt: pd.DataFrame, label_dt: pd.Series, train_test_data_id: dict):
        all_pred_info = {}
        for cluster_id in train_test_data_id.keys():
            print(f"Cluster{cluster_id} Train and Test")
            set_index = train_test_data_id[cluster_id]
            train_index, test_index = cluster_train_test_index(set_index)
            train_dt, train_label = input_dt.loc[train_index], label_dt[train_index]
            test_dt, test_label = input_dt.loc[test_index], label_dt[test_index]
            train_dt, test_dt = data_drop_na(train_dt, test_dt)
            if self.normalize:
                train_dt, test_dt = normalize_train_test(train_dt, test_dt)
            self.model.fit(train_dt, train_label)
            test_pred = self.model.predict(test_dt)
            all_pred_info[f"cluster{cluster_id}"] = test_pred
            r2_val = round(r2_score(test_label, test_pred), 4)
            print(f"R-squared: {r2_val}")
        return all_pred_info

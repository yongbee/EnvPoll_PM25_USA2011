import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def _cluster_train_test_index(set_index: pd.DataFrame):
    train_in = set_index['train_in_cluster']
    train_out = set_index['train_out_cluster']
    train_index = train_in.union(train_out)
    test_index = set_index['test_cluster']
    return train_index, test_index

def _data_drop_na(train_dt: pd.DataFrame, test_dt: pd.DataFrame):
    train_drop_dt = train_dt.dropna(axis=1)
    test_drop_dt = test_dt[train_drop_dt.columns]
    test_drop_dt = test_drop_dt.dropna(axis=0)
    return train_drop_dt, test_drop_dt

def _normalize_train_test(train_dt: pd.DataFrame, test_dt: pd.DataFrame):
    _std = train_dt.std(axis=0)
    train_dt = train_dt.loc[:,_std>0]
    test_dt = test_dt.loc[:, _std>0]
    mean, std = train_dt.mean(axis=0), train_dt.std(axis=0)
    train_dt = (train_dt - mean) / std
    test_dt = (test_dt - mean) / std
    return train_dt, test_dt

class TrainTest:
    def __init__(self, model_name: str, normalize=False):
        if model_name == 'RF':
            self.model = RandomForestRegressor(max_features=12, n_estimators=1000, random_state=10)
        self.model_name = model_name
        self.normalize = normalize

    def train_predict(self, input_dt: pd.DataFrame, label_dt: pd.Series, train_test_data_id: dict):
        all_pred_info = {}
        for cluster_id in train_test_data_id.keys():
            print(f"Cluster{cluster_id} Train and Test")
            set_index = train_test_data_id[cluster_id]
            train_index, test_index = _cluster_train_test_index(set_index)
            train_dt, train_label = input_dt.loc[train_index], label_dt[train_index]
            test_dt, test_label = input_dt.loc[test_index], label_dt[test_index]
            train_dt, test_dt = _data_drop_na(train_dt, test_dt)
            if self.normalize:
                train_dt, test_dt = _normalize_train_test(train_dt, test_dt)
            self.model.fit(train_dt, train_label)
            test_pred = self.model.predict(test_dt)
            all_pred_info[f"cluster{cluster_id}"] = test_pred
            r2_val = round(r2_score(test_label, test_pred), 4)
            print(f"R-squared: {r2_val}")
        return all_pred_info

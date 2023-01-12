import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def _cluster_train_test_index(set_index):
    train_in = set_index['train_in_cluster']
    train_out = set_index['train_out_cluster']
    train_index = pd.concat([train_in, train_out])
    test_index = set_index['test_cluster']
    return train_index, test_index

class TrainTest:
    def __init__(self, model_name:str, normalize=False):
        if model_name == 'RF':
            self.model = RandomForestRegressor(max_features=12, n_estimators=1000, random_state=0)
        self.normalize = normalize

    def train_model(self, input_dt:pd.DataFrame, label_dt:pd.Series, train_test_data_id:dict):
        all_models = {}
        if self.normalize:
            self.all_statistics = {}
        for cluster_id in train_test_data_id.keys():
            set_index = train_test_data_id[cluster_id]
            train_index, test_index = _cluster_train_test_index(set_index)
            train_dt, train_label = input_dt.loc[train_index], label_dt[train_index]
            test_dt, test_label = input_dt.loc[test_index], label_dt[test_index]
            

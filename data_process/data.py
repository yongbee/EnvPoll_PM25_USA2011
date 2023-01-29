import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def _convert_loader(input_dt:np.ndarray, output_dt:np.ndarray, batch:int):
    if len(input_dt) < 1:
        raise Exception("input_dt length is 0.")
    if len(output_dt) < 1:
        raise Exception("output_dt length is 0.")
    dt_set = InputOutputSet(input_dt, output_dt)
    dt_loader = DataLoader(dt_set, batch_size=batch, shuffle=False, pin_memory=True)
    return dt_loader

def cluster_train_valid_index(set_index: pd.DataFrame):
    train_in = set_index['train_in_cluster']
    train_out = set_index['train_out_cluster']
    train_index = train_in.union(train_out)
    valid_index = set_index['valid_cluster']
    return train_index, valid_index

def _drop_constant_col(train_dt: pd.DataFrame, valid_dt: pd.DataFrame):
    _std = train_dt.std(axis=0)
    train_dt_variable = train_dt.loc[:,_std>0]
    valid_dt_variable = valid_dt.loc[:, _std>0]
    return train_dt_variable, valid_dt_variable

def _drop_na_col(train_dt: pd.DataFrame, valid_dt: pd.DataFrame):
    train_drop_dt = train_dt.dropna(axis=1)
    valid_drop_dt = valid_dt[train_drop_dt.columns]
    return train_drop_dt, valid_drop_dt

def _drop_useless_col(train_data, valid_data):
    train_drop_na, valid_drop_na = _drop_na_col(train_data, valid_data)
    train_drop_const, valid_drop_const = _drop_constant_col(train_drop_na, valid_drop_na)
    return train_drop_const, valid_drop_const

class InputOutputSet(Dataset):
    def __init__(self, input_dt, output_dt):
        super().__init__()
        self.input_dt = input_dt
        self.output_dt = output_dt

    def __getitem__(self, i):
        return self.input_dt[i], self.output_dt[i]

    def __len__(self):
        return len(self.input_dt)

class SingleData:
    def __init__(self, input_dt: pd.DataFrame, label_dt: pd.Series, train_valid_data_id: dict, normalize=False):
        self.input_dt = input_dt
        self.label_dt = label_dt
        self.train_valid_data_id = train_valid_data_id
        self.split_train_valid()
        if normalize:
            self._normalize_train_valid()

    def split_train_valid(self):
        self.train_dt, self.valid_dt = {}
        for cluster_id in self.train_valid_data_id.keys():
            set_index = self.train_valid_data_id[cluster_id]
            train_index, valid_index = cluster_train_valid_index(set_index)
            train_input, train_label = self.input_dt.loc[train_index], self.label_dt[train_index]
            valid_input, valid_label = self.input_dt.loc[valid_index], self.label_dt[valid_index]
            train_input, valid_input = _drop_useless_col(train_input, valid_input)
            self.train_dt[cluster_id] = {"input":train_input, "label":train_label}
            self.valid_dt[cluster_id] = {"input":valid_input, "label":valid_label}
        
    def _normalize_train_valid(self):
        for cluster_id in self.train_dt.keys():
            train_input = self.train_dt[cluster_id]["input"]
            valid_input = self.valid_dt[cluster_id]["input"]
            mean, std = train_input.mean(axis=0), train_input.std(axis=0)
            self.train_dt[cluster_id]["input"] = (train_input - mean) / std
            self.valid_dt[cluster_id]["input"] = (valid_input - mean) / std

    def data_convert_loader(self):
        for cluster_id in self.train_dt.keys():
            train_input = self.train_dt[cluster_id]["input"]
            train_label = self.train_dt[cluster_id]["label"]
            valid_input = self.valid_dt[cluster_id]["input"]
            valid_label = self.valid_dt[cluster_id]["label"]
            train_loader = _convert_loader(train_input, train_label, 128)
            valid_loader = _convert_loader(valid_input, valid_label, 128)
            self.train_dt[cluster_id] = train_loader
            self.valid_dt[cluster_id] = valid_loader

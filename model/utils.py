import pandas as pd

def cluster_train_test_index(set_index: pd.DataFrame):
    train_in = set_index['train_in_cluster']
    train_out = set_index['train_out_cluster']
    train_index = train_in.union(train_out)
    test_index = set_index['test_cluster']
    return train_index, test_index

def data_drop_na(train_dt: pd.DataFrame, test_dt: pd.DataFrame):
    train_drop_dt = train_dt.dropna(axis=1)
    test_drop_dt = test_dt[train_drop_dt.columns]
    test_drop_dt = test_drop_dt.dropna(axis=0)
    return train_drop_dt, test_drop_dt

def normalize_train_test(train_dt: pd.DataFrame, test_dt: pd.DataFrame):
    _std = train_dt.std(axis=0)
    train_dt = train_dt.loc[:,_std>0]
    test_dt = test_dt.loc[:, _std>0]
    mean, std = train_dt.mean(axis=0), train_dt.std(axis=0)
    train_dt = (train_dt - mean) / std
    test_dt = (test_dt - mean) / std
    return train_dt, test_dt

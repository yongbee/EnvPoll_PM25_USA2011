import numpy as np
import pandas as pd
from data_process.spatial_validation import SingleGrid
from data_process.data import SingleData
from model.scikit import TrainTest

def _get_clusters(input_dt: pd.DataFrame, label_dt: pd.Series):
    single_grid = SingleGrid("KMeans")
    whole_cluster, _ = single_grid.cluster_grids(input_dt, pd.Series(label_dt))
    _, train_test_data_id = single_grid.split_train_test(input_dt, whole_cluster)
    return train_test_data_id

def _save_results(all_pred: dict, model_name: str):
    save_dir = f"result/{model_name}.npz"
    np.savez(save_dir, **all_pred)

if __name__=='__main__':
    model_name = "RF"
    monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")

    input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
    label_dt = monitoring_whole_data["pm25_value"]
    train_test_data_id = _get_clusters(input_dt, label_dt)
    single_data = SingleData(input_dt, label_dt, train_test_data_id, True)
    model_train_test = TrainTest(model_name)
    model_train_test.train(single_data.train_dt)
    all_pred = model_train_test.train_predict(input_dt, label_dt, train_test_data_id)
    _save_results(all_pred, model_name)

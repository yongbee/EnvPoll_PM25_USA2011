import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from data_process.spatial_validation import get_clusters, get_in_clusters
from data_process.data import SingleData, tag_names
from model.scikit import TrainTest

def _save_multi_results(all_pred: dict, model_name: str):
    save_dir = f"result/{model_name}.npz"
    np.savez(save_dir, **all_pred)

def _save_split_results(all_pred: dict, model_name: str, save_name: str):
    save_dir = f"result/target cluster split/{model_name}_{save_name}.npz"
    np.savez(save_dir, **all_pred)

def _save_accuracy(all_label, all_pred, model_name, train_num):
    save_dir = f"result/target cluster split/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = f"{model_name}_with_wa_mean_accuracy.csv"
    file_full_path = save_dir + file_name
    all_r2, all_mse = [], []
    for cluster_id in all_label.keys():
        clsuter_label = all_label[cluster_id]
        cluster_pred = all_pred[cluster_id]
        r2 = r2_score(clsuter_label, cluster_pred)
        mse = mean_absolute_error(clsuter_label, cluster_pred)
        all_r2.append(r2)
        all_mse.append(mse)
    mean_r2, mean_mse = np.mean(all_r2), np.mean(all_mse)
    if os.path.exists(file_full_path):
        file_dt = pd.read_csv(file_full_path, index_col=0)
    else:
        file_dt = pd.DataFrame(columns=["Mean R2", "Mean MSE"])
    file_dt.loc[train_num] = [mean_r2, mean_mse]
    file_dt.to_csv(file_full_path)

def _multiple_cluster_model(model_name, input_dt, label_dt, weight_average):
    train_test_data_id, _ = get_clusters(input_dt, label_dt)
    single_data = SingleData(input_dt, label_dt, train_test_data_id, "index", True, weight_average)
    model_train_test = TrainTest(model_name)
    model_train_test.train(single_data.train_dt)
    all_pred = model_train_test.predict(single_data.valid_dt)
    _save_multi_results(all_pred, model_name)

def _one_cluster_model(model_name, train_num, input_dt, label_dt, weight_average, save_preds=False):
    data_path = "data/split-data/single/"
    save_name = f"cmaq_id-{train_num}"
    train_test_data_id = get_in_clusters(data_path, train_num)
    single_data = SingleData(input_dt, label_dt, train_test_data_id, "cmaq_id", True, weight_average)
    model_train_test = TrainTest(model_name)
    model_train_test.train(single_data.train_dt)
    all_pred, all_label = model_train_test.predict(single_data.valid_dt)
    if save_preds:
        _save_split_results(all_pred, model_name, save_name)
    else:
        _save_accuracy(all_label, all_pred, model_name, train_num)

if __name__=='__main__':
    model_name = "GBM"
    weight_average = True
    train_numbers = [5, 10, 15, 20, 30, 40, 50]

    monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")[tag_names]
    input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
    label_dt = monitoring_whole_data["pm25_value"]

    for train_num in train_numbers:
        _one_cluster_model(model_name, train_num, input_dt, label_dt, weight_average)

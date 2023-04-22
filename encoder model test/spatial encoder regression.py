import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from data_process.spatial_validation import get_in_clusters
from data_process.data import NearStationData, tag_names
from model.autoencoder import AutoencoderTrainTest
from model.scikit import TrainTest

def _save_split_results(all_pred: dict, model_name: str, save_name: str):
    save_dir = f"result/target cluster split/{model_name}_{save_name}.npz"
    np.savez(save_dir, **all_pred)

def _save_accuracy(all_label, all_pred, model_name, train_num):
    save_dir = f"result/target cluster split/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = f"{model_name}_encode_mean_accuracy.csv"
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

def _one_cluster_model(model_name, train_num, input_dt, label_dt, save_preds=False):
    cluster_id = 0
    compose_data = False
    data_path = "data/split-data/single/"

    compose_path = f"{data_path}tl-cal-{train_num}/split-{cluster_id}/"
    save_name = f"cmaq_id-{train_num}"
    train_test_data_id = get_in_clusters(data_path, train_num)
    near_station_data = NearStationData(input_dt, label_dt, train_test_data_id, cluster_id, compose_path, compose_data, ["cmaq_id"], True)
    autoencoder_train_test = AutoencoderTrainTest("CNN", 4, near_station_data.input_shape)
    autoencode_train_dt, autoencode_valid_dt = near_station_data.autoencode_data_convert_loader()
    autoencoder_train_test.train(autoencode_train_dt, 20)
    train_encode = autoencoder_train_test.encode(autoencode_train_dt)
    valid_encode = autoencoder_train_test.encode(autoencode_valid_dt)
    train_dt = {c:{"input":train_encode[c], "label":near_station_data.train_dt[cluster_id]["label"]} for c in train_encode.keys()}
    valid_dt = {c:{"input":valid_encode[c], "label":near_station_data.valid_dt[cluster_id]["label"]} for c in valid_encode.keys()}
    model_train_test = TrainTest(model_name)
    model_train_test.train(train_dt)
    all_pred, all_label = model_train_test.predict(valid_dt)
    if save_preds:
        _save_split_results(all_pred, model_name, save_name)
    else:
        _save_accuracy(all_label, all_pred, model_name, train_num)

if __name__=='__main__':
    model_name = "GBM"
    weight_average = False
    train_numbers = [5, 10, 15, 20, 30, 40, 50]

    monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")[tag_names]
    input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
    label_dt = monitoring_whole_data["pm25_value"]

    for train_num in train_numbers:
        _one_cluster_model(model_name, train_num, input_dt, label_dt)

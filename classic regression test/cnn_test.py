import numpy as np
import pandas as pd
from data_process.spatial_validation import get_clusters, get_in_clusters
from data_process.data import MultipleData, tag_names
from model.neuralnet import TrainTest

def _save_multi_results(all_pred: dict, model_name: str):
    save_dir = f"result/{model_name}.npz"
    np.savez(save_dir, **all_pred)

def _save_split_results(all_pred: dict, model_name: str, save_name: str):
    save_dir = f"result/target cluster split/{model_name}_{save_name}.npz"
    np.savez(save_dir, **all_pred)

if __name__=='__main__':
    model_name = "CNN"
    id_type, save_type = "cmaq_id", "cluster4"
    # id_type, save_type = "index", "multiple_cluster"
    if save_type == "cluster4":
        data_path, train_num = "data/split-data/multiple/", 10
        save_name = f"{save_type}-{train_num}"
    elif save_type == "multiple_cluster":
        save_name = save_type

    monitoring_whole_data = np.load("data/us_monitoring.csv")
    input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
    label_dt = monitoring_whole_data["pm25_value"]

    if id_type == "index":
        train_test_data_id, _ = get_clusters(input_dt, label_dt)
    elif id_type == "cmaq_id":
        train_test_data_id = get_in_clusters(data_path, train_num)

    multiple_data = MultipleData(input_dt, label_dt, train_test_data_id, id_type, True)
    multiple_data.data_convert_loader()
    model_train_test = TrainTest(model_name, multiple_data.input_dim)
    model_train_test.train(multiple_data.train_dt, 25)
    all_pred = model_train_test.predict(multiple_data.valid_dt)

    if id_type == "index":
        _save_multi_results(all_pred, model_name)
    elif id_type == "cmaq_id":
        _save_split_results(all_pred, model_name, save_name)

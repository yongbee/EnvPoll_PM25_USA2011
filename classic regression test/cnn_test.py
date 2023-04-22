import numpy as np
import pandas as pd
from data_process.spatial_validation import get_clusters, get_in_clusters
from data_process.data import MultipleData, tag_names
from model.nn_regressor import RegressionTrainTest

def _extract_center_target(target_data: np.ndarray):
    height, width = target_data.shape[1], target_data.shape[2]
    return target_data[:, height//2, width//2]

def _save_multi_results(all_pred: dict, model_name: str):
    save_dir = f"result/{model_name}.npz"
    np.savez(save_dir, **all_pred)

def _save_split_results(all_pred: dict, model_name: str, save_name: str):
    save_dir = f"result/target cluster split/{model_name}_{save_name}.npz"
    np.savez(save_dir, **all_pred)

if __name__=='__main__':
    model_name = "CNN"
    id_type, save_type, grid_width = "cmaq_id", "cluster4", 5
    # id_type, save_type = "index", "multiple_cluster"
    if save_type == "cluster4":
        data_path, train_num = "data/split-data/multiple/", 10
        save_name = f"{save_type}-{train_num}"
    elif save_type == "multiple_cluster":
        save_name = save_type

    input_dt = np.load("data/grid5_dataset_without_WA.npy")
    label_dt = pd.Series(_extract_center_target(np.load("data/grid5_target.npy")))

    if id_type == "index":
        train_test_data_id, _ = get_clusters(input_dt, label_dt)
    elif id_type == "cmaq_id":
        train_test_data_id = get_in_clusters(data_path, train_num)

    multiple_data = MultipleData(input_dt, label_dt, tag_names[:-1], grid_width, train_test_data_id, id_type, True)
    multiple_data.data_convert_loader()
    model_train_test = RegressionTrainTest(model_name, multiple_data.input_dim)
    model_train_test.train(multiple_data.train_dt, 25)
    all_pred = model_train_test.predict(multiple_data.valid_dt)

    if id_type == "index":
        _save_multi_results(all_pred, model_name)
    elif id_type == "cmaq_id":
        _save_split_results(all_pred, model_name, save_name)

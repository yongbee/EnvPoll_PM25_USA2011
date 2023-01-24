import numpy as np
import pandas as pd
from data_process.spatial_validation import SingleGrid, extract_center_data
from model.scikit import TrainTest

def _create_tags(tag_num: int):
    first_tags = ['cmaq_x', 'cmaq_y']
    rest_tags = [f'tag{x}' for x in range(tag_num-2)]
    return first_tags + rest_tags

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
    data_path = 'v10_170713_5x5_include_na_dataset.npz'
    label_path = "v10_170713_5x5_include_na_label.npz"
    x_tr_blended = np.load(data_path)['arr_0']
    y_tr_blended = np.load(label_path)['arr_0']
    tag_names = _create_tags(28)

    center_input_dt = extract_center_data(tag_names, x_tr_blended, 5)
    center_frame = pd.DataFrame(center_input_dt, columns=tag_names, index=np.arange(len(y_tr_blended)))
    train_test_data_id = _get_clusters(center_frame, y_tr_blended)
    model_train_test = TrainTest(model_name, True)
    all_pred = model_train_test.train_predict(center_frame, y_tr_blended, train_test_data_id)
    _save_results(all_pred, model_name)

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from data_process.spatial_validation import SingleGrid, extract_center_data

def _create_tags(tag_num):
    first_tags = ['cmaq_x', 'cmaq_y']
    rest_tags = [f'tag{x}' for x in range(tag_num-2)]
    return first_tags + rest_tags

def _get_clusters(input_dt: pd.DataFrame, label_dt: pd.Series):
    single_grid = SingleGrid("KMeans")
    whole_cluster, _ = single_grid.cluster_grids(input_dt, pd.Series(label_dt))
    _, train_test_data_id = single_grid.split_train_test(input_dt, whole_cluster)
    return train_test_data_id

def _get_labels(label_dt: pd.DataFrame, train_test_data_id: dict):
    all_labels = []
    for cluster_id in train_test_data_id.keys():
        cluster_test_index = train_test_data_id[cluster_id]['test_cluster']
        all_labels.append(label_dt[cluster_test_index])
    all_labels = np.hstack(all_labels)
    return all_labels

def _get_results(model_name: str):
    result_data = np.load(f"result/{model_name}.npz")
    all_pred_data = []
    for cluster_id in range(10):
        all_pred_data.append(result_data[f"cluster{cluster_id}"])
    all_pred_data = np.hstack(all_pred_data)
    return all_pred_data

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
    all_label = _get_labels(y_tr_blended, train_test_data_id)
    all_pred = _get_results(model_name)
    r2_val = round(r2_score(all_label, all_pred), 4)
    print(r2_val)

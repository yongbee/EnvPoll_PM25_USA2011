import numpy as np
import pandas as pd
from data_process.spatial_validation import SingleGrid
from model.scikit import TrainTest

def _create_tags(tag_num):
    first_tags = ['cmaq_x', 'cmaq_y']
    rest_tags = [f'tag{x}' for x in range(tag_num-2)]
    return first_tags + rest_tags

def _extract_center_data(tag_names: list, input_dt: np.ndarray, target_dt:pd.Series, grid_scale:int):
    if type(input_dt) is not np.ndarray:
        raise Exception("Input data type is not np.array.")
    if input_dt.shape[1] != (len(tag_names)*(grid_scale**2)):
        raise Exception("Input data column number is inappropriate.")
        
    center_cell_id = (grid_scale**2)//2
    center_dt = input_dt[:,center_cell_id*len(tag_names):(center_cell_id+1)*len(tag_names)]
    center_frame = pd.DataFrame(center_dt, columns=tag_names, index=target_dt.index)
    return center_frame

def _get_clusters(input_dt: pd.DataFrame, label_dt: pd.Series):
    single_grid = SingleGrid("KMeans")
    whole_cluster, _ = single_grid.cluster_grids(input_dt, pd.Series(label_dt))
    _, train_test_data_id = single_grid.split_train_test(input_dt, whole_cluster)
    return train_test_data_id

if __name__=='__main__':
    data_path = 'v10_170713_5x5_include_na_dataset.npz'
    label_path = "v10_170713_5x5_include_na_label.npz"
    x_tr_blended = np.load(data_path)['arr_0']
    y_tr_blended = np.load(label_path)['arr_0']
    tag_names = _create_tags(28)
    center_input_dt = _extract_center_data(tag_names, x_tr_blended, pd.Series(y_tr_blended), 5)
    train_test_data_id = _get_clusters(center_input_dt, y_tr_blended)
    model_train_test = TrainTest("RF", True)
    all_pred_info = model_train_test.train_predict(center_input_dt, y_tr_blended, train_test_data_id)
    a=3

import numpy as np
import pandas as pd
from data_process.spatial_validation import MultipleGrid

def _create_tags(tag_num):
    first_tags = ['x', 'y']
    rest_tags = [f'tag{x}' for x in range(tag_num-2)]
    return first_tags + rest_tags

def _get_clusters(input_dt, label_dt):
    tag_names = _create_tags(28)
    multi_grid = MultipleGrid(5, "KMeans")
    whole_cluster, _ = multi_grid.cluster_grids(tag_names, input_dt, pd.Series(label_dt))
    _, train_test_data_id = multi_grid.split_train_test(tag_names, input_dt, whole_cluster)
    return train_test_data_id

if __name__=='__main__':
    data_path = 'v10_170713_5x5_include_na_dataset.npz'
    label_path = "v10_170713_5x5_include_na_label.npz"
    x_tr_blended = np.load(data_path)['arr_0']
    y_tr_blended = np.load(label_path)['arr_0']
    train_test_data_id = _get_clusters(x_tr_blended, y_tr_blended)


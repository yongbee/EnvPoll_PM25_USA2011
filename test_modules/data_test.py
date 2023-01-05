import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data.spatial import MultipleGrid

def _create_tags(tag_num):
    first_tags = ['x', 'y']
    rest_tags = [f'tag{x}' for x in range(tag_num-2)]
    return first_tags + rest_tags

if __name__=='__main__':
    data_path = 'v10_170713_5x5_include_na_dataset.npz'
    label_path = "v10_170713_5x5_include_na_label.npz"
    x_tr_blended = np.load(data_path)['arr_0']
    y_tr_blended = np.load(label_path)['arr_0']
    tag_names = _create_tags(28)
    multi_grid = MultipleGrid(5, "KMeans")
    whole_cluster, coor_cluster = multi_grid.split_train_test(tag_names, x_tr_blended, pd.Series(y_tr_blended))
    plt.scatter(coor_cluster['x'], coor_cluster['y'], 3, coor_cluster['cluster id'])
    plt.show()

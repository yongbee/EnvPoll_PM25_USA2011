import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_process.spatial_validation import MultipleGrid

def _create_tags(tag_num):
    first_tags = ['cmaq_x', 'cmaq_y']
    rest_tags = [f'tag{x}' for x in range(tag_num-2)]
    return first_tags + rest_tags

def _plot_grid_clusters(coor_cluster):
    plt.scatter(coor_cluster['cmaq_x'], coor_cluster['cmaq_y'], 3, coor_cluster['cluster id'])
    plt.savefig('figures/coordinate cluster')
    plt.cla()
    plt.clf()

def _plot_train_test(cluster_train_test, cluster_id):
    set_dt = cluster_train_test[cluster_id]
    train_in = set_dt['train_in_cluster']
    train_out = set_dt['train_out_cluster']
    test_dt = set_dt['test_cluster']
    plt.scatter(train_out['cmaq_x'], train_out['cmaq_y'], 3, 'b', label='out-of-cluster train data')
    plt.scatter(test_dt['cmaq_x'], test_dt['cmaq_y'], 3, 'r', label='test data')
    plt.scatter(train_in['cmaq_x'], train_in['cmaq_y'], 5, 'g', label='in-cluster train data')
    plt.legend()
    plt.savefig(f'figures/cluster{cluster_id} train-test set')
    plt.cla()
    plt.clf()

if __name__=='__main__':
    data_path = 'v10_170713_5x5_include_na_dataset.npz'
    label_path = "v10_170713_5x5_include_na_label.npz"
    x_tr_blended = np.load(data_path)['arr_0']
    y_tr_blended = np.load(label_path)['arr_0']
    tag_names = _create_tags(28)
    
    multi_grid = MultipleGrid(5, "KMeans")
    whole_cluster, coor_cluster = multi_grid.cluster_grids(tag_names, x_tr_blended, pd.Series(y_tr_blended))
    _plot_grid_clusters(coor_cluster)
    train_test_grids, train_test_data_id = multi_grid.split_train_test(tag_names, x_tr_blended, whole_cluster)
    for i in train_test_grids.keys():
        _plot_train_test(train_test_grids, i)

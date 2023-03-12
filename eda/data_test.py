import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_process.spatial_validation import SingleGrid

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
    tag_names = ['day', 'month', 'cmaq_x', 'cmaq_y', 'cmaq_id', 'rid', 'elev', 'forest_cover', 'pd', 'local', 'limi', 'high', 'is', 
    'nldas_pevapsfc','nldas_pressfc', 'nldas_cape', 'nldas_ugrd10m', 'nldas_vgrd10m', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_dlwrfsfc', 
    'nldas_dswrfsfc', 'nldas_pcpsfc', 'nldas_fpcsfc', 'gc_aod', 'aod_value', 'emissi11_pm25', 'pm25_value_k', 'pm25_value']
    monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")

    input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
    label_dt = monitoring_whole_data["pm25_value"]
    
    single_grid = SingleGrid("KMeans")
    whole_cluster, coor_cluster = single_grid.cluster_grids(input_dt, label_dt)
    _plot_grid_clusters(coor_cluster)
    train_test_grids, train_test_data_id = single_grid.split_train_test(input_dt, whole_cluster)
    for i in train_test_grids.keys():
        _plot_train_test(train_test_grids, i)

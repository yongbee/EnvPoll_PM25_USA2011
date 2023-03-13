import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_process.data import tag_names
from data_process.spatial_validation import SingleGrid, TargetSingle

def _source_target_split(coord_cluster, cluster_id):
    target_coords = coord_cluster.loc[coord_cluster["cluster id"] == cluster_id]
    source_coords = coord_cluster.loc[coord_cluster["cluster id"] != cluster_id]
    return {"source": source_coords, "target": target_coords}

def _plot_target_data(source_coord, target_coord_set, train_num):
    save_dir = f"figures/target cluster split/tl-cal-{train_num}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for set_id in target_coord_set.keys():
        set_data = target_coord_set[set_id]
        train_coord = set_data["train"]
        test_coord = set_data["test"]
        fig_savedir = f"{save_dir}split-{set_id}.png"
        plt.scatter(source_coord["cmaq_x"], source_coord["cmaq_y"], s=3, color='blue')
        plt.scatter(train_coord["cmaq_x"], train_coord["cmaq_y"], s=3, color="green")
        plt.scatter(test_coord["cmaq_x"], test_coord["cmaq_y"], s=3, color='red')
        plt.savefig(fig_savedir)
        plt.cla()
        plt.clf()

def _save_target_cmaqs(target_coord_set, train_num):
    for set_id in target_coord_set.keys():
        set_data = target_coord_set[set_id]
        train_cmaq = np.array(set_data["train"]["cmaq_id"])
        test_cmaq = np.array(set_data["test"]["cmaq_id"])
        save_dir = f'data/split-data/tl-cal-{train_num}/split-{set_id}/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_file = f"{save_dir}source_cal_label.npz"
        np.savez(save_file, train=train_cmaq, test=test_cmaq)

if __name__=='__main__':
    cluster_id = 4
    mn_range_list = [5,6,7,8,9,10,11,12,13,14,15,20,30,40,50]
    monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")[tag_names]

    input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
    label_dt = monitoring_whole_data["pm25_value"]
    
    single_grid = SingleGrid("KMeans")
    _, coor_cluster = single_grid.cluster_grids(input_dt, label_dt)

    source_target_set = _source_target_split(coor_cluster, cluster_id)
    target_single = TargetSingle()
    
    for sp in mn_range_list:
        train_test_coord = target_single.split_data_coord(source_target_set["target"], sp)
        _plot_target_data(source_target_set["source"], train_test_coord, sp)
        _save_target_cmaqs(train_test_coord, sp)


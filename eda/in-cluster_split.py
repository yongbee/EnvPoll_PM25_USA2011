import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_process.data import tag_names
from data_process.spatial_validation import SingleGrid, MultipleGrid, TargetGrid

def _source_target_split(coord_cluster, cluster_id):
    target_coords = coord_cluster.loc[coord_cluster["cluster id"] == cluster_id]
    source_coords = coord_cluster.loc[coord_cluster["cluster id"] != cluster_id]
    return {"source": source_coords, "target": target_coords}

def _plot_target_data(source_coord, target_coord_set, grid_type, train_num):
    save_dir = f"figures/target cluster split/{grid_type}/tl-cal-{train_num}/"
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

def _save_source_cmaqs(source_coord, grid_type):
    save_dir = f'data/split-data/{grid_type}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = f"{save_dir}source_cmaq.npy"
    np.save(save_file, source_coord)

def _save_target_cmaqs(target_coord_set, grid_type, train_num):
    for set_id in target_coord_set.keys():
        set_data = target_coord_set[set_id]
        train_cmaq = np.array(set_data["train"]["cmaq_id"])
        test_cmaq = np.array(set_data["test"]["cmaq_id"])
        save_dir = f'data/split-data/{grid_type}/tl-cal-{train_num}/split-{set_id}/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_file = f"{save_dir}target_cmaq.npz"
        np.savez(save_file, train=train_cmaq, test=test_cmaq)

if __name__=='__main__':
    cluster_id = 4
    mn_range_list = [5,6,7,8,9,10,11,12,13,14,15,20,30,40,50]
    grid_type = "single"

    if grid_type == "single":
        monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")[tag_names]
        input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
        label_dt = monitoring_whole_data["pm25_value"]
        single_grid = SingleGrid("KMeans")
        _, coor_cluster = single_grid.cluster_grids(input_dt, label_dt)
    elif grid_type == "multiple":
        input_dt = np.load("data/grid5_dataset_without_WA.npy")
        input_dt = np.reshape(input_dt, (input_dt.shape[0], -1))
        label_dt = pd.Series(np.load("data/grid5_target.npy")[:,2,2])
        multiple_grid = MultipleGrid(5, "KMeans")
        _, coor_cluster = multiple_grid.cluster_grids(tag_names[:-1], input_dt, label_dt)
    target_grid = TargetGrid()
    source_target_set = _source_target_split(coor_cluster, cluster_id)
    _save_source_cmaqs(source_target_set["source"]["cmaq_id"], grid_type)
    for sp in mn_range_list:
        train_test_coord = target_grid.split_data_coord(source_target_set["target"], sp)
        _plot_target_data(source_target_set["source"], train_test_coord, grid_type, sp)
        _save_target_cmaqs(train_test_coord, grid_type, sp)

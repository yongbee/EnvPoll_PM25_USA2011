import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

def create_distance_matrix(dt1, dt2):
    all_distance = distance_matrix(dt1, dt2)
    all_distance[all_distance==0] = np.inf
    return all_distance

def get_distance_statistic(whole_xy):
    cut_len = 5000
    all_min_distance = []
    for set_id in range(len(whole_xy)//cut_len + 1):
        all_distance = create_distance_matrix(whole_xy.iloc[set_id*cut_len:(set_id+1)*cut_len].values, whole_xy.values)
        all_min_distance.append(all_distance.min(axis=1))
    all_min_distance = np.hstack(all_min_distance)
    return all_min_distance

def remove_distant_coords(whole_dt):
    coord_dt = whole_dt[["cmaq_x", "cmaq_y"]]
    all_min_distance = get_distance_statistic(coord_dt)
    outlier_threshold = np.percentile(all_min_distance, 99)
    ordinary_dt = whole_dt.loc[all_min_distance<=outlier_threshold]
    return ordinary_dt

if __name__ == '__main__':
    coord_data = pd.read_csv("data/largeUS_coords_pred.csv", index_col=0).drop_duplicates().reset_index(drop=True)
    monitoring_coord = pd.read_csv("data/us_monitoring.csv")[["cmaq_x", "cmaq_y", "cmaq_id"]].drop_duplicates().reset_index(drop=True)
    coord_data = remove_distant_coords(coord_data)
    monitoring_coord = monitoring_coord.loc[np.isin(monitoring_coord["cmaq_id"], coord_data["cmaq_id"])]

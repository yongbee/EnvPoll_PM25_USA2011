import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

def _create_distance_matrix(dt1: pd.DataFrame, dt2: pd.DataFrame):
    all_distance = distance_matrix(dt1, dt2)
    all_distance[all_distance==0] = np.inf
    return all_distance

def _get_distance_statistic(whole_xy: pd.DataFrame):
    cut_len = 5000
    all_min_distance = []
    for set_id in range(len(whole_xy)//cut_len + 1):
        all_distance = _create_distance_matrix(whole_xy.iloc[set_id*cut_len:(set_id+1)*cut_len].values, whole_xy.values)
        all_min_distance.append(all_distance.min(axis=1))
    all_min_distance = np.hstack(all_min_distance)
    return all_min_distance

def _compute_square_points(point_xy, longest_dist, grid_left, grid_right):
    square_top = point_xy['cmaq_y'] + grid_left*longest_dist
    square_bot = point_xy['cmaq_y'] - grid_right*longest_dist
    square_left = point_xy['cmaq_x'] - grid_left*longest_dist
    square_right = point_xy['cmaq_x'] + grid_right*longest_dist
    return {'top':square_top, 'bot':square_bot, 'left':square_left, 'right':square_right}

def _grid_dataset(whole_xy, point_xy, longest_dist, grid_left, grid_right):
    sqaure_points = _compute_square_points(point_xy, longest_dist, grid_left, grid_right)
    square_xy = whole_xy.loc[
        (whole_xy['cmaq_y']<=sqaure_points['top']) &
        (whole_xy['cmaq_y']>=sqaure_points['bot']) &
        (whole_xy['cmaq_x']<=sqaure_points['right']) &
        (whole_xy['cmaq_x']>=sqaure_points['left'])
    ]
    return square_xy

def _compose_grid_coord(whole_xy, point_xy, longest_dist, grid_left, grid_right):
    larger_grids = _grid_dataset(whole_xy, point_xy, longest_dist, grid_left+2, grid_right+2)
    sqaure_points = _compute_square_points(point_xy, longest_dist, grid_left, grid_right)
    vertical_points = np.linspace(sqaure_points['top'], sqaure_points['bot'], grid_left+grid_right+1)
    horizontal_points = np.linspace(sqaure_points['left'], sqaure_points['right'], grid_left+grid_right+1)
    all_grids =  np.transpose(np.dstack(np.meshgrid(vertical_points, horizontal_points)), (1,0,2))

    larger_grids_copy = larger_grids.copy()
    cmaq_grids = np.full(all_grids.shape[:2], np.nan)
    for v_id in range(all_grids.shape[0]):
        for h_id in range(all_grids.shape[1]):
            target_coor = all_grids[v_id, h_id]
            all_dist = distance_matrix([np.flip(target_coor)], larger_grids_copy[['cmaq_x', 'cmaq_y']].values)
            min_dist = np.min(all_dist)
            if min_dist <= longest_dist:
                grid_id = np.argmin(all_dist)
                cmaq_grids[v_id, h_id] = larger_grids_copy['cmaq_id'].iloc[grid_id]
                larger_grids_copy.drop(larger_grids_copy.index[grid_id], inplace=True)
    return cmaq_grids, larger_grids, all_grids

class MultiGridCompose:
    def __init__(self, whole_coord: pd.DataFrame, monitor_coord: pd.DataFrame):
        self.whole_coord = whole_coord
        self.monitor_coord = monitor_coord
        self.longest_dist = self._find_dist_threshold()

    def _find_dist_threshold(self):
        coord_dt = self.whole_coord[["cmaq_x", "cmaq_y"]]
        all_min_distance = _get_distance_statistic(coord_dt)
        longest_dist = np.percentile(all_min_distance, 99.9)
        return longest_dist

    def plot_map(self):
        plt.scatter(self.whole_coord['cmaq_x'], self.whole_coord['cmaq_y'], s=1, c='gray', alpha=0.3)
        plt.scatter(self.monitor_coord['cmaq_x'], self.monitor_coord['cmaq_y'], s=1, c='r')
        plt.show()
        
    def plot_grid_dataset(self, grid_left: int, grid_right: int):
        np.random.seed(1000)
        monitoring_xy_shuffle = self.monitor_coord.sample(frac=1)
        for i in range(10):
            point_xy = monitoring_xy_shuffle.iloc[i]
            square_xy = _grid_dataset(self.whole_coord, point_xy, self.longest_dist, grid_left, grid_right)
            plt.scatter(self.whole_coord['cmaq_x'], self.whole_coord['cmaq_y'], s=1, c='gray', alpha=0.3)
            plt.scatter(square_xy['cmaq_x'], square_xy['cmaq_y'], s=1, c='g')
            plt.scatter(point_xy['cmaq_x'], point_xy['cmaq_y'], s=1, c='r')
            plt.show()

    def plot_grid_allocation(self, grid_left: int, grid_right: int):
        np.random.seed(1000)
        monitoring_xy_shuffle = self.monitor_coord.sample(frac=1)
        for i in range(10):
            point_xy = monitoring_xy_shuffle.iloc[i]
            cmaq_grids, larger_grids, all_grids = _compose_grid_coord(self.whole_coord, point_xy, self.longest_dist, grid_left, grid_right)
            allocate_grid_id = cmaq_grids[~np.isnan(cmaq_grids)]
            allocate_grids = larger_grids.loc[np.isin(larger_grids['cmaq_id'], allocate_grid_id)]
            value_grids = np.flip(all_grids[~np.isnan(cmaq_grids),:])
            nan_grids = np.flip(all_grids[np.isnan(cmaq_grids),:])
            plt.scatter(self.whole_coord['cmaq_x'], self.whole_coord['cmaq_y'], s=1, c='gray', alpha=0.3)
            plt.scatter(allocate_grids['cmaq_x'], allocate_grids['cmaq_y'], s=1, c='g')
            plt.scatter(value_grids[:,0], value_grids[:,1], s=1, c='b')
            plt.scatter(nan_grids[:,0], nan_grids[:,1], s=1, c='purple', alpha=0.3)
            plt.scatter(point_xy['cmaq_x'], point_xy['cmaq_y'], s=1, c='r')
            plt.show()

    def compose_multi_grid(self, grid_left: int, grid_right: int):
        all_cmaq_grids = []
        for i in range(len(self.monitor_coord)):
            point_xy = self.monitor_coord.iloc[i]
            cmaq_grids, _, _ = _compose_grid_coord(self.whole_coord, point_xy, self.longest_dist, grid_left, grid_right)
            all_cmaq_grids.append(cmaq_grids)
        all_cmaq_grids = np.stack(all_cmaq_grids)
        return all_cmaq_grids
        
    def allocate_grids_data(self, whole_data, grids_cmaq, monitoring_data):
        all_dates = monitoring_data[['day', 'month']].drop_duplicates()

        all_grid_data = []
        for date_id in range(len(all_dates[:3])):
            date = all_dates.iloc[date_id]
            date_whole_data = whole_data.loc[np.all(whole_data[['day', 'month']]==date, axis=1)]
            date_monitoring_data = monitoring_data.loc[np.all(monitoring_data[['day', 'month']]==date, axis=1)]
            date_data_shape = [len(date_monitoring_data)] + list(grids_cmaq.shape[1:]) + [whole_data.shape[1]]
            date_grid_data = np.full(date_data_shape, np.nan)
            for dt_id in range(date_data_shape[0]):
                monitor_site_cmaq_id = date_monitoring_data.iloc[dt_id]['cmaq_id']
                monitor_site_coord_index = self.monitor_coord.index[self.monitor_coord['cmaq_id']==monitor_site_cmaq_id]
                monitor_grid_set = grids_cmaq[monitor_site_coord_index][0]
                for v_id in range(monitor_grid_set.shape[0]):
                    for h_id in range(monitor_grid_set.shape[1]):
                        cmaq_id = monitor_grid_set[v_id, h_id]
                        if not np.isnan(cmaq_id):
                            cmaq_data = date_whole_data.loc[date_whole_data['cmaq_id']==cmaq_id]
                            if len(cmaq_data) > 0:
                                date_grid_data[dt_id, v_id, h_id] = cmaq_data
            all_grid_data.append(date_grid_data)
        all_grid_data = np.vstack(all_grid_data)
        return all_grid_data

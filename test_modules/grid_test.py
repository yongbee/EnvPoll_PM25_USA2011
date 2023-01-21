import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from data_process.compose import MultiGridCompose

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

def find_dist_threshold(whole_dt):
    coord_dt = whole_dt[["cmaq_x", "cmaq_y"]]
    all_min_distance = get_distance_statistic(coord_dt)
    outlier_threshold = np.percentile(all_min_distance, 99.9)
    return outlier_threshold

def plot_map(whole_xy, monitor_xy):
    plt.scatter(whole_xy['cmaq_x'], whole_xy['cmaq_y'], s=1, c='gray', alpha=0.3)
    plt.scatter(monitor_xy['cmaq_x'], monitor_xy['cmaq_y'], s=1, c='r')
    plt.show()

def compute_square_points(point_xy, longest_dist, grid_left, grid_right):
    square_top = point_xy['cmaq_y'] + grid_left*longest_dist
    square_bot = point_xy['cmaq_y'] - grid_right*longest_dist
    square_left = point_xy['cmaq_x'] - grid_left*longest_dist
    square_right = point_xy['cmaq_x'] + grid_right*longest_dist
    return {'top':square_top, 'bot':square_bot, 'left':square_left, 'right':square_right}

def grid_dataset(whole_xy, point_xy, longest_dist, grid_left, grid_right):
    sqaure_points = compute_square_points(point_xy, longest_dist, grid_left, grid_right)
    square_xy = whole_xy.loc[
        (whole_xy['cmaq_y']<=sqaure_points['top']) &
        (whole_xy['cmaq_y']>=sqaure_points['bot']) &
        (whole_xy['cmaq_x']<=sqaure_points['right']) &
        (whole_xy['cmaq_x']>=sqaure_points['left'])
    ]
    return square_xy

def plot_grid_dataset(whole_xy, monitoring_xy, longest_dist, grid_left, grid_right):
    np.random.seed(1000)
    monitoring_xy_shuffle = monitoring_xy.sample(frac=1)
    for i in range(10):
        point_xy = monitoring_xy_shuffle.iloc[i]
        square_xy = grid_dataset(whole_xy, point_xy, longest_dist, grid_left, grid_right)
        plt.scatter(whole_xy['cmaq_x'], whole_xy['cmaq_y'], s=1, c='gray', alpha=0.3)
        plt.scatter(square_xy['cmaq_x'], square_xy['cmaq_y'], s=1, c='g')
        plt.scatter(point_xy['cmaq_x'], point_xy['cmaq_y'], s=1, c='r')
        plt.show()

def compose_grid_coord(whole_xy, point_xy, longest_dist, grid_left, grid_right):
    larger_grids = grid_dataset(whole_xy, point_xy, longest_dist, grid_left+2, grid_right+2)
    sqaure_points = compute_square_points(point_xy, longest_dist, grid_left, grid_right)
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

def plot_grid_allocation(whole_xy, monitoring_xy, longest_dist, grid_left, grid_right):
    np.random.seed(1000)
    monitoring_xy_shuffle = monitoring_xy.sample(frac=1)
    for i in range(10):
        point_xy = monitoring_xy_shuffle.iloc[i]
        cmaq_grids, larger_grids, all_grids = compose_grid_coord(whole_xy, point_xy, longest_dist, grid_left, grid_right)
        allocate_grid_id = cmaq_grids[~np.isnan(cmaq_grids)]
        allocate_grids = larger_grids.loc[np.isin(larger_grids['cmaq_id'], allocate_grid_id)]
        value_grids = np.flip(all_grids[~np.isnan(cmaq_grids),:])
        nan_grids = np.flip(all_grids[np.isnan(cmaq_grids),:])
        plt.scatter(whole_xy['cmaq_x'], whole_xy['cmaq_y'], s=1, c='gray', alpha=0.3)
        plt.scatter(allocate_grids['cmaq_x'], allocate_grids['cmaq_y'], s=1, c='g')
        plt.scatter(value_grids[:,0], value_grids[:,1], s=1, c='b')
        plt.scatter(nan_grids[:,0], nan_grids[:,1], s=1, c='purple', alpha=0.3)
        plt.scatter(point_xy['cmaq_x'], point_xy['cmaq_y'], s=1, c='r')
        plt.show()

def save_multi_grid(whole_xy, monitoring_xy, longest_dist, grid_left, grid_right):
    grid_size = grid_left + grid_right + 1
    all_cmaq_grids = []
    for i in range(len(monitoring_xy)):
        point_xy = monitoring_xy.iloc[i]
        cmaq_grids, _, _ = compose_grid_coord(whole_xy, point_xy, longest_dist, grid_left, grid_right)
        all_cmaq_grids.append(cmaq_grids)
    all_cmaq_grids = np.stack(all_cmaq_grids)
    np.save(f"data/grid{grid_size}_cmaq_id", all_cmaq_grids)

def allocate_grids_data(whole_data, grids_cmaq, monitoring_data, monitoring_coord):
    all_dates = monitoring_data[['day', 'month']].drop_duplicates()

    all_grid_data = []
    for date_id in range(len(all_dates)):
        date = all_dates.iloc[date_id]
        date_whole_data = whole_data.loc[np.all(whole_data[['day', 'month']]==date, axis=1)]
        date_monitoring_data = monitoring_data.loc[np.all(monitoring_data[['day', 'month']]==date, axis=1)]
        date_data_shape = [len(date_monitoring_data)] + list(grids_cmaq.shape[1:]) + [whole_data.shape[1]]
        date_grid_data = np.full(date_data_shape, np.nan)
        for dt_id in range(date_data_shape[0]):
            monitor_site_cmaq_id = date_monitoring_data.iloc[dt_id]['cmaq_id']
            monitor_site_coord_index = monitoring_coord.index[monitoring_coord['cmaq_id']==monitor_site_cmaq_id]
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

def _save_multi_grid(all_cmaq_grids: np.ndarray, grid_left: int, grid_right: int):
    grid_size = grid_left + grid_right + 1
    np.save(f"data/grid{grid_size}_cmaq_id", all_cmaq_grids)

if __name__ == '__main__':
    small_set_test = True
    columns = ['day', 'month', 'cmaq_x', 'cmaq_y', 'cmaq_id', 'rid', 'elev', 'forest_cover', 'pd', 'local', 'limi', 'high', 'is', 
    'nldas_pevapsfc','nldas_pressfc', 'nldas_cape', 'nldas_ugrd10m', 'nldas_vgrd10m', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_dlwrfsfc', 
    'nldas_dswrfsfc', 'nldas_pcpsfc', 'nldas_fpcsfc', 'gc_aod', 'aod_value', 'emissi11_pm25', 'pm25_value_k', 'pm25_value']

    coord_whole_data = pd.read_csv("data/largeUS_coords_pred.csv", index_col=0)
    monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")
    coord_data = coord_whole_data.drop_duplicates().reset_index(drop=True)
    monitoring_coord = monitoring_whole_data[["cmaq_x", "cmaq_y", "cmaq_id"]].drop_duplicates().reset_index(drop=True)
    left_grids, right_grids = 31, 32

    multi_compose = MultiGridCompose(coord_data, monitoring_coord)
    all_cmaq_grids = multi_compose.compose_multi_grid(left_grids, right_grids)
    _save_multi_grid(all_cmaq_grids, left_grids, right_grids)
    if small_set_test:
        all_multigrid_dataset = multi_compose.allocate_grids_data(monitoring_whole_data[columns], all_cmaq_grids, monitoring_whole_data[columns])
    else:
        whole_data = pd.read_csv("data/largeUS_pred.csv.gz", compression='gzip', index_col=0)
        all_multigrid_dataset = multi_compose.allocate_grids_data(whole_data[columns], all_cmaq_grids, monitoring_whole_data[columns])
    a=3

import numpy as np
import pandas as pd
from data_process.data import tag_names
from data_process.compose import MultiGridCompose

def _save_multi_grid(all_cmaq_grids: np.ndarray, grid_left: int, grid_right: int):
    grid_size = grid_left + grid_right + 1
    np.save(f"data/grid{grid_size}_cmaq_id", all_cmaq_grids)

def _read_multi_grid(grid_left: int, grid_right: int):
    grid_size = grid_left + grid_right + 1
    return np.load(f"data/grid{grid_size}_cmaq_id.npy")

def _save_multi_grid_dataset(multigrid_dataset: np.ndarray, grid_left: int, grid_right: int):
    grid_size = grid_left + grid_right + 1
    np.save(f"data/grid{grid_size}_dataset", multigrid_dataset)

def _save_multi_grid_dataset_target(multigrid_dataset: np.ndarray, multigrid_target: np.ndarray, grid_left: int, grid_right: int):
    grid_size = grid_left + grid_right + 1
    np.save(f"data/grid{grid_size}_dataset_without_WA", multigrid_dataset)
    np.save(f"data/grid{grid_size}_target", multigrid_target)

if __name__ == '__main__':
    small_set_test = False
    save_grid = True
    weight_pm_target = True
    left_grids, right_grids = 31, 32
    
    coord_whole_data = pd.read_csv("data/largeUS_coords_pred.csv", index_col=0)
    monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")
    coord_data = coord_whole_data.drop_duplicates().reset_index(drop=True)
    monitoring_coord = monitoring_whole_data[["cmaq_x", "cmaq_y", "cmaq_id"]].drop_duplicates().reset_index(drop=True)

    multi_compose = MultiGridCompose(coord_data, monitoring_coord)
    if save_grid:
        all_cmaq_grids = multi_compose.compose_multi_grid(left_grids, right_grids)
        _save_multi_grid(all_cmaq_grids, left_grids, right_grids)
    else:
        all_cmaq_grids = _read_multi_grid(left_grids, right_grids)
    
    if small_set_test:
        whole_data = monitoring_whole_data[tag_names]
    else:
        whole_data = pd.read_csv("data/largeUS_pred.csv.gz", compression='gzip', index_col=0)
    if weight_pm_target:
        all_multigrid_dataset, all_multigrid_target = multi_compose.allocate_grids_data(whole_data[tag_names], all_cmaq_grids, monitoring_whole_data[tag_names], weight_pm_target)
        _save_multi_grid_dataset_target(all_multigrid_dataset, all_multigrid_target, left_grids, right_grids)
    else:
        all_multigrid_dataset = multi_compose.allocate_grids_data(whole_data[tag_names], all_cmaq_grids, monitoring_whole_data[tag_names], weight_pm_target)
        _save_multi_grid_dataset(all_multigrid_dataset, left_grids, right_grids)

import numpy as np
import pandas as pd
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

if __name__ == '__main__':
    small_set_test = False
    save_grid = True
    left_grids, right_grids = 31, 32
    columns = ['day', 'month', 'cmaq_x', 'cmaq_y', 'cmaq_id', 'rid', 'elev', 'forest_cover', 'pd', 'local', 'limi', 'high', 'is', 
    'nldas_pevapsfc','nldas_pressfc', 'nldas_cape', 'nldas_ugrd10m', 'nldas_vgrd10m', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_dlwrfsfc', 
    'nldas_dswrfsfc', 'nldas_pcpsfc', 'nldas_fpcsfc', 'gc_aod', 'aod_value', 'emissi11_pm25', 'pm25_value_k', 'pm25_value']

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
        all_multigrid_dataset = multi_compose.allocate_grids_data(monitoring_whole_data[columns], all_cmaq_grids, monitoring_whole_data[columns])
    else:
        whole_data = pd.read_csv("data/largeUS_pred.csv.gz", compression='gzip', index_col=0)
        all_multigrid_dataset = multi_compose.allocate_grids_data(whole_data[columns], all_cmaq_grids, monitoring_whole_data[columns])
    _save_multi_grid_dataset(all_multigrid_dataset, left_grids, right_grids)

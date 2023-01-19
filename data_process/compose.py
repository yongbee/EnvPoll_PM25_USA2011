import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix

class SqaureGrid:
    def __init__(self, size):
        self.size = size

    def _get_monitoring_distance(self, whole_xy: pd.DataFrame):
        all_distance = distance_matrix(whole_xy.values, whole_xy.values)
        all_distance[all_distance==0] = np.nan
        return all_distance.min()

    def compose_grid_coord(self, xy_coord: pd.DataFrame, label_data: pd.DataFrame):
        unique_whole_xy = xy_coord.drop_duplicates()
        unique_label_xy = label_data[["x","y"]].drop_duplicates()
        


import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

def _cluster_coords(coordinates: pd.DataFrame, method: str, n_clusters: int):
    if "x" not in coordinates.columns:
        raise Exception("'x' must be in the input coordinate columns.")
    if "y" not in coordinates.columns:
        raise Exception("'y' must be in the input coordinate columns.")
    if method not in ["GaussianMixture", "KMeans"]:
        raise Exception("Inappropriate method.")

    unique_coors = coordinates.drop_duplicates()
    if method == 'GaussianMixture':
        cluster_model = GaussianMixture(n_components=n_clusters).fit(unique_coors)
    elif method == "KMeans":
        cluster_model = KMeans(n_clusters=n_clusters).fit(unique_coors)
    coor_pred = cluster_model.predict(unique_coors)
    whole_pred = cluster_model.predict(coordinates)
    coor_clusters = unique_coors.copy()
    coor_clusters["cluster id"] = coor_pred
    whole_df = pd.DataFrame(whole_pred, index=coordinates.index, columns=["cluster id"])
    return whole_df, coor_clusters
    
class SingleGrid:
    def __init__(self, cluster_method="GaussianMixture", cluster_num=10):
        self.cluster_num = cluster_num
        self.cluster_method = cluster_method

    def split_train_test(self, input_dt: pd.DataFrame, target_dt: pd.Series):
        if type(input_dt) is not pd.DataFrame:
            raise Exception("Input data type is not pd.DataFrame.")
        if type(target_dt) is not pd.Series:
            raise Exception("Target data type is not pd.Series.")
        if not input_dt.index.equals(target_dt.index):
            raise Exception("Input and Output indexes are not equal.")
        if "x" not in input_dt.columns:
            raise Exception("'x' must be in the input data columns.")
        if "y" not in input_dt.columns:
            raise Exception("'y' must be in the input data columns.")

        return _cluster_coords(input_dt[["x", "y"]], self.cluster_method, self.cluster_num)
        
class MultipleGrid(SingleGrid):
    def __init__(self, grid_scale, cluster_method="GaussianMixture", cluster_num=10):
        super().__init__(cluster_method, cluster_num)
        self.grid_scale = grid_scale

    def split_train_test(self, tag_names: list, input_dt: np.ndarray, target_dt: pd.Series):
        if type(input_dt) is not np.ndarray:
            raise Exception("Input data type is not np.array.")
        if input_dt.shape[1] != (len(tag_names)*(self.grid_scale**2)):
            raise Exception("Input data column number is inappropriate.")
        if type(target_dt) is not pd.Series:
            raise Exception("Target data type is not pd.Series.")
        if len(input_dt) != len(target_dt):
            raise Exception("The length of input data and target data are not equal.")
        
        center_cell_id = (self.grid_scale**2)//2
        center_dt = input_dt[:,center_cell_id*len(tag_names):(center_cell_id+1)*len(tag_names)]
        center_frame = pd.DataFrame(center_dt, columns=tag_names, index=target_dt.index)
        return super().split_train_test(center_frame, target_dt)

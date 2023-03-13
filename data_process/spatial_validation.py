import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
    
def _split_in_cluster(coordinates: pd.DataFrame):
    if "cmaq_x" not in coordinates.columns:
        raise Exception("'cmaq_x' must be in the input coordinate columns.")
    if "cmaq_y" not in coordinates.columns:
        raise Exception("'cmaq_y' must be in the input coordinate columns.")

    unique_coors, coor_nums = np.unique(coordinates, axis=0, return_counts=True)
    unique_coors = pd.DataFrame(unique_coors, columns=coordinates.columns)
    large_num_coors = unique_coors[coor_nums>=np.percentile(coor_nums,60)]
    train_idx = np.random.choice(large_num_coors.index, size=len(coor_nums)//10)
    train_coors =  unique_coors.loc[train_idx]
    test_coors =  unique_coors.drop(train_idx)
    return train_coors, test_coors

def _split_train_test(xy_cluster: pd.DataFrame):
    if type(xy_cluster) is not pd.DataFrame:
        raise Exception("xy_cluster data is not pd.DataFrame.")

    np.random.seed(1000)
    unique_xy_cluster = xy_cluster.drop_duplicates()
    all_split_grids, all_split_data_id = {}, {}
    for cluster in np.sort(pd.unique(unique_xy_cluster["cluster id"])):
        cluster_xy = xy_cluster.loc[xy_cluster["cluster id"]==cluster, ["cmaq_x","cmaq_y"]]
        out_cluster_xy = unique_xy_cluster.loc[unique_xy_cluster["cluster id"]!=cluster, ["cmaq_x","cmaq_y"]].drop_duplicates()
        cluster_train, cluster_test = _split_in_cluster(cluster_xy)
        all_split_grids[cluster] = {
            "train_in_cluster":cluster_train,
            "train_out_cluster":out_cluster_xy,
            "test_cluster":cluster_test
        }
        all_split_data_id[cluster] = {
            "train_in_cluster":xy_cluster.index[np.isin(xy_cluster[["cmaq_x","cmaq_y"]], cluster_train).min(axis=1)],
            "train_out_cluster":xy_cluster.index[np.isin(xy_cluster[["cmaq_x","cmaq_y"]], out_cluster_xy).min(axis=1)],
            "test_cluster":xy_cluster.index[np.isin(xy_cluster[["cmaq_x","cmaq_y"]], cluster_test).min(axis=1)]
        }
    return all_split_grids, all_split_data_id

def extract_center_data(tag_names: list, input_dt: np.ndarray, grid_scale: int):
    if type(input_dt) is not np.ndarray:
        raise Exception("Input data type is not np.array.")
    if input_dt.shape[1] != (len(tag_names)*(grid_scale**2)):
        raise Exception("Input data column number is inappropriate.")
        
    center_cell_id = (grid_scale**2)//2
    center_dt = input_dt[:,center_cell_id*len(tag_names):(center_cell_id+1)*len(tag_names)]
    return center_dt

def get_clusters(input_dt: pd.DataFrame, label_dt: pd.Series):
    single_grid = SingleGrid("KMeans")
    whole_cluster, _ = single_grid.cluster_grids(input_dt, pd.Series(label_dt))
    _, train_test_data_id = single_grid.split_train_test(input_dt, whole_cluster)
    return train_test_data_id, single_grid.cluster_model

class SingleGrid:
    def __init__(self, cluster_method="GaussianMixture", cluster_num=10):
        self.cluster_num = cluster_num
        self.cluster_method = cluster_method

    def _train_cluster_model(self, coordinates: pd.DataFrame, method: str, n_clusters: int):
        np.random.seed(1000)
        if method == 'GaussianMixture':
            self.cluster_model = GaussianMixture(n_components=n_clusters).fit(coordinates)
        elif method == "KMeans":
            self.cluster_model = KMeans(n_clusters=n_clusters).fit(coordinates)

    def _cluster_coords(self, coordinates: pd.DataFrame, method: str, n_clusters: int):
        if "cmaq_x" not in coordinates.columns:
            raise Exception("'cmaq_x' must be in the input coordinate columns.")
        if "cmaq_y" not in coordinates.columns:
            raise Exception("'cmaq_y' must be in the input coordinate columns.")
        if "cmaq_id" not in coordinates.columns:
            raise Exception("'cmaq_id' must be in the input coordinate columns.")
        if method not in ["GaussianMixture", "KMeans"]:
            raise Exception("Inappropriate method.")

        unique_coors = coordinates.drop_duplicates()
        self._train_cluster_model(unique_coors[["cmaq_x", "cmaq_y"]], method, n_clusters)
        coor_pred = self.cluster_model.predict(unique_coors[["cmaq_x", "cmaq_y"]])
        whole_pred = self.cluster_model.predict(coordinates[["cmaq_x", "cmaq_y"]])
        coor_clusters = unique_coors.copy()
        coor_clusters["cluster id"] = coor_pred
        whole_df = pd.DataFrame(whole_pred, index=coordinates.index, columns=["cluster id"])
        return whole_df, coor_clusters

    def cluster_grids(self, input_dt: pd.DataFrame, target_dt: pd.Series):
        if type(input_dt) is not pd.DataFrame:
            raise Exception("Input data type is not pd.DataFrame.")
        if type(target_dt) is not pd.Series:
            raise Exception("Target data type is not pd.Series.")
        if not input_dt.index.equals(target_dt.index):
            raise Exception("Input and Output indexes are not equal.")
        if "cmaq_x" not in input_dt.columns:
            raise Exception("'cmaq_x' must be in the input data columns.")
        if "cmaq_y" not in input_dt.columns:
            raise Exception("'cmaq_y' must be in the input data columns.")

        return self._cluster_coords(input_dt[["cmaq_x", "cmaq_y", "cmaq_id"]], self.cluster_method, self.cluster_num)

    def split_train_test(self, input_dt: pd.DataFrame, whole_cluster: pd.DataFrame):
        if type(input_dt) is not pd.DataFrame:
            raise Exception("Input data type is not pd.DataFrame.")
        if type(whole_cluster) is not pd.DataFrame:
            raise Exception("Whole Cluster data type is not pd.DataFrame.")

        xy_cluster = input_dt[["cmaq_x", "cmaq_y"]].join(whole_cluster)
        return _split_train_test(xy_cluster)
        
class TargetSingle:
    def __init__(self, set_num=10):
        self.set_num = set_num

    def split_data_coord(self, coord_dt: pd.DataFrame, train_num: int):
        if type(coord_dt) is not pd.DataFrame:
            raise Exception("Coordinate data type is not pd.DataFrame.")
        if type(train_num) is not int:
            raise Exception("Train monitor number must be int.")
        
        np.random.seed(1000)
        set_dataset = {}
        for set_id in range(self.set_num):
            train_coord = coord_dt["cmaq_id"].sample(train_num)
            set_train_coord = coord_dt[np.isin(coord_dt["cmaq_id"], train_coord)]
            set_test_coord = coord_dt[~np.isin(coord_dt["cmaq_id"], train_coord)]
            set_dataset[set_id] = {"train": set_train_coord, "test": set_test_coord}
        return set_dataset

class MultipleGrid(SingleGrid):
    def __init__(self, grid_scale, cluster_method="GaussianMixture", cluster_num=10):
        super().__init__(cluster_method, cluster_num)
        self.grid_scale = grid_scale

    def cluster_grids(self, tag_names: list, input_dt: np.ndarray, target_dt: pd.Series):
        if type(input_dt) is not np.ndarray:
            raise Exception("Input data type is not np.array.")
        if input_dt.shape[1] != (len(tag_names)*(self.grid_scale**2)):
            raise Exception("Input data column number is inappropriate.")
        if type(target_dt) is not pd.Series:
            raise Exception("Target data type is not pd.Series.")
        if len(input_dt) != len(target_dt):
            raise Exception("The length of input data and target data are not equal.")

        center_dt = extract_center_data(tag_names, input_dt, self.grid_scale)
        center_frame = pd.DataFrame(center_dt, columns=tag_names, index=target_dt.index)
        return super().cluster_grids(center_frame, target_dt)

    def split_train_test(self, tag_names: list, input_dt: np.ndarray, whole_cluster: pd.DataFrame):
        if type(input_dt) is not np.ndarray:
            raise Exception("Input data type is not np.array.")
        if input_dt.shape[1] != (len(tag_names)*(self.grid_scale**2)):
            raise Exception("Input data column number is inappropriate.")
        if type(whole_cluster) is not pd.DataFrame:
            raise Exception("Whole Cluster data type is not pd.DataFrame.")

        center_dt = extract_center_data(tag_names, input_dt, self.grid_scale)
        center_frame = pd.DataFrame(center_dt, columns=tag_names, index=whole_cluster.index)
        return super().split_train_test(center_frame, whole_cluster)

class TargetMultiple(TargetSingle):
    def __init__(self, grid_scale, set_num=10):
        super().__init__(set_num)
        self.grid_scale = grid_scale

    def split_data_coord(self, tag_names: list, coord_dt: np.ndarray, train_num: int):
        if coord_dt.shape[1] != (len(tag_names)*(self.grid_scale**2)):
            raise Exception("Input data column number is inappropriate.")
        if type(coord_dt) is not np.ndarray:
            raise Exception("Coordinate data type is not np.ndarray.")
        if type(train_num) is not int:
            raise Exception("Train monitor number must be int.")

        center_coord = extract_center_data(tag_names, coord_dt, self.grid_scale)
        center_frame = pd.DataFrame(center_coord, columns=tag_names)
        return super().split_data_coord(center_frame, train_num)

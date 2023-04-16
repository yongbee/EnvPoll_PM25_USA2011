import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from torch.utils.data import Dataset, DataLoader

tag_names = ['day', 'month', 'cmaq_x', 'cmaq_y', 'cmaq_id', 'rid', 'elev', 'forest_cover', 'pd', 'local', 'limi', 'high', 'is', 
    'nldas_pevapsfc','nldas_pressfc', 'nldas_cape', 'nldas_ugrd10m', 'nldas_vgrd10m', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_dlwrfsfc', 
    'nldas_dswrfsfc', 'nldas_pcpsfc', 'nldas_fpcsfc', 'gc_aod', 'aod_value', 'emissi11_pm25', 'pm25_value']

def _convert_loader(input_dt:np.ndarray, output_dt:np.ndarray, batch:int):
    if len(input_dt) < 1:
        raise Exception("input_dt length is 0.")
    if len(output_dt) < 1:
        raise Exception("output_dt length is 0.")
    dt_set = InputOutputSet(input_dt, output_dt)
    dt_loader = DataLoader(dt_set, batch_size=batch, shuffle=False, pin_memory=True)
    return dt_loader

def cluster_train_valid_index(set_index: dict):
    train_in = list(set_index['train_in_cluster'])
    train_out = list(set_index['train_out_cluster'])
    train_index = train_in + train_out
    valid_index = list(set_index['test_cluster'])
    return train_index, valid_index

def _drop_constant_col(train_dt: pd.DataFrame, valid_dt: pd.DataFrame):
    _std = train_dt.std(axis=0)
    train_dt_variable = train_dt.loc[:,_std>0]
    valid_dt_variable = valid_dt.loc[:, _std>0]
    return train_dt_variable, valid_dt_variable

def _drop_na_col(train_dt: pd.DataFrame, valid_dt: pd.DataFrame):
    train_drop_dt = train_dt.dropna(axis=1)
    valid_drop_dt = valid_dt[train_drop_dt.columns]
    return train_drop_dt, valid_drop_dt

def _drop_useless_col(train_data, valid_data):
    train_drop_na, valid_drop_na = _drop_na_col(train_data, valid_data)
    train_drop_const, valid_drop_const = _drop_constant_col(train_drop_na, valid_drop_na)
    return train_drop_const, valid_drop_const

def _sort_distance_stations(distance_data: pd.DataFrame):
        nearest_stations = pd.DataFrame(columns=range(distance_data.shape[1]), index=distance_data.index)
        for row in distance_data.index:
            nearest_stations.loc[row] = distance_data.columns[distance_data.loc[row].argsort()]
        return nearest_stations

def create_distance_matrix(dt1: pd.DataFrame, dt2: pd.DataFrame):
    all_distance = distance_matrix(dt1, dt2)
    all_distance[all_distance==0] = np.inf
    all_distance = pd.DataFrame(all_distance, index=dt1.index, columns=dt2.index)
    return all_distance

class WeightAverage:
    def __init__(self, train_data, valid_data, train_label):
        self.train_data = train_data
        self.valid_data = valid_data
        self.train_label = train_label
        self.cmaq_cols = ["cmaq_x", "cmaq_y", "cmaq_id"]
        self._allocate_weight()
        self._compute_all_wa()

    def _allocate_weight(self):
        train_cmaq = pd.DataFrame(np.unique(self.train_data[self.cmaq_cols], axis=0), columns=self.cmaq_cols).set_index("cmaq_id")
        valid_cmaq = pd.DataFrame(np.unique(self.valid_data[self.cmaq_cols], axis=0), columns=self.cmaq_cols).set_index("cmaq_id")
        self.train_weight = 1 / create_distance_matrix(train_cmaq, train_cmaq)
        self.valid_weight = 1 / create_distance_matrix(valid_cmaq, train_cmaq)

    def _date_weight_average(self, data: pd.DataFrame, weight, train_label, train_cmaq):
        exist_weight = weight.loc[data["cmaq_id"], np.isin(weight.columns, train_cmaq)]
        weight_label = train_label[exist_weight.columns]
        weight_sum = np.sum(exist_weight, axis=1)
        cmaq_wa = np.sum(exist_weight*weight_label, axis=1)/weight_sum
        cmaq_wa.index = data.index
        return cmaq_wa
        
    def _compute_date_wa(self, date):
        date_train_data = self.train_data.loc[self.train_data["day"]==date].copy()
        date_valid_data = self.valid_data.loc[self.valid_data["day"]==date].copy()
        date_train_label = self.train_label.loc[self.train_data["day"]==date]
        date_train_label.index = date_train_data["cmaq_id"]
        train_wa = self._date_weight_average(date_train_data, self.train_weight, date_train_label, date_train_data["cmaq_id"])
        valid_wa = self._date_weight_average(date_valid_data, self.valid_weight, date_train_label, date_train_data["cmaq_id"])
        date_train_data["pm25_wa"] = train_wa
        date_valid_data["pm25_wa"] = valid_wa
        return date_train_data, date_valid_data

    def _compute_all_wa(self):
        all_dates = np.unique(self.train_data["day"])
        all_train_data, all_valid_data = [], []
        for date in all_dates:
            date_train, date_valid = self._compute_date_wa(date)
            all_train_data.append(date_train)
            all_valid_data.append(date_valid)
        all_train_data = pd.concat(all_train_data)
        all_valid_data = pd.concat(all_valid_data)
        self.weight_inputs = (all_train_data, all_valid_data)

class StationAllocate:
    def __init__(self, train_data, valid_data, train_label, exclude_cols, station_num):
        self.train_data = train_data
        self.valid_data = valid_data
        self.train_label = train_label
        self.exclude_cols = exclude_cols
        self.station_num = station_num
        self.cmaq_cols = ["cmaq_x", "cmaq_y", "cmaq_id"]
        self._compute_distances()
        self.train_sort_stations = _sort_distance_stations(self.train_distance)
        self.valid_sort_stations = _sort_distance_stations(self.valid_distance)
        self._allocate_all_data()

    def _compute_distances(self):
        train_cmaq = pd.DataFrame(np.unique(self.train_data[self.cmaq_cols], axis=0), columns=self.cmaq_cols).set_index("cmaq_id")
        valid_cmaq = pd.DataFrame(np.unique(self.valid_data[self.cmaq_cols], axis=0), columns=self.cmaq_cols).set_index("cmaq_id")
        self.train_distance = create_distance_matrix(train_cmaq, train_cmaq)
        self.valid_distance = create_distance_matrix(valid_cmaq, train_cmaq)

    def _date_allocate_data(self, data: pd.DataFrame, train_data: pd.DataFrame, sort_stations: pd.DataFrame, train_label):
        cmaq_id_data = data.set_index("cmaq_id")
        cmaq_id_data["PM25"] = 0
        cmaq_id_train = train_data.set_index("cmaq_id")
        cmaq_id_train["PM25"] = train_label
        date_stations = sort_stations.loc[np.isin(sort_stations.index, cmaq_id_data.index)]

        allocate_number = pd.Series(0, cmaq_id_data.index)
        station_data = []
        for s in range(self.station_num):
            not_allocated_bool = pd.Series(True, cmaq_id_data.index)
            num_station_data = pd.DataFrame(index=cmaq_id_data.index, columns=cmaq_id_train.columns)
            while not_allocated_bool.sum() > 0:
                unique_nums = np.unique(allocate_number[not_allocated_bool])
                if max(unique_nums) > date_stations.columns[-1]:
                    raise Exception("The number of train data in the date is not enough.")
                for num in unique_nums:
                    num_index = cmaq_id_data.index[(allocate_number==num) & (not_allocated_bool)]
                    nearest_cmaq = date_stations[num].loc[num_index]
                    allocate_index = nearest_cmaq.index[np.isin(nearest_cmaq, cmaq_id_train.index)]
                    not_allocated_bool.loc[allocate_index] = False
                    allocate_cmaq = nearest_cmaq[allocate_index]
                    num_station_data.loc[allocate_cmaq.index] = np.array(cmaq_id_train.loc[allocate_cmaq])
                allocate_number[not_allocated_bool] += 1
            station_data.append(num_station_data.copy())
            allocate_number += 1
        station_data.insert(len(station_data)//2, cmaq_id_data)
        stack_station_data = np.stack(station_data, -1)
        return stack_station_data
        
    def _compute_date_wa(self, date):
        date_train_data = self.train_data.loc[self.train_data["day"]==date].copy()
        date_valid_data = self.valid_data.loc[self.valid_data["day"]==date].copy()
        date_train_label = self.train_label.loc[self.train_data["day"]==date]
        date_train_label.index = date_train_data["cmaq_id"]
        date_train_dataset = self._date_allocate_data(date_train_data, date_train_data, self.train_sort_stations, date_train_label)
        date_valid_dataset = self._date_allocate_data(date_valid_data, date_train_data, self.valid_sort_stations, date_train_label)
        return date_train_dataset, date_valid_dataset

    def _allocate_all_data(self):
        all_dates = np.unique(self.train_data["day"])
        all_train_data, all_valid_data = [], []
        for date_num, date in enumerate(all_dates):
            print(f"date {date_num}")
            date_train, date_valid = self._compute_date_wa(date)
            all_train_data.append(date_train)
            all_valid_data.append(date_valid)
        all_train_data = np.vstack(all_train_data)
        all_valid_data = np.vstack(all_valid_data)
        self.near_inputs = (all_train_data, all_valid_data)

class InputOutputSet(Dataset):
    def __init__(self, input_dt, output_dt):
        super().__init__()
        self.input_dt = input_dt
        self.output_dt = output_dt

    def __getitem__(self, i):
        return self.input_dt[i], self.output_dt[i]

    def __len__(self):
        return len(self.input_dt)

class SingleData:
    def __init__(self, input_dt: pd.DataFrame, label_dt: pd.Series, train_valid_data_id: dict, id_type: str, normalize=False, weight_average_compute=False):
        self.input_dt = input_dt
        self.label_dt = label_dt
        self.train_valid_data_id = train_valid_data_id
        self.weight_average_compute = weight_average_compute
        if id_type == "index":
            self.split_train_valid_index()
        elif id_type == "cmaq_id":
            self.split_train_valid_cmaq()
        if normalize:
            self._normalize_train_valid()

    def split_train_valid_index(self):
        self.train_dt, self.valid_dt = {}, {}
        self.input_dim = {}
        for cluster_id in self.train_valid_data_id.keys():
            print(f"cluster{cluster_id} compsing...")
            set_index = self.train_valid_data_id[cluster_id]
            train_index, valid_index = cluster_train_valid_index(set_index)
            train_input, train_label = self.input_dt.loc[train_index], self.label_dt[train_index]
            valid_input, valid_label = self.input_dt.loc[valid_index], self.label_dt[valid_index]
            train_input, valid_input = _drop_useless_col(train_input, valid_input)
            if self.weight_average_compute:
                weight_average = WeightAverage(train_input, valid_input, train_label)
                train_input, valid_input = weight_average.weight_inputs
            self.train_dt[cluster_id] = {"input":train_input, "label":train_label}
            self.valid_dt[cluster_id] = {"input":valid_input, "label":valid_label}
            self.input_dim[cluster_id] = train_input.shape[1]

    def split_train_valid_cmaq(self):
        self.train_dt, self.valid_dt = {}, {}
        self.input_dim = {}
        for cluster_id in self.train_valid_data_id.keys():
            print(f"cluster{cluster_id} compsing...")
            set_index = self.train_valid_data_id[cluster_id]
            train_index, valid_index = cluster_train_valid_index(set_index)
            train_input, train_label = self.input_dt.loc[np.isin(self.input_dt["cmaq_id"], train_index)], self.label_dt[np.isin(self.input_dt["cmaq_id"], train_index)]
            valid_input, valid_label = self.input_dt.loc[np.isin(self.input_dt["cmaq_id"], valid_index)], self.label_dt[np.isin(self.input_dt["cmaq_id"], valid_index)]
            train_input, valid_input = _drop_useless_col(train_input, valid_input)
            if self.weight_average_compute:
                weight_average = WeightAverage(train_input, valid_input, train_label)
                train_input, valid_input = weight_average.weight_inputs
            self.train_dt[cluster_id] = {"input":train_input, "label":train_label}
            self.valid_dt[cluster_id] = {"input":valid_input, "label":valid_label}
            self.input_dim[cluster_id] = train_input.shape[1]
        
    def _normalize_train_valid(self):
        for cluster_id in self.train_dt.keys():
            train_input = self.train_dt[cluster_id]["input"]
            valid_input = self.valid_dt[cluster_id]["input"]
            mean, std = train_input.mean(axis=0), train_input.std(axis=0)
            self.train_dt[cluster_id]["input"] = (train_input - mean) / std
            self.valid_dt[cluster_id]["input"] = (valid_input - mean) / std

    def data_convert_loader(self):
        for cluster_id in self.train_dt.keys():
            train_input = np.array(self.train_dt[cluster_id]["input"])
            train_label = np.array(self.train_dt[cluster_id]["label"])
            valid_input = np.array(self.valid_dt[cluster_id]["input"])
            valid_label = np.array(self.valid_dt[cluster_id]["label"])
            train_loader = _convert_loader(train_input, train_label, 128)
            valid_loader = _convert_loader(valid_input, valid_label, 128)
            self.train_dt[cluster_id] = train_loader
            self.valid_dt[cluster_id] = valid_loader

class MultipleData:
    def __init__(self, input_dt: np.ndarray, label_dt: pd.Series, tag_names: list, grid_width: int, train_valid_data_id: dict, id_type: str, normalize=False):
        self.input_dt = input_dt
        self.label_dt = label_dt
        self.tag_names = tag_names
        self.grid_width = grid_width
        self.center_df = self._get_center_data()
        self.train_valid_data_id = train_valid_data_id
        if id_type == "index":
            self.split_train_valid_index()
        elif id_type == "cmaq_id":
            self.split_train_valid_cmaq()
        if normalize:
            self._normalize_train_valid()

    def _get_center_data(self):
        if len(tag_names) * (self.grid_width**2) != self.input_dt.shape[1]:
            raise Exception("input data, tag names, and grid width are wrongly correlated.")
        center_grid_id = (self.grid_width**2) // 2
        center_arr = self.input_dt[len(tag_names)*center_grid_id:len(tag_names)*(center_grid_id+1)]
        center_df = pd.DataFrame(center_arr, columns=tag_names)
        return center_df

    def split_train_valid_index(self):
        self.train_dt, self.valid_dt = {}, {}
        self.input_dim = {}
        for cluster_id in self.train_valid_data_id.keys():
            set_index = self.train_valid_data_id[cluster_id]
            train_index, valid_index = cluster_train_valid_index(set_index)
            train_input, train_label = self.input_dt[train_index], self.label_dt[train_index]
            valid_input, valid_label = self.input_dt[valid_index], self.label_dt[valid_index]
            train_input, valid_input = _drop_useless_col(train_input, valid_input)
            self.train_dt[cluster_id] = {"input":train_input, "label":train_label}
            self.valid_dt[cluster_id] = {"input":valid_input, "label":valid_label}
            self.input_dim[cluster_id] = train_input.shape[1]

    def split_train_valid_cmaq(self):
        self.train_dt, self.valid_dt = {}, {}
        self.input_dim = {}
        for cluster_id in self.train_valid_data_id.keys():
            set_index = self.train_valid_data_id[cluster_id]
            train_index, valid_index = cluster_train_valid_index(set_index)
            train_input, train_label = self.input_dt[np.isin(self.center_df["cmaq_id"], train_index)], self.label_dt[np.isin(self.center_df["cmaq_id"], train_index)]
            valid_input, valid_label = self.input_dt[np.isin(self.center_df["cmaq_id"], valid_index)], self.label_dt[np.isin(self.center_df["cmaq_id"], valid_index)]
            train_input, valid_input = _drop_useless_col(train_input, valid_input)
            self.train_dt[cluster_id] = {"input":train_input, "label":train_label}
            self.valid_dt[cluster_id] = {"input":valid_input, "label":valid_label}
            self.input_dim[cluster_id] = train_input.shape[1]
        
    def _normalize_train_valid(self):
        for cluster_id in self.train_dt.keys():
            train_input = self.train_dt[cluster_id]["input"]
            valid_input = self.valid_dt[cluster_id]["input"]
            mean, std = train_input.mean(axis=0), train_input.std(axis=0)
            self.train_dt[cluster_id]["input"] = (train_input - mean) / std
            self.valid_dt[cluster_id]["input"] = (valid_input - mean) / std

    def data_convert_loader(self):
        for cluster_id in self.train_dt.keys():
            train_input = np.array(self.train_dt[cluster_id]["input"])
            train_label = np.array(self.train_dt[cluster_id]["label"])
            valid_input = np.array(self.valid_dt[cluster_id]["input"])
            valid_label = np.array(self.valid_dt[cluster_id]["label"])
            train_loader = _convert_loader(train_input, train_label, 128)
            valid_loader = _convert_loader(valid_input, valid_label, 128)
            self.train_dt[cluster_id] = train_loader
            self.valid_dt[cluster_id] = valid_loader

class NearStationData:
    def __init__(self, input_dt: pd.DataFrame, label_dt: pd.Series, train_valid_data_id: dict, exclude_cols=[], normalize=False):
        self.input_dt = input_dt
        self.label_dt = label_dt
        self.train_valid_data_id = train_valid_data_id
        self.exclude_cols = exclude_cols
        self.split_train_valid_cmaq()
        if normalize:
            self._normalize_train_valid()
        self._allocate_near_data()

    def split_train_valid_cmaq(self):
        self.train_dt, self.valid_dt = {}, {}
        self.input_dim = {}
        for cluster_id in self.train_valid_data_id.keys():
            set_index = self.train_valid_data_id[cluster_id]
            train_index, valid_index = cluster_train_valid_index(set_index)
            train_input, train_label = self.input_dt.loc[np.isin(self.input_dt["cmaq_id"], train_index)], self.label_dt[np.isin(self.input_dt["cmaq_id"], train_index)]
            valid_input, valid_label = self.input_dt.loc[np.isin(self.input_dt["cmaq_id"], valid_index)], self.label_dt[np.isin(self.input_dt["cmaq_id"], valid_index)]
            train_input, valid_input = _drop_useless_col(train_input, valid_input)
            self.train_dt[cluster_id] = {"input":train_input, "label":train_label}
            self.valid_dt[cluster_id] = {"input":valid_input, "label":valid_label}
            self.input_dim[cluster_id] = train_input.shape[1]
        
    def _normalize_train_valid(self):
        for cluster_id in self.train_dt.keys():
            train_input = self.train_dt[cluster_id]["input"]
            valid_input = self.valid_dt[cluster_id]["input"]
            mean, std = train_input.mean(axis=0), train_input.std(axis=0)
            mean[self.exclude_cols], std[self.exclude_cols] = 0, 1
            self.train_dt[cluster_id]["input"] = (train_input - mean) / std
            self.valid_dt[cluster_id]["input"] = (valid_input - mean) / std

    # def _remove_cols(self, exclude_cols):
    #     for cluster_id in self.train_dt.keys():
    #         train_input = self.train_dt[cluster_id]["input"]
    #         valid_input = self.valid_dt[cluster_id]["input"]
    #         self.train_dt[cluster_id]["input"] = train_input.drop(columns=exclude_cols)
    #         self.valid_dt[cluster_id]["input"] = valid_input.drop(columns=exclude_cols)

    def _allocate_near_data(self):
        for cluster_id in self.train_dt.keys():
            print(f"cluster{cluster_id} compsing...")
            train_input = self.train_dt[cluster_id]["input"]
            valid_input = self.valid_dt[cluster_id]["input"]
            train_label = self.train_dt[cluster_id]["label"]
            station_allocate = StationAllocate(train_input, valid_input, train_label, self.exclude_cols, 6)
            

    def data_convert_loader(self):
        for cluster_id in self.train_dt.keys():
            train_input = np.array(self.train_dt[cluster_id]["input"])
            train_label = np.array(self.train_dt[cluster_id]["label"])
            valid_input = np.array(self.valid_dt[cluster_id]["input"])
            valid_label = np.array(self.valid_dt[cluster_id]["label"])
            train_loader = _convert_loader(train_input, train_label, 128)
            valid_loader = _convert_loader(valid_input, valid_label, 128)
            self.train_dt[cluster_id] = train_loader
            self.valid_dt[cluster_id] = valid_loader

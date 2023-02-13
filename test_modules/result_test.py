import numpy as np
import pandas as pd
from data_process.spatial_validation import SingleGrid
from model.result import SingleAnalyzer

def _get_clusters(input_dt: pd.DataFrame, label_dt: pd.Series):
    single_grid = SingleGrid("KMeans")
    whole_cluster, _ = single_grid.cluster_grids(input_dt, pd.Series(label_dt))
    _, train_test_data_id = single_grid.split_train_test(input_dt, whole_cluster)
    return train_test_data_id, single_grid.cluster_model

def _get_input_label(input_dt: pd.DataFrame, label_dt: pd.DataFrame, train_test_data_id: dict):
    all_inputs, all_labels = [], []
    for cluster_id in train_test_data_id.keys():
        cluster_test_index = train_test_data_id[cluster_id]['test_cluster']
        all_labels.append(label_dt[cluster_test_index])
        all_inputs.append(input_dt.loc[cluster_test_index])
    all_labels = pd.concat(all_labels)
    all_inputs = pd.concat(all_inputs)
    return all_inputs, all_labels

def _get_results(model_name: str, index):
    result_data = np.load(f"result/{model_name}.npz")
    all_pred_data = []
    for cluster_id in range(10):
        all_pred_data.append(result_data[f"cluster{cluster_id}"])
    all_pred_data = pd.Series(np.hstack(all_pred_data), index=index)
    return all_pred_data

if __name__=='__main__':
    model_name = "RF"
    tag_names = ['day', 'month', 'cmaq_x', 'cmaq_y', 'cmaq_id', 'rid', 'elev', 'forest_cover', 'pd', 'local', 'limi', 'high', 'is', 
    'nldas_pevapsfc','nldas_pressfc', 'nldas_cape', 'nldas_ugrd10m', 'nldas_vgrd10m', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_dlwrfsfc', 
    'nldas_dswrfsfc', 'nldas_pcpsfc', 'nldas_fpcsfc', 'gc_aod', 'aod_value', 'emissi11_pm25', 'pm25_value_k', 'pm25_value']
    monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")[tag_names]
    coord_whole_data = pd.read_csv("data/largeUS_coords_pred.csv", index_col=0)
    whole_coord = coord_whole_data.drop_duplicates().reset_index(drop=True)[['cmaq_x', 'cmaq_y']]

    input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
    label_dt = monitoring_whole_data["pm25_value"]
    train_test_data_id, cluster_model = _get_clusters(input_dt, label_dt)
    all_input, all_label = _get_input_label(input_dt, label_dt, train_test_data_id)
    all_pred = _get_results(model_name, all_label.index)

    single_analyzer = SingleAnalyzer(model_name, cluster_model, whole_coord, all_input, all_label, all_pred, train_test_data_id)
    single_analyzer.scatter_label_pred()
    single_analyzer.plot_whole_cluster(True, 1)
    single_analyzer.plot_pm_vals()
    single_analyzer.plot_accuracy_map()
    single_analyzer.plot_cluster_accuracy()
    single_analyzer.hist_accuracy()
    single_analyzer.scatter_std_accuracy()

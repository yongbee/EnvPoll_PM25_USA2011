import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from data_process.spatial_validation import SingleGrid
from model.result import SingleAnalyzer

def _get_clusters(input_dt: pd.DataFrame, label_dt: pd.Series):
    single_grid = SingleGrid("KMeans")
    whole_cluster, _ = single_grid.cluster_grids(input_dt, pd.Series(label_dt))
    _, train_test_data_id = single_grid.split_train_test(input_dt, whole_cluster)
    return train_test_data_id, single_grid.cluster_model

def _get_labels(label_dt: pd.DataFrame, train_test_data_id: dict):
    all_labels = []
    for cluster_id in train_test_data_id.keys():
        cluster_test_index = train_test_data_id[cluster_id]['test_cluster']
        all_labels.append(label_dt[cluster_test_index])
    all_labels = np.hstack(all_labels)
    return all_labels

def _get_results(model_name: str):
    result_data = np.load(f"result/{model_name}.npz")
    all_pred_data = []
    for cluster_id in range(10):
        all_pred_data.append(result_data[f"cluster{cluster_id}"])
    all_pred_data = np.hstack(all_pred_data)
    return all_pred_data

if __name__=='__main__':
    model_name = "FNN"
    tag_names = ['day', 'month', 'cmaq_x', 'cmaq_y', 'cmaq_id', 'rid', 'elev', 'forest_cover', 'pd', 'local', 'limi', 'high', 'is', 
    'nldas_pevapsfc','nldas_pressfc', 'nldas_cape', 'nldas_ugrd10m', 'nldas_vgrd10m', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_dlwrfsfc', 
    'nldas_dswrfsfc', 'nldas_pcpsfc', 'nldas_fpcsfc', 'gc_aod', 'aod_value', 'emissi11_pm25', 'pm25_value_k', 'pm25_value']
    monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")[tag_names]
    coord_whole_data = pd.read_csv("data/largeUS_coords_pred.csv", index_col=0)
    whole_coord = coord_whole_data.drop_duplicates().reset_index(drop=True)[['cmaq_x', 'cmaq_y']]

    input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
    label_dt = monitoring_whole_data["pm25_value"]
    train_test_data_id, cluster_model = _get_clusters(input_dt, label_dt)
    all_label = _get_labels(label_dt, train_test_data_id)
    all_pred = _get_results(model_name)
    r2_val = round(r2_score(all_label, all_pred), 4)
    mae_val = round(mean_absolute_error(all_label, all_pred), 4)

    single_analyzer = SingleAnalyzer(cluster_model, whole_coord)
    single_analyzer.plot_whole_cluster(True, 1)
    single_analyzer.plot_pm_vals(monitoring_whole_data)
    single_analyzer.plot_errors(train_test_data_id, monitoring_whole_data, model_name)
    print(f"Whole MAE: {mae_val}")
    print(f"Whole R-squared: {r2_val}")

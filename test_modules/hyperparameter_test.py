import numpy as np
import pandas as pd
from data_process.spatial_validation import SingleGrid
from data_process.data import SingleData
from model.scikit import HyperparameterTest

def _get_clusters(input_dt: pd.DataFrame, label_dt: pd.Series):
    single_grid = SingleGrid("KMeans")
    whole_cluster, _ = single_grid.cluster_grids(input_dt, pd.Series(label_dt))
    _, train_test_data_id = single_grid.split_train_test(input_dt, whole_cluster)
    return train_test_data_id

if __name__=='__main__':
    model_name = "GBM"
    model_hyperparameters = {"learning_rate":[0.01, 0.05, 0.1], "n_estimators":[100, 500, 1000], "max_depth":[6, 7, 8]}
    columns = ['day', 'month', 'cmaq_x', 'cmaq_y', 'cmaq_id', 'rid', 'elev', 'forest_cover', 'pd', 'local', 'limi', 'high', 'is', 
    'nldas_pevapsfc','nldas_pressfc', 'nldas_cape', 'nldas_ugrd10m', 'nldas_vgrd10m', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_dlwrfsfc', 
    'nldas_dswrfsfc', 'nldas_pcpsfc', 'nldas_fpcsfc', 'gc_aod', 'aod_value', 'emissi11_pm25', 'pm25_value_k', 'pm25_value']
    monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")[columns]

    input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
    label_dt = monitoring_whole_data["pm25_value"]
    train_test_data_id = _get_clusters(input_dt, label_dt)
    single_data = SingleData(input_dt, label_dt, train_test_data_id, True)
    model_train_test = HyperparameterTest(model_name, model_hyperparameters)
    model_train_test.hyperparameter_test(single_data.train_dt, single_data.valid_dt)

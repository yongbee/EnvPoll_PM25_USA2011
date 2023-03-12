import numpy as np
import pandas as pd
from data_process.spatial_validation import get_clusters
from data_process.data import SingleData
from model.scikit import TrainTest

def _save_results(all_pred: dict, model_name: str):
    save_dir = f"result/{model_name}.npz"
    np.savez(save_dir, **all_pred)

if __name__=='__main__':
    model_name = "GBM"
    columns = ['day', 'month', 'cmaq_x', 'cmaq_y', 'cmaq_id', 'rid', 'elev', 'forest_cover', 'pd', 'local', 'limi', 'high', 'is', 
    'nldas_pevapsfc','nldas_pressfc', 'nldas_cape', 'nldas_ugrd10m', 'nldas_vgrd10m', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_dlwrfsfc', 
    'nldas_dswrfsfc', 'nldas_pcpsfc', 'nldas_fpcsfc', 'gc_aod', 'aod_value', 'emissi11_pm25', 'pm25_value_k', 'pm25_value']
    monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")[columns]

    input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
    label_dt = monitoring_whole_data["pm25_value"]
    train_test_data_id, _ = get_clusters(input_dt, label_dt)
    single_data = SingleData(input_dt, label_dt, train_test_data_id, True)
    model_train_test = TrainTest(model_name)
    model_train_test.train(single_data.train_dt)
    all_pred = model_train_test.predict(single_data.valid_dt)
    _save_results(all_pred, model_name)

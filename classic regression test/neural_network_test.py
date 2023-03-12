import numpy as np
import pandas as pd
from data_process.spatial_validation import get_clusters
from data_process.data import SingleData, tag_names
from model.neuralnet import TrainTest

def _save_results(all_pred: dict, model_name: str):
    save_dir = f"result/{model_name}.npz"
    np.savez(save_dir, **all_pred)

if __name__=='__main__':
    model_name = "FNN"
    monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")[tag_names]

    input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
    label_dt = monitoring_whole_data["pm25_value"]
    train_test_data_id, _ = get_clusters(input_dt, label_dt)
    single_data = SingleData(input_dt, label_dt, train_test_data_id, True)
    single_data.data_convert_loader()
    model_train_test = TrainTest(model_name, single_data.input_dim)
    model_train_test.train(single_data.train_dt, 25)
    all_pred = model_train_test.predict(single_data.valid_dt)
    _save_results(all_pred, model_name)

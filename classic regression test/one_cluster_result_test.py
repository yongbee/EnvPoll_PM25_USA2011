import numpy as np
import pandas as pd
from data_process.spatial_validation import get_in_clusters
from data_process.data import tag_names
from model.result import SingleAnalyzerCmaq

def _get_input_label(input_dt: pd.DataFrame, label_dt: pd.DataFrame, train_test_data_id: dict):
    all_inputs, all_labels = [], []
    for cluster_id in train_test_data_id.keys():
        cluster_test_cmaq = train_test_data_id[cluster_id]['test_cluster']
        all_labels.append(label_dt[np.isin(input_dt["cmaq_id"], cluster_test_cmaq)])
        all_inputs.append(input_dt.loc[np.isin(input_dt["cmaq_id"], cluster_test_cmaq)])
    all_labels = pd.concat(all_labels)
    all_inputs = pd.concat(all_inputs)
    return all_inputs, all_labels

def _get_results(model_name: str, index):
    result_data = np.load(f"result/target cluster split/{model_name}.npz")
    all_pred_data = []
    for cluster_id in range(10):
        all_pred_data.append(result_data[f"cluster{cluster_id}"])
    all_pred_data = pd.Series(np.hstack(all_pred_data), index=index)
    return all_pred_data

if __name__=='__main__':
    coord_check = False
    model_name = "GBM_cluster4"
    data_path, train_num = "data/split-data/single/", 10

    model_full_name = f"{model_name}-{train_num}"
    monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")[tag_names]
    coord_whole_data = pd.read_csv("data/largeUS_coords_pred.csv", index_col=0)
    whole_coord = coord_whole_data.drop_duplicates().reset_index(drop=True)[['cmaq_x', 'cmaq_y', 'cmaq_id']]

    input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
    label_dt = monitoring_whole_data["pm25_value"]
    train_test_data_id = get_in_clusters(data_path, train_num)
    test_input, test_label = _get_input_label(input_dt, label_dt, train_test_data_id)
    test_pred = _get_results(model_full_name, test_label.index)

    single_analyzer = SingleAnalyzerCmaq(model_full_name, whole_coord, test_input, test_label, test_pred, train_test_data_id)
    print(single_analyzer.cluster_info)
    single_analyzer.scatter_label_pred()
    single_analyzer.hist_accuracy()
    single_analyzer.scatter_std_accuracy()
    if coord_check:
        single_analyzer.plot_dataset_map()

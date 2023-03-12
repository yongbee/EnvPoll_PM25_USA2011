import pandas as pd
from data_process.spatial_validation import get_clusters
from data_process.data import SingleData, tag_names
from model.scikit import HyperparameterTest

if __name__=='__main__':
    model_name = "GBM"
    # model_hyperparameters = {"learning_rate":[0.01, 0.05, 0.1], "n_estimators":[100, 500, 1000], "max_depth":[6, 7, 8]}
    # model_hyperparameters = {"learning_rate":[0.01], "n_estimators":[100, 500, 1000], "max_depth":[6]}
    learning_rate_hyperparameters = {"learning_rate":[0.01, 0.05, 0.1], "n_estimators":[100], "max_depth":[6]}
    depth_hyperparameters = {"learning_rate":[0.01], "n_estimators":[100], "max_depth":[6, 7, 8]}
    monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")[tag_names]

    input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
    label_dt = monitoring_whole_data["pm25_value"]
    train_test_data_id, _ = get_clusters(input_dt, label_dt)
    single_data = SingleData(input_dt, label_dt, train_test_data_id, True)
    model_train_test = HyperparameterTest(model_name, learning_rate_hyperparameters)
    model_train_test.hyperparameter_test(single_data.train_dt, single_data.valid_dt, "learning_rate")

    model_train_test = HyperparameterTest(model_name, depth_hyperparameters)
    model_train_test.hyperparameter_test(single_data.train_dt, single_data.valid_dt, "depth")

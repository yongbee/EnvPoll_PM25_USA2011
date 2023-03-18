import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_process.data import tag_names
from data_process.spatial_validation import get_clusters

def _get_cluster_input_label(input_dt: pd.DataFrame, label_dt: pd.DataFrame, train_test_data_id: dict, cluster_id: int):
    cluster_test_index = train_test_data_id[cluster_id]['test_cluster']
    target_label = label_dt[cluster_test_index]
    target_input = input_dt.loc[cluster_test_index]
    target_data = pd.concat([target_input, target_label], axis=1)
    source_label = label_dt[~label_dt.index.isin(cluster_test_index)]
    source_input = input_dt[~input_dt.index.isin(cluster_test_index)]
    source_data = pd.concat([source_input, source_label], axis=1)
    return {"target":target_data, "source":source_data}

def _boxplot_compare(source_data, target_data):
    fig_row, fig_col = 4, 4
    fig_num = fig_row * fig_col
    for col_id, col in enumerate(source_data.columns):
        col_source = source_data[col].dropna()
        col_target = target_data[col].dropna()
        if col_id % fig_num == 0:
            fig, axes = plt.subplots(nrows=fig_row, ncols=fig_col, figsize=(12, 9))
            fig.tight_layout()
            start_col_id = col_id
        boxplot_id = col_id % fig_num
        boxplot_col = boxplot_id // fig_row
        boxplot_row = boxplot_id % fig_row
        axes[boxplot_col][boxplot_row].boxplot([col_source, col_target])
        axes[boxplot_col][boxplot_row].set_xticklabels(["source", "target"])
        axes[boxplot_col][boxplot_row].set_title(col)
        if (col_id % fig_num == fig_num - 1) or col_id == source_data.shape[1]-1:
            plt.savefig(f"figures/whole boxplot col {start_col_id}-{col_id}")
            plt.cla()
            plt.clf()

def _correlation_compare(source_data, target_data):
    source_corr = source_data.corr()
    target_corr = target_data.corr()
    np.fill_diagonal(source_corr.values, 0)
    np.fill_diagonal(target_corr.values, 0)
    plt.figure(figsize=(6,12))
    plt.subplot(3,1,1)
    plt.title("Source")
    plt.imshow(source_corr, cmap="RdYlBu", vmin=-1, vmax=1)
    plt.colorbar()
    plt.subplot(3,1,2)
    plt.title("Target")
    plt.imshow(target_corr, cmap="RdYlBu", vmin=-1, vmax=1)
    plt.colorbar()
    plt.subplot(3,1,3)
    plt.title("Difference")
    plt.imshow(np.abs(source_corr-target_corr), vmin=0, vmax=2)
    plt.colorbar()
    plt.savefig("figures/correlation matrix comapre")
    plt.cla()
    plt.clf()

if __name__=='__main__':
    cluster_id = 4
    monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")[tag_names]
    coord_whole_data = pd.read_csv("data/largeUS_coords_pred.csv", index_col=0)
    whole_coord = coord_whole_data.drop_duplicates().reset_index(drop=True)[['cmaq_x', 'cmaq_y']]

    input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
    label_dt = monitoring_whole_data["pm25_value"]
    train_test_data_id, cluster_model = get_clusters(input_dt, label_dt)
    data_combination = _get_cluster_input_label(input_dt, label_dt, train_test_data_id, cluster_id)
    _boxplot_compare(data_combination["source"], data_combination["target"])
    _correlation_compare(data_combination["source"], data_combination["target"])

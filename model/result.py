import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

class SingleAnalyzer:
    def __init__(self, cluster_model, whole_coord):
        self.cluster_model = cluster_model
        self.whole_coord = whole_coord
        self.whole_clsuter = cluster_model.predict(whole_coord)

    def plot_whole_cluster(self, save, alpha):
        plt.figure(figsize=(15,8))
        for cluster_id in np.sort(np.unique(self.whole_clsuter)):
            cluster_coords = self.whole_coord.iloc[self.whole_clsuter==cluster_id]
            plt.scatter(cluster_coords['cmaq_x'], cluster_coords['cmaq_y'], s=3, alpha=alpha, label=cluster_id)
        plt.legend()
        if save:
            plt.savefig("figures/USA_whole_cluster")
        
    def plot_pm_vals(self, monitoring_data: pd.DataFrame):
        monitor_coor_pm = monitoring_data[['cmaq_x', 'cmaq_y', 'cmaq_id', 'pm25_value']]
        unique_coords = monitor_coor_pm.drop_duplicates(subset=['cmaq_id'])
        all_pm_mean = []
        for unique_coor_id in unique_coords.index:
            coor_id = unique_coords.loc[unique_coor_id, 'cmaq_id']
            coor_data = monitor_coor_pm.loc[monitor_coor_pm['cmaq_id']==coor_id]
            pm_mean = coor_data['pm25_value'].mean()
            all_pm_mean.append(pm_mean)
        unique_coords['pm25_value'] = all_pm_mean
        min_val = np.percentile(unique_coords['pm25_value'], 5)
        max_val = np.percentile(unique_coords['pm25_value'], 95)

        self.plot_whole_cluster(False, 0.1)
        plt.scatter(unique_coords['cmaq_x'], unique_coords['cmaq_y'], s=3, 
        c=unique_coords['pm25_value'], cmap='turbo', vmin=min_val, vmax=max_val)
        plt.colorbar()
        plt.savefig("figures/PM25_whole_average_values")
        plt.cla()
        plt.clf()

    def plot_errors(self, train_test_id: dict, monitoring_data: pd.DataFrame, model_name: str):
        all_pred_data = np.load(f"result/{model_name}.npz")
        all_coord_values = []
        for cluster_id in train_test_id.keys():
            cluster_test_index = train_test_id[cluster_id]['test_cluster']
            cluster_label =  monitoring_data.loc[cluster_test_index, "pm25_value"]
            cluster_pred = all_pred_data[f"cluster{cluster_id}"]
            cluster_r2 = round(r2_score(cluster_label, cluster_pred), 4)
            cluster_mae = round(mean_absolute_error(cluster_label, cluster_pred), 4)
            print(f"cluster{cluster_id}")
            print(f"R-squared: {cluster_r2}")
            print(f"MAE: {cluster_mae}")
            print('-'*70)

            cluster_xy_val = monitoring_data.loc[cluster_test_index, ['cmaq_x', 'cmaq_y', 'cmaq_id', 'pm25_value']]
            unique_coords = cluster_xy_val.drop_duplicates(subset=['cmaq_id'])
            coords_values = unique_coords.copy()
            all_pm_mean, all_r2 = [], []
            for unique_coor_id in unique_coords.index:
                coor_id = unique_coords.loc[unique_coor_id, 'cmaq_id']
                coor_data = cluster_xy_val.loc[cluster_xy_val['cmaq_id']==coor_id]
                coor_pred = cluster_pred[cluster_xy_val['cmaq_id']==coor_id]
                pm_mean = coor_data['pm25_value'].mean()
                r2_val = r2_score(coor_data['pm25_value'], coor_pred)
                all_pm_mean.append(pm_mean)
                all_r2.append(r2_val)
            coords_values['pm25_value'] = all_pm_mean
            coords_values['r2_value'] = all_r2
            all_coord_values.append(coords_values.copy())
        all_coord_values = pd.concat(all_coord_values)

        min_val = np.percentile(all_coord_values['pm25_value'], 5)
        max_val = np.percentile(all_coord_values['pm25_value'], 95)
        self.plot_whole_cluster(False, 0.1)
        plt.scatter(all_coord_values['cmaq_x'], all_coord_values['cmaq_y'], s=3, 
        c=all_coord_values['pm25_value'], cmap='turbo', vmin=min_val, vmax=max_val)
        plt.colorbar()
        plt.savefig("figures/PM25_test_average_values")
        plt.cla()
        plt.clf()

        self.plot_whole_cluster(False, 0.1)
        plt.scatter(all_coord_values['cmaq_x'], all_coord_values['cmaq_y'], s=3, 
        c=all_coord_values['r2_value'], cmap='rainbow', vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(f"figures/{model_name}_R2_map")
        plt.cla()
        plt.clf()

        high_r2 = all_coord_values['r2_value'][all_coord_values['r2_value']>0]
        plt.subplot(1,2,1)
        plt.hist(all_coord_values['r2_value'], bins=30)
        plt.subplot(1,2,2)
        plt.hist(high_r2, bins=30)
        plt.savefig(f"figures/{model_name}_R2_histogram")
        plt.cla()
        plt.clf()

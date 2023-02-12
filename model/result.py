import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error

class SingleAnalyzer:
    def __init__(self, model_name, cluster_model, whole_coord, input_data: pd.DataFrame, label_data: pd.Series, pred_data: pd.Series, train_test_id):
        self.model_name = model_name
        self.cluster_model = cluster_model
        self.whole_coord = whole_coord
        self.input_data = input_data
        self.label_data = label_data
        self.pred_data = pred_data
        self.train_test_id = train_test_id
        self.whole_clsuter = cluster_model.predict(whole_coord)
        
        self.coord_info = self._compute_coord_info()
        self.cluster_info = self._compute_cluster_info()

    def _compute_coord_info(self):
        monitor_coor = self.input_data[['cmaq_x', 'cmaq_y', 'cmaq_id']]
        coord_info = monitor_coor.drop_duplicates(subset=['cmaq_id']).reset_index(drop=True)
        all_pm_mean, all_r2, all_corr, all_pm_std, all_mae = [], [], [], [], []
        for unique_coor_id in coord_info.index:
            coor_id = coord_info.loc[unique_coor_id, 'cmaq_id']
            coor_data = self.input_data.loc[self.input_data['cmaq_id']==coor_id]
            if len(coor_data) < 5:
                coord_info = coord_info[coord_info["cmaq_id"] != coor_id]
                continue
            coor_label = self.label_data[self.input_data['cmaq_id']==coor_id]
            coor_pred = self.pred_data[self.input_data['cmaq_id']==coor_id]
            pm_mean = coor_label.mean()
            pm_std = coor_label.std()
            r2_val = r2_score(coor_label, coor_pred)
            corr_val = stats.pearsonr(coor_label, coor_pred).statistic
            mae_val = mean_absolute_error(coor_label, coor_pred)
            all_pm_mean.append(pm_mean)
            all_pm_std.append(pm_std)
            all_r2.append(r2_val)
            all_corr.append(corr_val)
            all_mae.append(mae_val)
        coord_info['pm25_mean'] = all_pm_mean
        coord_info['pm25_std'] = all_pm_std
        coord_info['pm25_r2'] = all_r2
        coord_info['pm25_corr'] = all_corr
        coord_info['pm25_mae'] = all_mae
        return coord_info

    def _compute_cluster_info(self):
        cluster_info = {}
        for cluster_id in self.train_test_id.keys():
            cluster_test_index = self.train_test_id[cluster_id]['test_cluster']
            cluster_label = self.label_data[cluster_test_index]
            cluster_pred = self.pred_data[cluster_test_index]
            cluster_r2 = round(r2_score(cluster_label, cluster_pred), 4)
            cluster_mae = round(mean_absolute_error(cluster_label, cluster_pred), 4)
            cluster_info[cluster_id] = {
                'r2':cluster_r2, 'mae':cluster_mae
            }
        cluster_info["whole"] = {
            "r2": round(r2_score(self.label_data, self.pred_data), 4),
            "mae": round(mean_absolute_error(self.label_data, self.pred_data), 4),
        }
        return cluster_info

    def plot_whole_cluster(self, save, alpha):
        plt.figure(figsize=(15,8))
        for cluster_id in np.sort(np.unique(self.whole_clsuter)):
            cluster_coords = self.whole_coord.iloc[self.whole_clsuter==cluster_id]
            plt.scatter(cluster_coords['cmaq_x'], cluster_coords['cmaq_y'], s=3, alpha=alpha, label=cluster_id)
        plt.legend()
        if save:
            plt.savefig("figures/USA_whole_cluster")
        
    def plot_pm_vals(self):
        min_val = np.percentile(self.coord_info['pm25_mean'], 5)
        max_val = np.percentile(self.coord_info['pm25_mean'], 95)
        self.plot_whole_cluster(False, 0.1)
        plt.scatter(self.coord_info['cmaq_x'], self.coord_info['cmaq_y'], s=3, 
        c=self.coord_info['pm25_mean'], cmap='turbo', vmin=min_val, vmax=max_val)
        plt.colorbar()
        plt.savefig("figures/PM25_whole_average_values")
        plt.cla()
        plt.clf()

    def plot_accuracy_map(self):
        self.plot_whole_cluster(False, 0.1)
        plt.scatter(self.coord_info['cmaq_x'], self.coord_info['cmaq_y'], s=3, 
        c=self.coord_info['pm25_r2'], cmap='rainbow', vmin=0, vmax=1)
        plt.title(self.model_name)
        plt.colorbar()
        plt.savefig(f"figures/{self.model_name}_R2_map")
        plt.cla()
        plt.clf()

    def hist_accuracy(self):
        high_r2 = self.coord_info['pm25_r2'][self.coord_info['pm25_r2']>0]
        fig, axs = plt.subplots(2,1)
        axs[0].hist(self.coord_info['pm25_r2'], bins=30)
        axs[1].hist(high_r2, bins=30)
        fig.suptitle(self.model_name)
        plt.savefig(f"figures/{self.model_name}_R2_histogram")
        plt.cla()
        plt.clf()

    def scatter_std_accuracy(self):
        fig, axs = plt.subplots(2,2)
        fig.set_figheight(9)
        fig.set_figwidth(9)
        axs[0,0].scatter(self.coord_info['pm25_std'], self.coord_info['pm25_r2'], s=3)
        axs[0,0].set_xlabel("std")
        axs[0,0].set_ylabel("R-squared")
        axs[0,1].scatter(self.coord_info['pm25_std'], self.coord_info['pm25_mae'], s=3)
        axs[0,1].set_xlabel("std")
        axs[0,1].set_ylabel("mae")
        axs[1,0].scatter(self.coord_info['pm25_std'], self.coord_info['pm25_corr'], s=3)
        axs[1,0].set_xlabel("std")
        axs[1,0].set_ylabel("correlation")
        plt.savefig(f"figures/{self.model_name}_std_R2")
        plt.cla()
        plt.clf()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error

def _density_scatter(x_val, y_val, ax):
    xy_val = np.vstack([x_val, y_val])
    z = stats.gaussian_kde(xy_val)(xy_val)
    ax.scatter(x_val, y_val, c=z, s=3)

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
            plt.cla()
            plt.clf()

    def plot_cluster_accuracy(self):
        all_cluster_accuracy = []
        for cluster_id in np.sort(np.unique(self.whole_clsuter)):
            cluster_coords = self.whole_coord.iloc[self.whole_clsuter==cluster_id].copy()
            cluster_coords["r2"] = self.cluster_info[cluster_id]["r2"]
            cluster_coords["mae"] = self.cluster_info[cluster_id]["mae"]
            all_cluster_accuracy.append(cluster_coords)
        all_cluster_accuracy = pd.concat(all_cluster_accuracy)
        plt.figure(figsize=(15,4.5))
        plt.subplot(1,2,1)
        plt.scatter(all_cluster_accuracy['cmaq_x'], all_cluster_accuracy['cmaq_y'], c=all_cluster_accuracy["r2"], s=3, cmap="autumn")
        plt.colorbar()
        plt.title("R-squared")
        plt.subplot(1,2,2)
        plt.scatter(all_cluster_accuracy['cmaq_x'], all_cluster_accuracy['cmaq_y'], c=all_cluster_accuracy["mae"], s=3, cmap="autumn_r")
        plt.colorbar()
        plt.title("MAE")
        plt.savefig(f"figures/{self.model_name}_cluster_accuracy")
        plt.cla()
        plt.clf()
        
    def plot_pm_vals(self):
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.hist(self.label_data, bins=30)
        plt.title("all PM2.5")
        plt.subplot(1,2,2)
        plt.hist(self.coord_info['pm25_mean'], bins=30)
        plt.title("grid average PM2.5")
        plt.savefig("figures/PM25_grid_average_distribution")
        plt.cla()
        plt.clf()
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

    def scatter_label_pred(self):
        month_season = {
            "Spring": [3, 4, 5],
            "Summer": [6, 7, 8],
            "Fall": [9, 10, 11],
            "Winter": [12, 1, 2]
        }
        fig = plt.figure(figsize=(16,8))
        gs = GridSpec(2,4) # 2 rows, 3 columns
        ax1 = fig.add_subplot(gs[:2,:2]) # First row, first column
        ax2 = fig.add_subplot(gs[0,2]) # First row, second column
        ax3 = fig.add_subplot(gs[0,3]) # First row, third column
        ax4 = fig.add_subplot(gs[1,2])
        ax5 = fig.add_subplot(gs[1,3])
        max_val = np.max(np.hstack([self.label_data, self.pred_data]))
        min_val = np.min(np.hstack([self.label_data, self.pred_data]))
        ax1.scatter(self.label_data, self.pred_data, s=3)
        ax1.set_xlim(min_val, max_val)
        ax1.set_ylim(min_val, max_val)
        ax1.set_title("Whole")
        ax1.set_xlabel("Label")
        ax1.set_ylabel("Prediction")
        for season_id, season in enumerate(month_season.keys()):
            season_ax = eval(f"ax{season_id+2}")
            season_label = self.label_data[np.isin(self.input_data["month"], month_season[season])]
            season_pred = self.pred_data[np.isin(self.input_data["month"], month_season[season])]
            # max_val = np.max(np.hstack([season_label, season_pred]))
            # min_val = np.min(np.hstack([season_label, season_pred]))
            season_ax.scatter(season_label, season_pred, s=3)
            season_ax.set_xlim(min_val, max_val)
            season_ax.set_ylim(min_val, max_val)
            season_ax.set_title(season)
        plt.savefig(f"figures/{self.model_name}_label_pred_scatter")
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

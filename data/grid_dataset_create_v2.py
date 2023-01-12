import pandas as pd
import sys
import numpy as np
from pandas import DataFrame
from math import sqrt
import numpy.ma as ma
import math
print("Repositories uploaded!!")


def create_grid(df_BUS_monitoring, df_BUS_predictions, rast_us):
    N = 5
    N_ = int(N/2)
    key = 0
    targetlist = []
    traininglist = []
    training_difflist = []

    for day in range(1,366):
        print(day)
        df_BUS_pd_1 = df_BUS_predictions[df_BUS_predictions['day'] == day]
        df_BUS_pd_1 = df_BUS_pd_1.reset_index(drop = True)

        df_BUS_mn_1 = df_BUS_monitoring[df_BUS_monitoring['day'] == day]
        df_BUS_mn_1 = df_BUS_mn_1.reset_index(drop = True)

        for idx, rowval in df_BUS_mn_1.iterrows(): ### iterate over each row
            featarr = np.ones((5, 5))
            feat_diffdict = {}
            feature_count = 0

            cmaqval = rowval['cmaq_id']
            row, col = np.where(rast_us == cmaqval) ### get (row,col) where cmaq_id value is same as raster pixel value.  IF_STATEMENT_1
            if(len(row) != 0 and len(col) != 0): ## make sure the row,col values are not empty.
                cmaqgrid = np.array(rast_us[row[0] - N_:row[0]+(N_+1), col[0]- N_:col[0]+(N_+1)])

                if cmaqgrid.shape[0] == 5 and cmaqgrid.shape[1] == 5: ### What if this is not true?? <----- Look into this Shrey.   IF_STATEMENT_2
                    for colname in df_BUS_mn_1: ###iterate over every column and for every column you create the cmaqgrid and then add corresponding column values
                        # print(rowval['pm25_value'], rowval[colname])
                        if colname == 'cmaq_id': ##or colname == 'cmaq_x' or colname == 'cmaq_y' or colname == 'day' or colname == 'month' or colname == 'rid'
                            continue
                        elif colname == 'pm25_value':
                            targetval = rowval['pm25_value']
                            continue
                        else:
                            feature_count = feature_count + 1
                            temparr = np.ones((5, 5))
                            for i in range(5): ## Add values element-wise
                                for j in range(5):
                                    if i == 2 and j == 2:
                                        # print(rowval[colname])
                                        temparr[i,j] = rowval[colname]
                                    else:
                                        match_cmaq = cmaqgrid[i,j]
                                        if match_cmaq == 0: ## compare this to 0 || previously: if np.isnan(match_cmaq):
                                            temparr[i,j] = math.nan   ##match_cmaq

                                        else:
                                            locval_df = df_BUS_pd_1[df_BUS_pd_1['cmaq_id'] == match_cmaq]
                                            locval = locval_df[colname]
                                            if len(locval) == 0: ### If no cmaq_id matches are found
                                                locval = math.nan ##assigning the column value at the location as nan (previously 0.0)

                                            temparr[i,j] = locval

                        ##making sure empty values are averaged.
                        temparr = np.where(np.isnan(temparr), ma.array(temparr, mask=np.isnan(temparr)).mean(axis=0), temparr)
                        featarr = np.dstack((featarr,temparr))


                else: ##### ELSE_STATEMENT_2
                    continue

            else: ##### ELSE_STATEMENT_1
                continue


            featarr = np.dstack((featarr,temparr))
    #         print("No. of features added: ",feature_count)

            if featarr.shape[0] == 5 and featarr.shape[1] == 5 and featarr.shape[2] == 29:
                featarr = np.delete(featarr, (0), axis = 2) ##delete first
                featarr = np.delete(featarr, (27), axis = 2) ##delete the last which is an extra addition.
                print(featarr.shape)

                targetlist.append(targetval)
                traininglist.append(featarr)

    return traininglist, targetlist ### return the final list where the data is appended


def std_scale(df_to_std):
    from sklearn.preprocessing import StandardScaler

    ########## split into features to be standardized and the ones not to be standardizes.
    df_to_std_noscale = df_to_std[['day', 'month', 'cmaq_x', 'cmaq_y', 'cmaq_id', 'rid', 'pm25_value', 'pm25_value_k']]
    df_to_std_scale = df_to_std[['elev', 'forest_cover', 'pd', 'local', 'limi', 'high', 'is', 'nldas_pevapsfc', 'nldas_pressfc', 'nldas_cape',
                       'nldas_ugrd10m', 'nldas_vgrd10m', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_dlwrfsfc', 'nldas_dswrfsfc', 'nldas_pcpsfc',
                       'nldas_fpcsfc', 'gc_aod', 'aod_value', 'emissi11_pm25']]

    ########### Run the standard scaler.

    scale_cols = ['elev', 'forest_cover', 'pd', 'local', 'limi', 'high', 'is', 'nldas_pevapsfc', 'nldas_pressfc', 'nldas_cape',
    'nldas_ugrd10m', 'nldas_vgrd10m', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_dlwrfsfc', 'nldas_dswrfsfc', 'nldas_pcpsfc',
    'nldas_fpcsfc', 'gc_aod', 'aod_value', 'emissi11_pm25']

    ss = StandardScaler()
    df_to_std_scale[scale_cols] = ss.fit_transform(df_to_std_scale[scale_cols])

    ####### concatenate the 'standardized + non-standardized'
    df_to_std_fs = pd.concat([df_to_std_scale, df_to_std_noscale], axis=1)
    df_to_std_fs = df_to_std_fs.reset_index(drop = True)

    return(df_to_std_fs)


def main():
    # df_BUS = pd.read_csv("US_data/BigUS/largeUS_pred.csv")
    df_BUS = pd.read_csv("US_data/BigUS/largeUS_pred.csv.gz", compression='gzip')
    df_BUS = df_BUS.drop(['Unnamed: 0'], axis = 1)
    df_BUS = df_BUS[['day', 'month', 'cmaq_x', 'cmaq_y', 'cmaq_id', 'rid', 'elev', 'forest_cover', 'pd', 'local', 'limi', 'high', 'is', 'nldas_pevapsfc',
                 'nldas_pressfc', 'nldas_cape', 'nldas_ugrd10m', 'nldas_vgrd10m', 'nldas_tmp2m', 'nldas_rh2m', 'nldas_dlwrfsfc', 'nldas_dswrfsfc',
                 'nldas_pcpsfc', 'nldas_fpcsfc', 'gc_aod', 'aod_value', 'emissi11_pm25', 'pm25_value_k', 'pm25_value']] ####


    # df_BUS_std = std_scale(df_BUS)

    # ##### create a bigger dataset which has not null values from pm 2.5 (this is monitoring locations dataset)
    # df_BUS_monitoring = df_BUS_std[df_BUS_std.pm25_value.notnull()]
    # df_BUS_predictions = df_BUS_std[~df_BUS_std.pm25_value.notnull()]

    ##### create a bigger dataset which has not null values from pm 2.5 (this is monitoring locations dataset) [when features are not standardized]
    df_BUS_monitoring = df_BUS[df_BUS.pm25_value.notnull()]
    df_BUS_predictions = df_BUS[~df_BUS.pm25_value.notnull()]


    ###### reader the raster file for the US
    import rasterio as rt

    rast_data = rt.open('US_data/BigUS/test.tif')

    rast_us = rast_data.read(1)
    rast_us[rast_us == rast_data.nodata] = 0

    traininglist, targetlist = create_grid(df_BUS_monitoring, df_BUS_predictions, rast_us)

    trainingArr = np.array(traininglist)
    targetArr = np.array(targetlist)

    ftrain = open('US_data/BigUS/training_grid_v3.npy','wb')
    np.save(ftrain, trainingArr)

    ftest = open('US_data/BigUS/target_grid_v3.npy','wb')
    np.save(ftest, targetArr)


if __name__ == "__main__":
    main()

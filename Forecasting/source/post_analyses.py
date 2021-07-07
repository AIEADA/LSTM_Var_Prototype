import os
import xarray as xr 
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(dir_path)

import numpy as np
np.random.seed(10)
import matplotlib.pyplot as plt
from utils import plot_averaged_errors, plot_windowed_errors, plot_contours, plot_bars


def perform_analyses(var_time,num_ips,num_ops,test_fields,snapshots_mean,pod_modes,forecast,save_path,subregions):

    lead_time = num_ops
    test_fields = test_fields.reshape(103,120,-1)[:,:,:var_time+num_ips+num_ops]
    snapshots_mean = snapshots_mean.reshape(103,120)

    persistence_maes = np.zeros(shape=(num_ops,len(subregions)),dtype='float32')
    predicted_maes = np.zeros(shape=(num_ops,len(subregions)),dtype='float32')

    # For different lead times
    for lead_time in range(num_ops):
        # Predicted test
        pred_test = forecast[:var_time,lead_time,:]

        # Global analyses
        # Reconstruct
        predicted = snapshots_mean[:,:,None] + np.matmul(pod_modes,pred_test.T).reshape(103,120,-1)

        # persistence predictions
        persistence_fields = test_fields[:,:,num_ips-(lead_time+1):num_ips-(lead_time+1)+var_time]

        # Post analyses - unify time slices
        test_fields_temp = test_fields[:,:,num_ips+lead_time:num_ips+lead_time+var_time]

        # Local analysis
        region_num = 0
        for region in subregions:
            mask = np.asarray(xr.open_dataset(region)['mask'])

            pred_local = predicted[mask==1,:]
            pers_local = persistence_fields[mask==1,:]
            test_fields_local = test_fields_temp[mask==1,:]

            mae = np.mean(np.abs(pers_local-test_fields_local))
            persistence_maes[lead_time,region_num] = mae

            mae = np.mean(np.abs(pred_local-test_fields_local))
            predicted_maes[lead_time,region_num] = mae

            region_num+=1

        if lead_time == num_ops-1:
            # Visualizations
            pred_mae, pred_cos = plot_averaged_errors(test_fields_temp,predicted,snapshots_mean)
            pers_mae, pers_cos = plot_averaged_errors(test_fields_temp,persistence_fields,snapshots_mean)

            plot_contours(pers_mae-pred_mae,-10,10,'Difference MAE',save_path+'/Difference_MAE.png')
            plot_contours(pred_cos-pers_cos,-0.5,0.5,'Difference Cosine Similarity',save_path+'/Difference_COS.png')

            # # For the specific days
            # pred_mae, pred_cos = plot_windowed_errors(test_fields,predicted,snapshots_mean,int_start=120,int_end=150)
            # pers_mae, pers_cos = plot_windowed_errors(test_fields,persistence_fields,snapshots_mean,int_start=120,int_end=150)

            # plot_contours(pers_mae-pred_mae,-10,10,'Difference MAE',save_path+'/Difference_MAE_Windowed.png')
            # plot_contours(pred_cos-pers_cos,-0.5,0.5,'Difference Cosine Similarity',save_path+'/Difference_COS_Windowed.png')


    # Save RMSE predictions
    np.savetxt(save_path+'/persistence_maes.txt',persistence_maes)
    np.savetxt(save_path+'/predicted_maes.txt',predicted_maes)

    # Make a plot of them
    plot_bars(persistence_maes,predicted_maes,subregions,save_path)




if __name__ == '__main__':
    print('Analysis module')
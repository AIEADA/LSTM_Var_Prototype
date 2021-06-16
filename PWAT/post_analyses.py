import numpy as np
np.random.seed(10)
import matplotlib.pyplot as plt
import os
from lstm_archs import num_ips, num_ops
from utils import plot_averaged_errors, plot_windowed_errors

lead_time = num_ops
var_time = 200
era5 = True

if era5:
    test_fields = np.load('ERA5_Testing_snapshots.npy').reshape(103,120,-1)[:,:,:var_time+num_ips+num_ops]
else:
    test_fields = np.load('Testing_snapshots.npy').reshape(103,120,-1)[:,:,:var_time+num_ips+num_ops]

snapshots_mean = np.load('Training_mean.npy').reshape(103,120)

# Load POD modes
pod_modes = np.load('POD_Modes.npy')[:,:20]

for lead_time in range(num_ops):
    # Predicted test
    pred_test = np.load('./Regular/Predicted.npy')[:var_time,lead_time,:]
    # Var assimilated test
    pred_var_test = np.load('./Var/Predicted.npy')[:var_time,lead_time,:]

    # Reconstruct
    predicted = snapshots_mean[:,:,None] + np.matmul(pod_modes,pred_test.T).reshape(103,120,-1)
    predicted_var = snapshots_mean[:,:,None] + np.matmul(pod_modes,pred_var_test.T).reshape(103,120,-1)

    # persistence predictions
    persistence_fields = test_fields[:,:,num_ips-(lead_time+1):-(num_ops+lead_time+1)]

    # Post analyses - unify time slices
    test_fields_temp = test_fields[:,:,num_ips+lead_time:-(num_ops-lead_time)]

    rmse = np.sqrt(np.mean((persistence_fields-test_fields_temp)**2))
    print('Persistence RMSE:',rmse,' for lead time of', lead_time+1)

    rmse = np.sqrt(np.mean((predicted-test_fields_temp)**2))
    print('Prediction RMSE:',rmse,' for lead time of', lead_time+1)

    rmse = np.sqrt(np.mean((predicted_var-test_fields_temp)**2))
    print('Var Prediction RMSE:',rmse,' for lead time of', lead_time+1)

    # # Visualizations
    # plot_averaged_errors(test_fields,predicted,snapshots_mean)
    # plot_averaged_errors(test_fields,persistence_fields,snapshots_mean)

    # # For the specific days
    # plot_windowed_errors(test_fields,predicted,snapshots_mean,int_start=120,int_end=150)
    # plot_windowed_errors(test_fields,persistence_fields,snapshots_mean,int_start=120,int_end=150)
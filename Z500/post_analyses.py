import numpy as np
np.random.seed(10)
import matplotlib.pyplot as plt
import os
from lstm_archs import num_ips, num_ops
from utils import plot_averaged_errors, plot_windowed_errors

test_fields = np.load('Testing_snapshots.npy').reshape(103,120,-1)
snapshots_mean = np.load('Mean.npy').reshape(103,120)

lead_time = num_ops

# Load POD modes
pod_modes = np.load('Modes.npy')[:,:20]

# True test
true_test = np.load('./Regular/True.npy')[:,lead_time-1,:]
# Predicted test
pred_test = np.load('./Regular/Predicted.npy')[:,lead_time-1,:]
# Var assimilated test
# pred_var_test = np.load('./Var/Predicted.npy')[:,-1,:]

# Reconstruct
true_pod = snapshots_mean[:,:,None] + np.matmul(pod_modes,true_test.T).reshape(103,120,-1)
predicted = snapshots_mean[:,:,None] + np.matmul(pod_modes,pred_test.T).reshape(103,120,-1)
# var = snapshots_mean[:,:,None] + np.matmul(pod_modes,pred_var_test.T).reshape(103,120,-1)

# persistence predictions
persistence_fields = test_fields[:,:,num_ips:-lead_time]

# Post analyses - unify time slices
test_fields = test_fields[:,:,num_ips+lead_time:]

# # For all time steps
# plot_averaged_errors(test_fields,predicted,snapshots_mean)
# plot_averaged_errors(test_fields,persistence_fields,snapshots_mean)

# For the first 60 days of each year in testing
plot_windowed_errors(test_fields,predicted,snapshots_mean,int_start=120,int_end=150)
plot_windowed_errors(test_fields,persistence_fields,snapshots_mean,int_start=120,int_end=150)
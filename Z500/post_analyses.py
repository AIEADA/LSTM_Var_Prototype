import numpy as np
np.random.seed(10)
import matplotlib.pyplot as plt
import os
from lstm_archs import num_ips, num_ops
from utils import plot_averaged_errors

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

# Post analyses - unify time slices
test_fields = test_fields[:,:,num_ips+lead_time:]

plot_averaged_errors(test_fields,predicted,snapshots_mean)

# pred_loc = 2500 #10, 2500, 1500 is good, 100, 150, 200, 3200 is meh

# plt.figure()
# plt.imshow(test_fields[:,:,pred_loc],vmin=4500,vmax=6000)
# plt.colorbar()
# plt.title('True')

# plt.figure()
# plt.imshow(true_pod[:,:,pred_loc],vmin=4500,vmax=6000)
# plt.colorbar()
# plt.title('Projected')

# plt.figure()
# plt.imshow(predicted[:,:,pred_loc],vmin=4500,vmax=6000)
# plt.colorbar()
# plt.title('Predicted')
# plt.show()
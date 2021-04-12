import numpy as np
np.random.seed(10)
import matplotlib.pyplot as plt
import os
from lstm_archs import num_ips, num_ops

test_fields = np.load('Testing_snapshots.npy').reshape(103,120,-1)
snapshots_mean = np.load('Mean.npy').reshape(103,120)

lead_time = 0

# Load POD modes
pod_modes = np.load('Modes.npy')[:,:20]

# True test
true_test = np.load('./Regular/True.npy')[:,lead_time,:]
pred_test = np.load('./Regular/Predicted.npy')[:,lead_time,:]
# pred_var_test = np.load('./Var/Predicted.npy')[:,-1,:]


# Reconstruct
true_pod = snapshots_mean[:,:,None] + np.matmul(pod_modes,true_test.T).reshape(103,120,-1)
predicted = snapshots_mean[:,:,None] + np.matmul(pod_modes,pred_test.T).reshape(103,120,-1)
# var = snapshots_mean[:,:,None] + np.matmul(pod_modes,pred_var_test.T).reshape(103,120,-1)

pred_loc = 200 #10, 2500, 1500 is good, 100, 150, 200, 3200 is meh

plt.figure()
plt.imshow(test_fields[:,:,pred_loc+num_ips+lead_time],vmin=4500,vmax=6000)
plt.colorbar()
plt.title('True')

plt.figure()
plt.imshow(true_pod[:,:,pred_loc],vmin=4500,vmax=6000)
plt.colorbar()
plt.title('Projected')

plt.figure()
plt.imshow(predicted[:,:,pred_loc],vmin=4500,vmax=6000)
plt.colorbar()
plt.title('Predicted')
plt.show()

# plt.figure()
# plt.imshow(predicted[:,:,pred_loc]-test_fields[:,:,pred_loc+num_ips+lead_time],vmin=-500,vmax=500)
# plt.colorbar()
# plt.title('Difference regular')

# plt.figure()
# plt.imshow(predicted[:,:,pred_loc]-true_pod[:,:,pred_loc],vmin=-500,vmax=500)
# plt.colorbar()
# plt.title('Difference projected')
# plt.show()

# plt.figure()
# plt.imshow(var[:,:,pred_loc],vmin=4500,vmax=6000)
# plt.colorbar()
# plt.title('VAR (day 14)')
# plt.show()

# plt.figure()
# plt.imshow(var[:,:,pred_loc]-test_fields[:,:,pred_loc+14],vmin=-400,vmax=500)
# plt.colorbar()
# plt.title('Difference Var')
# plt.show()





import numpy as np
np.random.seed(10)
import matplotlib.pyplot as plt
import os

# Load mask
mask = np.asarray(np.load('mask',allow_pickle=True).data,dtype='bool')[0].flatten()
test_fields = np.load('Testing_snapshots.npy',allow_pickle=True).data.reshape(1487,-1)
snapshots_mean = np.load('Training_mean.npy')

# Remove mean and land points
test_fields = test_fields - snapshots_mean[:,None]
test_fields = test_fields[:500,mask]
# Load POD modes
pod_modes = np.load('POD_Modes.npy')

# Just emulator
lstm = np.load('./Regular/Predicted.npy')[:500]
pred_field = np.matmul(pod_modes,lstm.T)
pred_diff = np.sum((test_fields-pred_field.T)**2,axis=1)/np.shape(pred_field)[0]

# Data assimilated
file_list = os.listdir('./Varying_Observations/')
var_diff_list = []
for file in file_list:
    if '.npy' in file:
        filename = os.path.join('./Varying_Observations/',file)
        lstm_var = np.load(filename)[:500]

        # Reconstruct
        var_field = np.matmul(pod_modes,lstm_var.T)
        # Difference
        var_diff = np.sum((test_fields-var_field.T)**2,axis=1)/np.shape(var_field)[0]

        var_diff_list.append(var_diff)

plt_num = 0
for i in range(len(file_list)):
    file = file_list[i]
    if '.npy' in file:
        save_name = './Varying_Observations/'+file_list[i][:-4]+'.png'

        plt.figure()
        plt.plot(pred_diff,label='Difference regular')
        plt.plot(var_diff_list[plt_num],label=file_list[i])
        plt.legend()
        plt.ylabel('MSE')
        plt.xlabel('Test snapshot')
        plt.savefig(save_name)
        plt.close()

        plt_num+=1
import numpy as np
np.random.seed(10)
import matplotlib.pyplot as plt
import os

# Load mask
mask = np.asarray(np.load('mask',allow_pickle=True).data,dtype='bool')[0].flatten()
test_fields = np.load('test_fields.npy',allow_pickle=True).data.reshape(1487,-1)
# Remove mean and land points
test_fields = test_fields - np.mean(test_fields,axis=0)
test_fields = test_fields[:300,mask]
# Load POD modes
pod_modes = np.load('POD_Modes.npy')

# Just emulator
lstm = np.load('./Regular/Predicted.npy')[:300]
pred_field = np.matmul(pod_modes,lstm.T)
pred_diff = np.sum((test_fields-pred_field.T)**2,axis=1)/np.shape(pred_field)[0]

# Data assimilated
file_list = os.listdir('./Varying_Observations/')
var_diff_list = []
for file in file_list:
    filename = os.path.join('./Varying_Observations/',file)
    lstm_var = np.load(filename)[:300]

    # Reconstruct
    var_field = np.matmul(pod_modes,lstm_var.T)
    # Difference
    var_diff = np.sum((test_fields-var_field.T)**2,axis=1)/np.shape(var_field)[0]

    var_diff_list.append(var_diff)




for i in range(len(file_list)):
    save_name = './Varying_Observations/'+file_list[i][:-4]+'.png'

    plt.figure()
    plt.plot(pred_diff,label='Difference regular')
    plt.plot(var_diff_list[i],label=file_list[i])
    plt.legend()
    plt.xlabel('MSE')
    plt.ylabel('Test snapshot')
    plt.savefig(save_name)
    plt.close()
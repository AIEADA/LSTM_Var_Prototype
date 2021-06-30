import numpy as np
np.random.seed(10)
import matplotlib.pyplot as plt
import os
from utils import plot_averaged_errors, plot_windowed_errors

test_fields = np.load('Testing_snapshots.npy').reshape(103,120,-1)
snapshots_mean = np.load('Mean.npy').reshape(103,120)

# Load POD modes
pod_modes = np.load('Modes.npy')[:,:5]

# Coefficients
coeffs = np.load('./Test_Coefficients.npy')[:5,:]

# Var assimilated test
# pred_var_test = np.load('./Var/Predicted.npy')[:,-1,:]

# Reconstruct
reconstructed = snapshots_mean[:,:,None] + np.matmul(pod_modes,coeffs).reshape(103,120,-1)

# For all time steps
plot_averaged_errors(test_fields,reconstructed,snapshots_mean)

# For the specific days
# plot_windowed_errors(test_fields,reconstructed,snapshots_mean,int_start=120,int_end=150)
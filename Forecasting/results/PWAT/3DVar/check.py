import numpy as np
import matplotlib.pyplot as plt

true = np.load('True.npy')
true_norm = np.load('True_norm.npy')
predicted = np.load('Predicted.npy')
predicted_norm = np.load('Predicted_norm.npy')

plt.figure()
plt.plot(true[:20,-1,-1],label='True')
plt.plot(true_norm[:20,-1,-1],label='True regular')
plt.plot(predicted[:20,-1,-1],label='Var')
plt.plot(predicted_norm[:20,-1,-1],label='REgular')
plt.legend()
plt.show()
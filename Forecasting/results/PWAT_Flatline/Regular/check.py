import numpy as np
import matplotlib.pyplot as plt

true_data = np.load('True.npy')
pred_data = np.load('Predicted.npy')

plt.figure()
plt.plot(true_data[0,:,0],label='True')
plt.plot(pred_data[0,:,0],label='Predicted')
plt.legend()
plt.title('1 prediction over time')
plt.show()


plt.figure()
plt.plot(true_data[:365,-1,0],label='True')
plt.plot(pred_data[:365,-1,0],label='Predicted')
plt.legend()
plt.title('Last day of all predictions')
plt.show()
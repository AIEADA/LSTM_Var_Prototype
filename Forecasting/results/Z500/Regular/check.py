import numpy as np
import matplotlib.pyplot as plt

true = np.load('True.npy')
predicted = np.load('Predicted.npy')

print(true.shape)

plt.figure()
plt.plot(true[:365,0,0],label='True')
plt.plot(predicted[:365,0,0],label='Predicted')
plt.legend()
plt.show()
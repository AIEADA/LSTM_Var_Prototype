import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

scores = np.load('Scores.npy')[:,0,:,:]
dim_1 = scores.shape[0]
dim_2 = scores.shape[1]
dim_3 = scores.shape[2]

scores = scores.reshape(dim_1,dim_2*dim_3)

scaler = MinMaxScaler()
scores = scaler.fit_transform(scores)

scores = scores.reshape(dim_1,dim_2,dim_3)

average_1 = np.mean(scores,axis=0)

plt.figure()
plt.imshow(average_1)
plt.colorbar()
plt.show()
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import glob
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Dropout, Input
from tensorflow.keras import optimizers, regularizers, layers
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt

year_list = np.arange(1980,2000,step=1)
filename_list = []
for year in year_list:
    filename_list.append('Z500_'+str(year)+'_coarse.nc')

filename_list.remove('Z500_1983_coarse.nc')

filenum = 0
for file in filename_list:
    data = xr.open_dataset(file)
    extracted = np.asarray(data['z500']).reshape(-1,103*120).T
    
    if filenum == 0:
        data_matrix = extracted
        filenum+=1
    else:
        data_matrix = np.concatenate((data_matrix,extracted),axis=-1)

training_data = data_matrix[:,:6*365]
testing_data = data_matrix[:,6*365:]

# np.save('Training_snapshots.npy',training_data)
# np.save('Testing_snapshots.npy',testing_data)

data_mean = np.mean(training_data,axis=-1)
# np.save('Training_mean.npy',data_mean)

# Find fluctuations
training_fluc = training_data - data_mean[:,None]
testing_fluc = testing_data - data_mean[:,None]
training_fluc=np.transpose(training_fluc)
testing_fluc = np.transpose(testing_fluc)
print("#####################")
print((testing_fluc.shape))
print(training_fluc.shape)


encoding_dim1 = 8092
encoding_dim2 = 4096
encoding_dim3 = 2048
encoding_dim4 = 1024
encoding_dim5 = 512
encoding_dim6 = 256
final_encoded_dim = 80

input_data = keras.Input(shape=training_fluc.shape[1])
# "encoded" is the encoded representation of the input
encoded = layers.Dense(encoding_dim1, activation='relu')(input_data)
encoded = layers.Dense(encoding_dim2, activation='relu')(encoded)
encoded = layers.Dense(encoding_dim3, activation='relu')(encoded)
encoded = layers.Dense(encoding_dim4, activation='relu')(encoded)
encoded = layers.Dense(encoding_dim5, activation='relu')(encoded)
encoded = layers.Dense(encoding_dim6, activation='relu')(encoded)
encoded = layers.Dense(final_encoded_dim, activation='relu')(encoded)



# "decoded" is the lossy reconstruction of the input
decoded = layers.Dense(encoding_dim6, activation='relu')(encoded)
decoded = layers.Dense(encoding_dim5, activation='relu')(decoded)
decoded = layers.Dense(encoding_dim4, activation='relu')(decoded)
decoded = layers.Dense(encoding_dim3, activation='relu')(decoded)
decoded = layers.Dense(encoding_dim2, activation='relu')(decoded)
decoded = layers.Dense(encoding_dim1, activation='relu')(decoded)
decoded = layers.Dense(training_fluc.shape[1], activation='linear')(decoded)


# This model maps an input to its reconstruction
autoencoder = keras.Model(input_data, decoded)

# encoder part
encoder = keras.Model(input_data, encoded)
encoded_input = keras.Input(shape=(final_encoded_dim,))

# decoder part
# Retrieve the last layer of the autoencoder model
#decoder_layer = autoencoder.layers[-1]
# Create the decoder model
#decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
decoder_layer = autoencoder.layers[-7]
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

print(autoencoder.summary())
print(encoder.summary())
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(training_fluc, training_fluc,
                epochs=10,
                batch_size=32)

encoded_data = encoder.predict(testing_fluc)
decoded_data = decoder.predict(encoded_data)





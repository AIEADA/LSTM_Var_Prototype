import numpy as np
np.random.seed(10)
import tensorflow as tf
tf.random.set_seed(10)
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping, ModelCheckpoint
import glob
import pickle
import os
import time



def read_npy_file(file_path):
    file_path_string = bytes.decode(file_path.numpy())
    data = np.load(file_path_string)
    return data.astype(np.float32)


def parse_function(file_path):
    #example = tf.py_function(load_file_npy, inp=file_path, Tout=tf.float32)
    example = tf.py_function(read_npy_file, [file_path], tf.float32)
    # example has shape 21, 121, 281
    
    #example = tf.py_function(np.load, file_path, tf.string)
    #tf.print(file_path)
    #example = tf.io.read_file(file_path)
    #example = tf.io.decode_raw(example, tf.float32)
    #print(example)
    input_window = 14
    input_data = example[:input_window,:,:]

    # NOTE: when add LSTM part, switch output to this commented out line
    #output_data = example[input_window:,:,:]
    #return input_data, output_data
    return tf.expand_dims(input_data, axis=3)

def build_dataset(batch_size, directory):
    dataset_dir = directory + 'split_examples/train_data_z500_2d/'
    filelist = glob.glob(dataset_dir + '*.npy')
    train_dataset = tf.data.Dataset.from_tensor_slices(filelist)
    shuffle_buffer = 6000
    train_dataset = train_dataset.shuffle(shuffle_buffer,reshuffle_each_iteration=True)
    train_dataset = train_dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return train_dataset

def get_data(batch_size):
    # In physical space use 'raw_train.reshape(-1,121,281)' to visualize original
    #num_snapshots = 12564
    #directory = '/lcrc/project/AIEADA-2/era5_data/full_data/snapshots/'
    directory = '/lus/grand/projects/datascience/bethanyl/AIEADA/snapshots/'

    #var_list = [raw_train_z500,
    #            raw_train_u250,raw_train_v250,raw_train_t250,
    #            raw_train_u850,raw_train_v850,raw_train_t850,
    #            raw_train_blh,raw_train_tcwv]


    #example = np.load(directory + 'split_examples/train_data_z500_2d/example_151.npy')
    train_dataset = build_dataset(batch_size, directory)
    return train_dataset

def autoencoder1():
    encode_dim = 180
    ae_encoding_layers = []
    # TimeDistributed will apply the same layer to each snapshot in time (same weights)
    # https://keras.io/api/layers/recurrent_layers/time_distributed/
    ae_encoding_layers.append(layers.TimeDistributed(layers.Conv2D(filters=50, kernel_size=(3,3), strides=(1,1),activation='relu',padding='same'))) # (None, 14, 122, 282, 50)
    ae_encoding_layers.append(layers.TimeDistributed(layers.MaxPooling2D((2,2),padding='same'))) # (None, 14, 61, 141, 50)
    ae_encoding_layers.append(layers.TimeDistributed(layers.Conv2D(filters=25, kernel_size=(3,3), strides=(1,1),activation='relu',padding='same'))) # (None, 14, 62, 142, 25)
    ae_encoding_layers.append(layers.TimeDistributed(layers.MaxPooling2D((2,2),padding='same'))) # (None, 14, 31, 71, 25)
    ae_encoding_layers.append(layers.TimeDistributed(layers.ZeroPadding2D(padding=((1,0),(1,0)))))
    ae_encoding_layers.append(layers.TimeDistributed(layers.Conv2D(filters=12, kernel_size=(3,3), strides=(1,1),activation='relu',padding='same'))) # (None, 14, 31, 71, 12)
    ae_encoding_layers.append(layers.TimeDistributed(layers.MaxPooling2D((2,3),padding='same'))) # (None, 14, 16, 24, 12)
    ae_encoding_layers.append(layers.TimeDistributed(layers.Conv2D(filters=6, kernel_size=(3,3), strides=(1,1),activation='relu',padding='same'))) # (None, 14, 16, 24, 6) 
    ae_encoding_layers.append(layers.TimeDistributed(layers.MaxPooling2D((2,3),padding='same'))) # [None, 14, 8, 8, 6]
    ae_encoding_layers.append(layers.TimeDistributed(layers.Flatten())) # [None, 14, 384]
    ae_encoding_layers.append(layers.TimeDistributed(layers.Dense(encode_dim, activation=None))) # [None, 14, 180]

        
    # decoder for reconstruction
    ae_decoding_layers = []
    ae_decoding_layers.append(layers.TimeDistributed(layers.Dense(384, activation='relu'))) # (None, 14, 384) 
    ae_decoding_layers.append(layers.TimeDistributed(layers.Reshape((8, 8, 6)))) #  (None, 14, 8, 8, 6)
    ae_decoding_layers.append(layers.TimeDistributed(layers.Conv2D(filters=6, kernel_size=(3,3), strides=(1,1),activation='relu',padding='same'))) # (None, 14, 8, 8, 6)
    ae_decoding_layers.append(layers.TimeDistributed(layers.UpSampling2D((2,3)))) # (None, 14, 16, 24, 6)
    ae_decoding_layers.append(layers.TimeDistributed(layers.Conv2D(filters=12, kernel_size=(3,3), strides=(1,1),activation='relu',padding='same'))) # (None, 14, 16, 24, 12)
    ae_decoding_layers.append(layers.TimeDistributed(layers.UpSampling2D((2,3)))) # (None, 14, 32, 72, 12) 
    ae_decoding_layers.append(layers.TimeDistributed(layers.Cropping2D(cropping=((1, 0), (1, 0))))) # (None, 14, 31, 71, 12) 
    ae_decoding_layers.append(layers.TimeDistributed(layers.Conv2D(filters=25, kernel_size=(3,3), strides=(1,1),activation='relu',padding='same'))) # (None, 14, 31, 71, 25) 
    ae_decoding_layers.append(layers.TimeDistributed(layers.UpSampling2D((2,2)))) # (None, 14, 62, 142, 25) 
    ae_decoding_layers.append(layers.TimeDistributed(layers.Conv2D(filters=50, kernel_size=(3,3), strides=(1,1),activation='relu',padding='same'))) # (None, 14, 62, 142, 50)
    ae_decoding_layers.append(layers.TimeDistributed(layers.Cropping2D(cropping=((1, 0), (1, 0))))) # (None, 14, 61, 141, 50) 
    ae_decoding_layers.append(layers.TimeDistributed(layers.UpSampling2D((2,2)))) #  (None, 14, 122, 282, 50)
    ae_decoding_layers.append(layers.TimeDistributed(layers.Cropping2D(cropping=((1, 0), (1, 0))))) #  (None, 14, 121, 281, 50)
    ae_decoding_layers.append(layers.TimeDistributed(layers.Conv2D(filters=1,kernel_size=(1,1),activation=None,padding='same'))) # (None, 14, 121, 281, 1) 
    return ae_encoding_layers, ae_decoding_layers

def autoencoder2():
    encode_dim = 5
    ae_encoding_layers = []
    # TimeDistributed will apply the same layer to each snapshot in time (same weights)
    # https://keras.io/api/layers/recurrent_layers/time_distributed/
    ae_encoding_layers.append(layers.TimeDistributed(layers.Conv2D(filters=50, kernel_size=(3,3), strides=(1,1),activation='relu',padding='same'))) # (None, 14, 122, 282, 50)
    ae_encoding_layers.append(layers.TimeDistributed(layers.MaxPooling2D((2,2),padding='same'))) # (None, 14, 61, 141, 50)
    ae_encoding_layers.append(layers.TimeDistributed(layers.Conv2D(filters=25, kernel_size=(3,3), strides=(1,1),activation='relu',padding='same'))) # (None, 14, 62, 142, 25)
    ae_encoding_layers.append(layers.TimeDistributed(layers.MaxPooling2D((2,2),padding='same'))) # (None, 14, 31, 71, 25)
    ae_encoding_layers.append(layers.TimeDistributed(layers.ZeroPadding2D(padding=((1,0),(1,0)))))
    ae_encoding_layers.append(layers.TimeDistributed(layers.Conv2D(filters=12, kernel_size=(3,3), strides=(1,1),activation='relu',padding='same'))) # (None, 14, 31, 71, 12)
    ae_encoding_layers.append(layers.TimeDistributed(layers.MaxPooling2D((2,3),padding='same'))) # (None, 14, 16, 24, 12)
    ae_encoding_layers.append(layers.TimeDistributed(layers.Conv2D(filters=6, kernel_size=(3,3), strides=(1,1),activation='relu',padding='same'))) # (None, 14, 16, 24, 6) 
    ae_encoding_layers.append(layers.TimeDistributed(layers.MaxPooling2D((2,3),padding='same'))) # [None, 14, 8, 8, 6]
    ae_encoding_layers.append(layers.TimeDistributed(layers.Conv2D(filters=3, kernel_size=(3,3), strides=(1,1),activation='relu',padding='same'))) # [None, 14, 8, 8, 3]
    ae_encoding_layers.append(layers.TimeDistributed(layers.MaxPooling2D((2,2),padding='same'))) # [None, 14, 4, 4, 3]
    ae_encoding_layers.append(layers.TimeDistributed(layers.Flatten())) # [None, 14, 48]
    ae_encoding_layers.append(layers.TimeDistributed(layers.Dense(encode_dim, activation=None))) # [None, 14, 5]

        
    # decoder for reconstruction
    ae_decoding_layers = []
    ae_decoding_layers.append(layers.TimeDistributed(layers.Dense(48, activation='relu'))) # (None, 14, 48) 
    ae_decoding_layers.append(layers.TimeDistributed(layers.Reshape((4, 4, 3)))) #  (None, 14, 4, 4, 3)
    ae_decoding_layers.append(layers.TimeDistributed(layers.Conv2D(filters=3, kernel_size=(3,3), strides=(1,1),activation='relu',padding='same'))) # (None, 14, 4, 4, 3)
    ae_decoding_layers.append(layers.TimeDistributed(layers.UpSampling2D((2,2)))) # (None, 14, 8, 8, 3)
    ae_decoding_layers.append(layers.TimeDistributed(layers.Conv2D(filters=6, kernel_size=(3,3), strides=(1,1),activation='relu',padding='same'))) # (None, 14, 8, 8, 6)
    ae_decoding_layers.append(layers.TimeDistributed(layers.UpSampling2D((2,3)))) # (None, 14, 16, 24, 6)
    ae_decoding_layers.append(layers.TimeDistributed(layers.Conv2D(filters=12, kernel_size=(3,3), strides=(1,1),activation='relu',padding='same'))) # (None, 14, 16, 24, 12)
    ae_decoding_layers.append(layers.TimeDistributed(layers.UpSampling2D((2,3)))) # (None, 14, 32, 72, 12) 
    ae_decoding_layers.append(layers.TimeDistributed(layers.Cropping2D(cropping=((1, 0), (1, 0))))) # (None, 14, 31, 71, 12) 
    ae_decoding_layers.append(layers.TimeDistributed(layers.Conv2D(filters=25, kernel_size=(3,3), strides=(1,1),activation='relu',padding='same'))) # (None, 14, 31, 71, 25) 
    ae_decoding_layers.append(layers.TimeDistributed(layers.UpSampling2D((2,2)))) # (None, 14, 62, 142, 25) 
    ae_decoding_layers.append(layers.TimeDistributed(layers.Conv2D(filters=50, kernel_size=(3,3), strides=(1,1),activation='relu',padding='same'))) # (None, 14, 62, 142, 50)
    ae_decoding_layers.append(layers.TimeDistributed(layers.Cropping2D(cropping=((1, 0), (1, 0))))) # (None, 14, 61, 141, 50) 
    ae_decoding_layers.append(layers.TimeDistributed(layers.UpSampling2D((2,2)))) #  (None, 14, 122, 282, 50)
    ae_decoding_layers.append(layers.TimeDistributed(layers.Cropping2D(cropping=((1, 0), (1, 0))))) #  (None, 14, 121, 281, 50)
    ae_decoding_layers.append(layers.TimeDistributed(layers.Conv2D(filters=1,kernel_size=(1,1),activation=None,padding='same'))) # (None, 14, 121, 281, 1) 
    return ae_encoding_layers, ae_decoding_layers

class Autoencoder(tf.keras.models.Model):
    def __init__(self, model_name):
        tf.keras.models.Model.__init__(self)
        lat = 121
        long = 281
        input_window = 14
        #output_window = 7
        #self.inputs = layers.Input(shape=(input_window,lat,long,1)) # (None, 14, 121, 281)
        if model_name == 'autoencoder1':
            ae_encoding_layers, ae_decoding_layers = autoencoder1()
        elif model_name == 'autoencoder2':
            ae_encoding_layers, ae_decoding_layers = autoencoder2()

        self.ae_encoding_layers = ae_encoding_layers
        self.ae_decoding_layers = ae_decoding_layers

        # NOTE: commenting out this section for now to try to just get an autoencoder
        # decoder for prediction    
        #lstm_decoding_layers = []
        #lstm_decoding_layers.append(layers.RepeatVector(output_window))
        #lstm_decoding_layers.append(layers.LSTM(100,activation='relu', return_sequences=True))
        #lstm_decoding_layers.append(layers.TimeDistributed(layers.Dense(embed_dim)))
        # Encode from physical space
        #print('Input shape:',self.inputs.get_shape().as_list())

    @tf.function
    def call(self, inputs):
        x = inputs
        num_ae_encoder_layers = len(self.ae_encoding_layers)
        for i in range(num_ae_encoder_layers):
            x = self.ae_encoding_layers[i](x)
        encoded = x

        #print('AE Encoded shape:',encoded.get_shape().as_list())

        #preds = encoded
        #num_lstm_layers = len(lstm_layers)
        #for i in range(num_lstm_layers):
        #    preds = lstm_layers[i](preds)

        #print('LSTM prediction shape:',preds.get_shape().as_list())
            
        recon = encoded
        num_ae_decoder_layers = len(self.ae_decoding_layers)
        for i in range(num_ae_decoder_layers):
            recon = self.ae_decoding_layers[i](recon)
            
        #print('AE Output shape:',recon.get_shape().as_list())
        return recon


def load_and_evaluate(filename, model_name, batch_size):
    train_dataset = get_data(batch_size)
    model = define_model(model_name)
    model.load_weights(filename) #'weights.35.hdf5')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='mean_squared_error')
    train_loss = model.evaluate(train_dataset, verbose=2)
    print("Restored model, training loss: {:5.2f}%" % train_loss)


def training(train_dataset, model, learning_rate, num_epochs, folder_name):

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()

    # Refer to the CTL (custom training loop guide)
    @tf.function
    def train_step(x):
        with tf.GradientTape() as tape:
            recon = model(x)
            loss = loss_fn(x, recon)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    losses = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        avg_loss = 0
        steps = 0
        start_epoch = time.time()
        for _, data in enumerate(train_dataset):
            loss = train_step(data)
            avg_loss += loss
            steps += 1
        end_epoch = time.time()
        losses[epoch] = avg_loss/steps
        print("avg loss at epoch %d: %.2f, %.2f seconds" % (epoch, avg_loss, end_epoch-start_epoch))
        model.save_weights(filepath='%s/weights%d.hdf5' % (folder_name, epoch))
    np.save('%s/losses.npy' % folder_name, losses)

def main_custom_training(folder_name="exp1", model_name='autoencoder1', num_epochs=50, batch_size=32):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    train_dataset = get_data(batch_size)
    model = Autoencoder(model_name)
    training(train_dataset, model, 0.001, num_epochs, folder_name)

def main(folder_name="exp1", model_name='autoencoder1', num_epochs=50, batch_size=32):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    train_dataset = get_data(batch_size)
    model = Autoencoder(model_name)

    checkpointing = ModelCheckpoint(filepath='%s/weights.{epoch:02d}.hdf5' % folder_name, save_weights_only=True, save_freq='epoch')
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    #                            patience=5, min_lr=0.0001)
    #early_stop = EarlyStopping(monitor='val_loss',patience=20)

    #model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss='mean_squared_error',loss_weights=[1,1])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='mean_squared_error')
    model.summary()
    history = model.fit(train_dataset,epochs=num_epochs,callbacks=[checkpointing], verbose=2)
    with open('%s/history.pkl' % folder_name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

if __name__ == "__main__":
    main_custom_training(folder_name="exp3", model_name='autoencoder2', num_epochs=2, batch_size=16)

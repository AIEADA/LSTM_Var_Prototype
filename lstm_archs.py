import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

import tensorflow as tf
tf.random.set_seed(10)

from tensorflow.keras import Model
import numpy as np
np.random.seed(10)

from utils import coeff_determination
# preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Plotting
import matplotlib.pyplot as plt

# 3D Var
from scipy.optimize import minimize

#Build the model which does basic map of inputs to coefficients
class standard_lstm(Model):
    def __init__(self,data,var=False):
        super(standard_lstm, self).__init__()

        # Inference with 3D var?
        self.var = var
        self.num_obs = 3000

        # Set up the data for the LSTM
        self.data_tsteps = np.shape(data)[0]
        self.state_len = np.shape(data)[1]

        self.preproc_pipeline = Pipeline([('stdscaler', StandardScaler()),('minmax', MinMaxScaler(feature_range=(-1, 1)))])
        self.data = self.preproc_pipeline.fit_transform(data)

        # Need to make minibatches
        self.seq_num = 7
        self.total_size = np.shape(data)[0]-int(self.seq_num) # Limit of sampling

        input_seq = np.zeros(shape=(self.total_size,self.seq_num,self.state_len))  #[samples,n_inputs,state_len]
        output_seq = np.zeros(shape=(self.total_size,self.state_len)) #[samples,n_outputs,state_len]

        snum = 0
        for t in range(0,self.total_size):
            input_seq[snum,:,:] = self.data[None,t:t+self.seq_num,:]
            output_seq[snum,:] = self.data[None,t+self.seq_num,:]        
            snum = snum + 1

        # Shuffle dataset
        idx = np.arange(snum)
        np.random.shuffle(idx)
        input_seq = input_seq[idx]
        output_seq = output_seq[idx]

        # Split into train and test
        self.input_seq_test = input_seq[int(0.9*snum):]
        self.output_seq_test = output_seq[int(0.9*snum):]
        input_seq = input_seq[:int(0.9*snum)]
        output_seq = output_seq[:int(0.9*snum)]

        # Split into train and valid
        self.ntrain = int(0.8*np.shape(input_seq)[0])
        self.nvalid = np.shape(input_seq)[0] - self.ntrain

        self.input_seq_train = input_seq[:self.ntrain]
        self.output_seq_train = output_seq[:self.ntrain]

        self.input_seq_valid = input_seq[self.ntrain:]
        self.output_seq_valid = output_seq[self.ntrain:]

        # Define architecture
        xavier=tf.keras.initializers.GlorotUniform()
        self.l1=tf.keras.layers.LSTM(50,return_sequences=True,input_shape=(self.seq_num,self.state_len))
        self.l2=tf.keras.layers.LSTM(50,return_sequences=False)
        self.out = tf.keras.layers.Dense(self.state_len)
        self.train_op = tf.keras.optimizers.Adam(learning_rate=0.001)

    
    # Running the model
    def call(self,X):
        h1 = self.l1(X)
        h2 = self.l2(h1)
        out = self.out(h2)
        return out
    
    # Regular MSE
    def get_loss(self,X,Y):
        op=self.call(X)
        return tf.reduce_mean(tf.math.square(op-Y))

    # get gradients - regular
    def get_grad(self,X,Y):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(X,Y)
            g = tape.gradient(L, self.trainable_variables)
        return g
    
    # perform gradient descent - regular
    def network_learn(self,X,Y):
        g = self.get_grad(X,Y)
        self.train_op.apply_gradients(zip(g, self.trainable_variables))

    # Train the model
    def train_model(self):
        plot_iter = 0
        stop_iter = 0
        patience = 10
        best_valid_loss = 999999.0 # Some large number 

        self.num_batches = 20
        self.train_batch_size = int(self.ntrain/self.num_batches)
        self.valid_batch_size = int((self.nvalid)/self.num_batches)
        
        for i in range(100):
            # Training loss
            print('Training iteration:',i)
            
            for batch in range(self.num_batches):
                input_batch = self.input_seq_train[batch*self.train_batch_size:(batch+1)*self.train_batch_size]
                output_batch = self.output_seq_train[batch*self.train_batch_size:(batch+1)*self.train_batch_size]
                self.network_learn(input_batch,output_batch)

            # Validation loss
            valid_loss = 0.0
            valid_r2 = 0.0

            for batch in range(self.num_batches):
                input_batch = self.input_seq_valid[batch*self.valid_batch_size:(batch+1)*self.valid_batch_size]
                output_batch = self.output_seq_valid[batch*self.valid_batch_size:(batch+1)*self.valid_batch_size]

                valid_loss = valid_loss + self.get_loss(input_batch,output_batch).numpy()
                predictions = self.call(self.input_seq_valid)
                valid_r2 = valid_r2 + coeff_determination(predictions,self.output_seq_valid)

            valid_r2 = valid_r2/(batch+1)


            # Check early stopping criteria
            if valid_loss < best_valid_loss:
                
                print('Improved validation loss from:',best_valid_loss,' to:', valid_loss)
                print('Validation R2:',valid_r2)
                
                best_valid_loss = valid_loss

                self.save_weights('./checkpoints/my_checkpoint')
                
                stop_iter = 0
            else:
                print('Validation loss (no improvement):',valid_loss)
                print('Validation R2:',valid_r2)
                stop_iter = stop_iter + 1

            if stop_iter == patience:
                break
                
        # Check accuracy on test
        predictions = self.call(self.input_seq_test)
        print('Test loss:',self.get_loss(self.input_seq_test,self.output_seq_test).numpy())
        r2 = coeff_determination(predictions,self.output_seq_test)
        print('Test R2:',r2)
        r2_iter = 0

    # Load weights
    def restore_model(self):
        self.load_weights(dir_path+'/checkpoints/my_checkpoint') # Load pretrained model

    def regular_inference(self,rec_input_seq,rec_pred):
        # Regular predict
        for t in range(rec_pred.shape[0]):
            rec_pred[t] = self.call(rec_input_seq).numpy()[0]
            rec_input_seq[0,0:-1] = rec_input_seq[0,1:]
            rec_input_seq[0,-1] = rec_pred[t]

        # Rescale
        rec_pred = self.preproc_pipeline.inverse_transform(rec_pred)

        return rec_pred

    def variational_inference(self,rec_input_seq,rec_pred):
        # Load mask
        mask = np.asarray(np.load('mask',allow_pickle=True).data,dtype='bool')[0].flatten()
        test_fields = np.load('test_fields.npy',allow_pickle=True).data.reshape(1487,-1)
        # Remove mean and land points
        test_fields = test_fields - np.mean(test_fields,axis=0)
        test_fields = test_fields[:,mask]
        # Load POD modes
        pod_modes = np.load('POD_Modes.npy')
        
        # Random observation locations
        rand_idx = np.arange(test_fields.shape[1])
        np.random.shuffle(rand_idx)
        rand_idx = rand_idx[:self.num_obs]

        true_observations = test_fields[:,rand_idx]

        # 3D-Variational update
        for t in range(300):

            # Background vector - initial time window input
            x_ti = self.preproc_pipeline.inverse_transform(rec_input_seq[0,:].reshape(self.seq_num,-1))
            x_ti_rec = np.matmul(pod_modes,x_ti.T)#[:,0:1]

            # Observation
            y_ = true_observations[t+self.seq_num]

            # Perform optimization assuming identity covariances
            # Define residual
            def residual(x):
                # Prior
                x = x.reshape(self.seq_num,-1)
                xphys = self.preproc_pipeline.inverse_transform(x)
                x_star_rec = np.matmul(pod_modes,xphys.T)#[:,0:1]

                # Likelihood
                x = x.reshape(1,self.seq_num,-1)
                x_tf = self.preproc_pipeline.inverse_transform(self.call(x).numpy()[0].reshape(1,-1))
                x_tf_rec = np.matmul(pod_modes,x_tf.T)

                # Sensor predictions
                h_ = x_tf_rec[rand_idx,0]

                # J
                pred = np.sum(0.5*(x_star_rec - x_ti_rec)**2) + np.sum(0.5*(y_-h_)**2)
                
                return pred

            # Define gradient of residual
            def residual_gradient(x):
                # Prior
                x = x.reshape(self.seq_num,-1).astype('double')
                xphys = self.preproc_pipeline.inverse_transform(x)
                tf_x_star_rec = tf.convert_to_tensor(np.matmul(pod_modes,xphys.T),dtype='float64')#[:,0:1]
                tf_x_ti_rec = tf.convert_to_tensor(x_ti_rec,dtype='float64')
                tf_y_ = tf.convert_to_tensor(y_,dtype='float64')

                # Likelihood
                x = x.reshape(1,self.seq_num,-1)
                x = tf.convert_to_tensor(x,dtype='float64')
                tf_pod_modes = tf.convert_to_tensor(pod_modes,dtype='float64')

                std_scaler = self.preproc_pipeline.get_params()['steps'][0][1]
                minmax_scaler = self.preproc_pipeline.get_params()['steps'][1][1]

                with tf.GradientTape(persistent=True) as t:
                    t.watch(x)
                    
                    op = self.call(x)
                    op = (op+1)/2.0*(minmax_scaler.data_max_- minmax_scaler.data_min_) + minmax_scaler.data_min_
                    op = (op)*std_scaler.scale_ + std_scaler.mean_
                    op = tf.cast(op,dtype='float64')

                    x_tf_rec = tf.matmul(tf_pod_modes,tf.transpose(op))[:,0]

                    # Sensor predictions
                    tf_idx = tf.convert_to_tensor(rand_idx,dtype='int32')
                    h_ = tf.gather(x_tf_rec,tf_idx)

                    # J
                    pred = tf.math.reduce_sum(0.5*(tf_x_star_rec - tf_x_ti_rec)**2) + tf.math.reduce_sum(0.5*(tf_y_-h_)**2)
                 
                return t.gradient(pred, x).numpy()[0,:,:].flatten().astype('double')

            solution = minimize(residual,rec_input_seq[0].flatten(),jac=residual_gradient, method='CG',
                tol=1e-8,options={'disp': True, 'maxiter': 100, 'eps': 1.4901161193847656e-8})

            assimilated_rec_input_seq = solution.x.reshape(1,self.seq_num,-1)
            rec_pred[t] = self.call(assimilated_rec_input_seq).numpy()[0].reshape(1,-1)

            rec_input_seq[0,0:-1,:] = assimilated_rec_input_seq[0,1:,:]
            rec_input_seq[0,-1,:] = rec_pred[t]

            print('Finished variational prediction for timestep: ',t)

        rec_pred = self.preproc_pipeline.inverse_transform(rec_pred)

        return rec_pred

    # Do some testing
    def model_inference(self,test_data):
        # Restore from checkpoint
        self.restore_model()

        # Scale testing data
        test_data = self.preproc_pipeline.transform(test_data)

        # Test sizes
        test_total_size = np.shape(test_data)[0]-int(self.seq_num) # Limit of sampling

        # Initial condition setup
        rec_input_seq = test_data[:self.seq_num,:].reshape(1,self.seq_num,self.state_len)
        
        # True outputs
        rec_output_seq = np.zeros(shape=(test_total_size,self.state_len)) #[samples,n_outputs,state_len]
        snum = 0
        for t in range(test_total_size):
            rec_output_seq[snum,:] = test_data[None,t+self.seq_num,:]        
            snum = snum + 1

        # Prediction recursively
        rec_pred = np.copy(rec_output_seq)

        if self.var:
            rec_pred = self.variational_inference(rec_input_seq,rec_pred)
        else:
            rec_pred = self.regular_inference(rec_input_seq,rec_pred)

        rec_output_seq = self.preproc_pipeline.inverse_transform(rec_output_seq)

        if self.var:
            folder_name = './Var/'
        else:
            folder_name = './Regular/'

        np.save(folder_name+'True.npy',rec_output_seq)
        np.save(folder_name+'Predicted.npy',rec_pred)

        # plot
        for i in range(self.state_len):
            plt.figure()
            plt.title('Mode '+str(i))
            plt.plot(rec_pred[:,i],label='Predicted')
            plt.plot(rec_output_seq[:,i],label='True')
            plt.legend()
            plt.savefig(folder_name+'Mode '+str(i)+'.png')
            plt.close()

        return rec_output_seq, rec_pred


if __name__ == '__main__':
    print('Architecture file')

import os, yaml, sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

# Load YAML file for configuration
config_file = open('config.yaml')
configuration = yaml.load(config_file,Loader=yaml.FullLoader)

data_paths = configuration['data_paths']
subregion_paths = data_paths['subregions']
operation_mode = configuration['operation_mode']
hyperparameters = configuration.get('hyperparameters')

config_file.close()

# Location for test results
if not os.path.exists(data_paths['save_path']):
    os.makedirs(data_paths['save_path'])

if __name__ == '__main__':

    import numpy as np
    np.random.seed(10)
    from lstm_archs import standard_lstm

    # Loading data
    train_data = np.load(data_paths['training_coefficients']).T

    # Initialize model
    lstm_model = standard_lstm(train_data,data_paths['save_path'],hyperparameters)
    
    # Training model
    if operation_mode['train']:
        lstm_model.train_model()

    # Regular testing of model
    if operation_mode['test']:
        
        test_data = np.load(data_paths['testing_coefficients']).T
        true, forecast = lstm_model.regular_inference(test_data)

        if not os.path.exists(data_paths['save_path']+'/Regular/'):
            os.makedirs(data_paths['save_path']+'/Regular/')    
        np.save(data_paths['save_path']+'/Regular/True.npy',true)
        np.save(data_paths['save_path']+'/Regular/Predicted.npy',forecast)

    # 3DVar testing of model
    if operation_mode['perform_var']:
        
        test_data = np.load(data_paths['testing_coefficients']).T
        train_fields = np.load(data_paths['training_fields']).T
        test_fields = np.load(data_paths['da_testing_fields']).T
        pod_modes = np.load(data_paths['pod_modes'])[:,:20]
        training_mean = np.load(data_paths['training_mean'])

        true, forecast = lstm_model.variational_inference(test_data,train_fields,test_fields,pod_modes,training_mean)

        if not os.path.exists(data_paths['save_path']+'3DVar/'):
            os.makedirs(data_paths['save_path']+'3DVar/')
        np.save(data_paths['save_path']+'/3DVar/True.npy',true)
        np.save(data_paths['save_path']+'/3DVar/Predicted.npy',forecast)


    if operation_mode['perform_analyses']:

        from post_analyses import perform_analyses
       
        pod_modes = np.load(data_paths['pod_modes'])[:,:20]
        training_mean = np.load(data_paths['training_mean'])

        var_time = hyperparameters[0]
        num_inputs = hyperparameters[1]
        num_outputs = hyperparameters[2]

        if os.path.isfile(data_paths['save_path']+'/Regular/Predicted.npy'):
            forecast = np.load(data_paths['save_path']+'/Regular/Predicted.npy')
            test_fields = np.load(data_paths['testing_fields'])
            perform_analyses(var_time,num_inputs,num_outputs,
                            test_fields,training_mean,pod_modes,forecast,
                            data_paths['save_path']+'/Regular/',subregion_paths)
        else:
            print('No forecast for the test data. Skipping analyses.')
        
        if os.path.isfile(data_paths['save_path']+'/3DVar/Predicted.npy'):
            forecast = np.load(data_paths['save_path']+'/3DVar/Predicted.npy')
            test_fields = np.load(data_paths['da_testing_fields'])
            perform_analyses(var_time,num_inputs,num_outputs,
                            test_fields,training_mean,pod_modes,forecast,
                            data_paths['save_path']+'/3DVar/',subregion_paths)
        else:
            print('No forecast for the test data with 3D Var. Skipping analyses.')


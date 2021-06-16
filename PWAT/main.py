import numpy as np
np.random.seed(10)

from lstm_archs import standard_lstm

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='LSTM training of time-series data')
    parser.add_argument('--train', help='Do training', action='store_true') #Train a network or use a trained network for inference
    parser.add_argument('--test', help='Do testing', action='store_true') #Train a network or use a trained network for inference
    parser.add_argument('--var',help='Use 3DVar Assimilation',action='store_true')
    parser.add_argument('--era5',help='Use 3DVar ERA5',action='store_true')

    args = parser.parse_args()

    # Loading data
    data = np.load('Training_Coefficients.npy').T

    # Initialize model
    lstm_model = standard_lstm(data,flags=[args.var,args.era5])
    # Training model
    if args.train:
        lstm_model.train_model() # Train and exit
    
    elif args.test:
        if args.era5:
            data = np.load('ERA5_Testing_Coefficients.npy').T
        else:
            data = np.load('Testing_Coefficients.npy').T
        
        true, predicted = lstm_model.model_inference(data) # Do some inference


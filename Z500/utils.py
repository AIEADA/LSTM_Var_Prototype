import matplotlib.pyplot as plt
import numpy as np
np.random.seed(10)
from scipy import spatial
from scipy.stats import pearsonr

# Coefficient of determination
def coeff_determination(y_pred, y_true): #Order of function inputs is important here        
    SS_res =  np.sum(np.square( y_true-y_pred )) 
    SS_tot = np.sum(np.square( y_true - np.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + 2.22044604925e-16) )

def plot_contours(field,title='Figure'):
    plt.figure()
    plt.title(title)
    plt.imshow(field)
    plt.colorbar()
    plt.show()

def cosine_plot(snapshots_true,snapshots_pred,int_start,int_end):
    # dimensions
    dim1 = snapshots_pred.shape[0]
    dim2 = snapshots_pred.shape[1]

    # Finding the cosine similarity
    pred_vals = snapshots_pred[:,:,int_start:int_end].reshape(-1,int_end-int_start)
    true_vals = snapshots_true[:,:,int_start:int_end].reshape(-1,int_end-int_start)
    
    # For the provided prediction
    pred_cos_vals = np.zeros(shape=(pred_vals.shape[0]))
        
    for location in range(true_vals.shape[0]):
        pred_cos_vals[location] = 1-spatial.distance.cosine(pred_vals[location,:],true_vals[location,:])
            
    return pred_cos_vals.reshape(dim1,dim2)
    
def correlation_plot(snapshots_true,snapshots_pred,int_start,int_end):
    # dimensions
    dim1 = snapshots_pred.shape[0]
    dim2 = snapshots_pred.shape[1]

    # Finding the Pearson correlation (without time lag)    
    pred_vals = snapshots_pred[:,:,int_start:int_end].reshape(-1,int_end-int_start)
    true_vals = snapshots_true[:,:,int_start:int_end].reshape(-1,int_end-int_start)
    
    # For the provided prediction
    pred_r_vals = np.zeros(shape=(pred_vals.shape[0]))
        
    for location in range(true_vals.shape[0]):
        pred_r_vals[location], _ = pearsonr(pred_vals[location,:],true_vals[location,:])
        
    return pred_r_vals.reshape(dim1,dim2)

def plot_averaged_errors(true_fields, pred_fields, snapshots_mean):
    mae_fields = np.mean(np.abs(pred_fields - true_fields),axis=-1)
    rmse_fields = np.sqrt(np.mean((pred_fields - true_fields)**2,axis=-1))

    plot_contours(mae_fields,'Mean absolute error')
    plot_contours(rmse_fields,'Root Mean squared error')

    r_plot = correlation_plot(true_fields,pred_fields,int_start=0,int_end=3*365)

    true_fluc = true_fields - snapshots_mean[:,:,None]
    pred_fluc = pred_fields - snapshots_mean[:,:,None]

    cos_plot = cosine_plot(true_fluc,pred_fluc,int_start=0,int_end=3*365)

    plot_contours(r_plot,'Pearson R plot')
    plot_contours(cos_plot,'Cosine similarity plot')

    

if __name__ == '__main__':
    print('This is the utilities file')
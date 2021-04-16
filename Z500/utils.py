import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import numpy as np
np.random.seed(10)
from scipy import spatial
from scipy.stats import pearsonr
import matplotlib.ticker as mticker
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm 
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmaps

# Coefficient of determination
def coeff_determination(y_pred, y_true): #Order of function inputs is important here        
    SS_res =  np.sum(np.square( y_true-y_pred )) 
    SS_tot = np.sum(np.square( y_true - np.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + 2.22044604925e-16) )

def plot_contours(field,title='Figure'):
    lon = np.load('./xlong.npy')
    lat = np.load('./xlat.npy')

    # Set your data coordinates
    datacrs = ccrs.Mercator() 
    # Set the map projection
    mapcrs = ccrs.Mercator()

    plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    #ax.set_global()
    ax.set_extent([-179, -35,24, 77], crs=ccrs.PlateCarree())
    ax.coastlines()

    ax.add_feature(cfeature.STATES, edgecolor='0.2')  

    mycmap=cmaps.MPL_jet  
    rr = ax.pcolormesh(lon[0:102,5:119], lat[0:102,5:119], field[0:102,5:119],shading='auto',cmap=mycmap)#,norm=mynorm)  #gist_stern afmhot didn't use transform, but looks ok...

    cbar = plt.colorbar(rr, orientation='horizontal', pad=.1,shrink=1, aspect=28,extend='max')#,ticks=[250,255,260,265,270,275,280,285,290,295]) #,,,ticks=[-5,-4,-3,-2,-1,0,1,2,3,4,5]ticks=[0,5,10,15,20]label='number of days'),ticks=[0,.1,.2,.3,.4,.5,.6]
    cbar.set_label(label=title, fontsize=16)
    cbar.ax.tick_params(labelsize=16) 

    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'lakes', '50m', edgecolor='0.6', facecolor='white'))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
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
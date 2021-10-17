import os, yaml
import xarray as xr 
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(dir_path)

import numpy as np
np.random.seed(10)
import matplotlib.pyplot as plt
from utils import plot_averaged_errors, plot_windowed_errors, plot_contours, plot_bars


# Load YAML file for configuration - unique for each rank
config_file = open('config.yaml')
configuration = yaml.load(config_file,Loader=yaml.FullLoader)

data_paths = configuration['data_paths']
subregion_paths = data_paths['subregions']
operation_mode = configuration['operation_mode']
model_choice = operation_mode['model_choice']
hyperparameters = configuration.get('hyperparameters')
num_modes = hyperparameters[6]
num_ips = hyperparameters[1]
num_ops = hyperparameters[2]
var_time = hyperparameters[4]
cadence = hyperparameters[8]
output_gap = hyperparameters[10]
config_file.close()

# Save path
forecast = np.load('./Regular/Predicted.npy')
save_path = './Regular/'
plot_day = 10


pod_modes = np.load(parent_path+'/'+data_paths['pod_modes'])[:,:num_modes]
snapshots_mean = np.load(parent_path+'/'+data_paths['training_mean'])
test_fields = np.load(parent_path+'/'+data_paths['testing_fields'])

lead_time = num_ops
test_fields = test_fields.reshape(103,120,-1)[:,:,:var_time+num_ips+num_ops+output_gap]
snapshots_mean = snapshots_mean.reshape(103,120)

persistence_maes = np.zeros(shape=(num_ops,len(subregion_paths)+1),dtype='float32')
climatology_maes = np.zeros(shape=(num_ops,len(subregion_paths)+1),dtype='float32')
predicted_maes = np.zeros(shape=(num_ops,len(subregion_paths)+1),dtype='float32')

# Climatology calculation
train_fields = np.load(parent_path+'/'+data_paths['training_fields']).T
yearly_snaps = int(365*cadence)
num_years = train_fields.shape[0]//yearly_snaps

climatology = train_fields[:yearly_snaps]
for year in range(1,num_years):
    climatology = climatology + train_fields[year*yearly_snaps:(year+1)*yearly_snaps]
climatology = climatology/num_years
climatology = climatology.T.reshape(103,120,-1)

if var_time+num_ips+num_ops+output_gap > yearly_snaps:
    climatology_lead = climatology[:,:,:var_time+num_ips+num_ops]
    climatology_trail = climatology[:,:,var_time+num_ips+num_ops:]
    climatology = np.concatenate((climatology_trail,climatology_lead),axis=-1)

    # Num snaps predicted 
    num_snaps_pred = test_fields.shape[-1]
    num_years_pred = num_snaps_pred//yearly_snaps
    climatology = np.tile(climatology,(1,1,num_years_pred))

    # Fix trailing dimension
    if climatology.shape[-1] != test_fields.shape[-1]:
        tile_diff = abs(climatology.shape[-1]-test_fields.shape[-1])
        climatology = np.concatenate((climatology,climatology[:,:,:tile_diff]),axis=-1)
else:
    climatology = climatology[:,:,:var_time+num_ips+num_ops+output_gap]        

# For different lead times - output gap has been removed here
for lead_time in range(num_ops):
    # Predicted test
    pred_test = forecast[:var_time,lead_time,:]

    # Global analyses
    # Reconstruct
    predicted = snapshots_mean[:,:,None] + np.matmul(pod_modes,pred_test.T).reshape(103,120,-1)

    # persistence predictions
    persistence_fields = test_fields[:,:,num_ips-(lead_time+1):num_ips-(lead_time+1)+var_time]

    # Post analyses - unify time slices
    test_fields_temp = test_fields[:,:,output_gap+num_ips+lead_time:output_gap+num_ips+lead_time+var_time]

    # Climatology predictions
    clim_fields = climatology[:,:,output_gap+num_ips+lead_time:output_gap+num_ips+lead_time+var_time]

    # Local analysis
    region_num = 0
    for region in subregion_paths:
        mask = np.asarray(xr.open_dataset(parent_path+'/'+region)['mask'])

        pred_local = predicted[mask==1,:]
        pers_local = persistence_fields[mask==1,:]
        clim_local = clim_fields[mask==1,:]
        test_fields_local = test_fields_temp[mask==1,:]

        mae = np.mean(np.abs(pers_local-test_fields_local))
        persistence_maes[lead_time,region_num] = mae

        mae = np.mean(np.abs(pred_local-test_fields_local))
        predicted_maes[lead_time,region_num] = mae

        mae = np.mean(np.abs(clim_local-test_fields_local))
        climatology_maes[lead_time,region_num] = mae

        region_num+=1

    # Total
    region_num = -1
    mae = np.mean(np.abs(persistence_fields-test_fields_temp))
    persistence_maes[lead_time,region_num] = mae

    mae = np.mean(np.abs(predicted-test_fields_temp))
    predicted_maes[lead_time,region_num] = mae

    mae = np.mean(np.abs(clim_fields-test_fields_temp))
    climatology_maes[lead_time,region_num] = mae


    if lead_time == plot_day:
        # Visualizations
        pred_mae, pred_cos = plot_averaged_errors(test_fields_temp,predicted,snapshots_mean)
        pers_mae, pers_cos = plot_averaged_errors(test_fields_temp,persistence_fields,snapshots_mean)
        clim_mae, clim_cos = plot_averaged_errors(test_fields_temp,clim_fields,snapshots_mean)

        plot_contours(pred_mae,0,150,'MAE',save_path+'/MAE_Pred.png')
        plot_contours(pred_cos,-1.0,1.0,'Cosine Similarity',save_path+'/COS_Pred.png')


        plot_contours(clim_mae-pred_mae,-10,10,'Difference MAE',save_path+'/Difference_MAE_Clim.png')
        plot_contours(pred_cos-clim_cos,-0.15,0.15,'Difference Cosine Similarity',save_path+'/Difference_COS_Clim.png')

        plot_contours(pers_mae-pred_mae,-100,100,'Difference MAE',save_path+'/Difference_MAE_Pers.png')
        plot_contours(pred_cos-pers_cos,-0.5,0.5,'Difference Cosine Similarity',save_path+'/Difference_COS_Pers.png')

        # # For the specific days
        # pred_mae, pred_cos = plot_windowed_errors(test_fields,predicted,snapshots_mean,int_start=120,int_end=150)
        # pers_mae, pers_cos = plot_windowed_errors(test_fields,persistence_fields,snapshots_mean,int_start=120,int_end=150)

        # plot_contours(pers_mae-pred_mae,-10,10,'Difference MAE',save_path+'/Difference_MAE_Windowed.png')
        # plot_contours(pred_cos-pers_cos,-0.5,0.5,'Difference Cosine Similarity',save_path+'/Difference_COS_Windowed.png')


    # Save RMSE predictions
    np.savetxt(save_path+'/persistence_maes.txt',persistence_maes)
    np.savetxt(save_path+'/predicted_maes.txt',predicted_maes)
    np.savetxt(save_path+'/climatology_maes.txt',climatology_maes)

# Make a plot of them
plot_bars(persistence_maes[:,:-1],climatology_maes[:,:-1],predicted_maes[:,:-1],subregion_paths,save_path)

iter_num = -1 # All regions
fname = 'everything'
plt.figure()
plt.title('MAE for '+fname)
plt.plot(persistence_maes[:,iter_num],label='Persistence')
plt.plot(climatology_maes[:,iter_num],label='Climatology')

try:
    reg_maes = np.loadtxt('./Regular/predicted_maes.txt')
    plt.plot(reg_maes[:,iter_num],label='Regular')
except:
    print('Regular analysis not done')

try:
    var_maes = np.loadtxt('./3DVar/predicted_maes.txt')
    plt.plot(var_maes[:,iter_num],label='3DVar')
except:
    print('Var analysis not done')


plt.legend()
plt.xlabel('Timesteps')
plt.ylabel('MAE')
plt.savefig(fname+'.png')
plt.close()
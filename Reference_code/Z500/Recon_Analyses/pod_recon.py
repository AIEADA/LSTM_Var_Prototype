import numpy as np
np.random.seed(10)
import numpy.linalg as LA
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Generate POD basis
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def generate_pod_bases(snapshot_matrix,snapshot_matrix_test): #Mean removed
    '''
    Takes input of a snapshot matrix and computes POD bases
    Outputs truncated POD bases and coefficients
    '''

    snapshot_matrix_mean = np.mean(snapshot_matrix,axis=1)
    snapshot_matrix = snapshot_matrix-snapshot_matrix_mean[:,None]
    snapshot_matrix_test = snapshot_matrix_test-snapshot_matrix_mean[:,None]

    new_mat = np.matmul(np.transpose(snapshot_matrix),snapshot_matrix)

    w,v = LA.eig(new_mat)

    # Bases
    phi = np.real(np.matmul(snapshot_matrix,v))
    trange = np.arange(np.shape(snapshot_matrix)[1])
    phi[:,trange] = phi[:,trange]/np.sqrt(w[:])

    coefficient_matrix = np.matmul(np.transpose(phi),snapshot_matrix)
    coefficient_matrix_test = np.matmul(np.transpose(phi),snapshot_matrix_test)

    return phi, coefficient_matrix, coefficient_matrix_test, snapshot_matrix_mean

if __name__ == '__main__':
    
    training_snapshots = np.load('Training_snapshots.npy')
    testing_snapshots = np.load('Testing_snapshots.npy')
    # pod_modes, cf_truncs, cf_truncs_test, mean_value = generate_pod_bases(training_snapshots,testing_snapshots)
    # np.save('Modes.npy',pod_modes)
    # np.save('Train_Coefficients.npy',cf_truncs)
    # np.save('Test_Coefficients.npy',cf_truncs_test)
    # np.save('Mean.npy',mean_value)

    # exit()

    pod_modes = np.load('Modes.npy')
    cf_truncs = np.load('Train_Coefficients.npy')
    cf_truncs_test = np.load('Test_Coefficients.npy')
    mean_value = np.load('Mean.npy')

    num_modes = 5
    reconstruction = mean_value[:,None] + np.matmul(pod_modes[:,:num_modes],cf_truncs_test[:num_modes,:])
    reconstruction = reconstruction.reshape(103,120,-1)

    np.save('reconstruction.npy',reconstruction)
    exit()

    testing_snapshots = testing_snapshots.reshape(103,120,-1)

    snapshot_num = 500

    plt.figure()
    plt.title('True')
    plt.imshow(testing_snapshots[:,:,snapshot_num],vmin=4500,vmax=6000)
    plt.colorbar()


    plt.figure()
    plt.title('Rec')
    plt.imshow(reconstruction[:,:,snapshot_num],vmin=4500,vmax=6000)
    plt.colorbar()

    plt.figure()
    plt.title('Diff')
    plt.imshow(testing_snapshots[:,:,snapshot_num]-reconstruction[:,:,snapshot_num])
    plt.colorbar()
    plt.show()
    







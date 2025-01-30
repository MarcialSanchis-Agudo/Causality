import pyLOM
import datetime
import numpy as np
import os
import torch
import h5py
import time
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import pyvista as pv
import mpi4py
mpi4py.rc.recv_mprobe = False
pv.start_xvfb()
BASEDIR  = '/mimer/NOBACKUP/groups/kthmech/carlos/Datasets_3D/Obstacle_VAE/'
CASESTR  = 'obstacle_3D_28'
DATAFILE  = os.path.join(BASEDIR, f'{CASESTR}.h5')
VARIABLES = ["VELOX", "VELOY", "VELOZ"]
RESULTS_DIR = "3D_obstacle"
print('FILE', DATAFILE)
mesh = pyLOM.Mesh.load(DATAFILE)
dataset = pyLOM.Dataset.load(DATAFILE, ptable=mesh.partition_table)
time = dataset.get_variable('time')
print('DATA')
def preprocess_variable(var_name):
    var_data = dataset[var_name]
    print('MEAN')
    # Compute the temporal mean along the time axis
    var_mean = pyLOM.math.temporal_mean(var_data)  # shape (262144,)
    
    # Subtract the mean, expanding it for proper broadcasting
    mean_subtracted = pyLOM.math.subtract_mean(var_data, var_mean)  # shape (262144, 3999)
    print('STD')
    # Compute the temporal standard deviation
    std_dev = np.std(var_data, axis=1, keepdims=True)  # shape (262144, 1)
    
    # Normalize the data
    normalized_data = mean_subtracted / std_dev  # shape (262144, 3999)
    
    return normalized_data, std_dev, var_mean
def Energy_Rec(truth,pred):
    
    """
    Compute the reconstruction enery E_K using fluctuation components

    Args:
        truth   : NumPy Array with shape [N,H,W] The fluctuation componets of ground truth

        pred    : NumPy Array with shape [N,H,W] The fluctuation componets of ground pred

    Returns:

        Ek      : Energy level of reconstruction
    """
    import numpy as np 

    pred, truth = pred.squeeze(), truth.squeeze()
    err         = np.sum( (pred-truth)**2 ,axis = (0,1))/np.sum((truth)**2,axis = (0,1))
    Ek          = (1 -err) * 100 

    return Ek
# Function to extract latent space
def extract_latent_space(vae_model, dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    latent_vectors = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(vae_model.device)
            _, latent_space = vae_model(batch)
            latent_vectors.append(latent_space.cpu().numpy())

    latent_vectors = np.concatenate(latent_vectors, axis=0)
    return latent_vectors
# Save latent space to .h5 file
def save_latent_space_to_h5(latent_vectors, output_file):
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('latent_space', data=latent_vectors)
    print(f"Latent space saved to {output_file}")
def Rank_SpatialMode(model, latent_dim, u_truth, u_std , modes, device):
    """
    Rank the non-linear modes in latent-space according to the energy content

    Args: 
        model           : PyTorch nn.Module as beta-VAE
        lantent_dim     : (Int) The latent-space dimension
        u_truth         : Normalised the streamwise velocity 
        u_std           : Std of the streamwise velocity 
        modes           : The reparameterized mode for decoder     
        device          : The device for running model

    Returns:
        Ranks           : The ranks for non-linear modes 
        Ecum            : Ecumlative energy for each rank
    """
    

    import numpy as np 
    import torch 
    from torch.utils.data import DataLoader, TensorDataset

    print(f"The modes has shape of {modes.shape}")
    #u_truth = u_truth * u_std
    Ranks = np.zeros(latent_dim, dtype=int)
    Latent_Range = np.arange(latent_dim)

    Ecum = []
    partialModes = np.zeros_like(modes, dtype=np.float32)

    for i in range(latent_dim):
        Eks = []
        print(f"\nAt element {i}:\n")
        for j in Latent_Range:  # for mode in remaining modes
            #print(Ranks[:i], j, end="")
            partialModes *= 0
            partialModes[:, Ranks[:i]] = modes[:, Ranks[:i]]
            partialModes[:, j] = modes[:, j]
            u_pred = []

            dl = DataLoader(
                            TensorDataset(torch.from_numpy(partialModes)),
                            batch_size= 64,
                            )
            print('DECODING')
            for pmode in range(partialModes.shape[0]):
                #print('Points',pmode)
                u_p = model.decoder(torch.from_numpy(partialModes[np.newaxis,pmode,:]).float().to(device))
                u_p = torch.tensor(u_p, dtype=torch.float32)
                u_pred.append(u_p.detach().cpu().numpy())
            # print('U_P', u_p.shape())
            # print('DECODING 1',u_pred.shape())
            u_pred = np.array(u_pred).squeeze()

            #u_pred *= u_std
            try:
                u_pred.shape == u_truth.shape 
            except: 
                print(f"The shape Not matcheed")
                quit()
            u_pred = u_pred.reshape((len(time), 3, nx * ny * nz)).T
            print('Energy ranking:',u_truth.shape, u_pred.shape)
            energy = Energy_Rec(truth= u_truth , pred= u_pred)
            E_u = Energy_Rec(truth= u_truth[:,0,:], pred= u_pred[:,0,:])
            E_v =  Energy_Rec(truth= u_truth[:,1,:], pred= u_pred[:,1,:])
            E_w =  Energy_Rec(truth= u_truth[:,2,:], pred= u_pred[:,2,:])
            E_total = (E_u + E_v + E_w)/2
            print(energy.shape,E_total.shape, E_total)
            Eks.append(E_total)
            print(len(Eks))
            
            del u_pred
            #print(f'For mode {j}: Ek={Eks[-1]}')
            
        Eks = np.array(Eks).squeeze()
        print('Energy:',Eks.shape)
        ind = Latent_Range[np.argmax(Eks)]
        Ranks[i] = ind
        Latent_Range = np.delete(Latent_Range, np.argmax(Eks))
        Ecum.append(np.max(Eks))
        print('Adding: ', ind, ', Ek: ', np.max(Eks))
        print('############################################')
    
    Ecum = np.array(Ecum)
    print(f"Rank finished, the rank is {Ranks}")
    print(f"Cumulative Ek is {Ecum}")

    return np.array(Ranks), Ecum
###############################################

print('PRE')
# u_x, std_x, mean_x = preprocess_variable(VARIABLES[0])
# print('UY')
# u_y, std_y, mean_y = preprocess_variable(VARIABLES[1])
# print('UZ')
# u_z, std_z, mean_z = preprocess_variable(VARIABLES[2])
u_x = dataset[VARIABLES[0]]
u_y = dataset[VARIABLES[1]]
u_z = dataset[VARIABLES[2]]

# Extract spatial dimensions
nx = len(np.unique(dataset.xyz[:, 0]))
ny = len(np.unique(dataset.xyz[:, 1]))
nz = len(np.unique(dataset.xyz[:, 2]))
vae_dataset = pyLOM.NN.Dataset((u_x, u_y, u_z), (nx, ny, nz))
vae_dataset.crop(nx, ny, nz)

vae_dataset.pad(nx, ny, nz)
# print('REC AF', recdtset.shape)
# dataset.add_field('urec', 1, recdtset[:, 0, :, :].numpy().reshape((len(time), nx * ny * nz)).T)
# dataset.add_field('utra', 1, recdtset[:, 1, :, :].numpy().reshape((len(time), nx * ny * nz)).T)
# dataset.add_field('unor', 1, recdtset[:, 2, :, :].numpy().reshape((len(time), nx * ny * nz)).T)
print('VELOX',u_x.shape)
dataset.add_field('u_x', 1, u_x.reshape((len(time), nx * ny * nz)).T)
dataset.add_field('u_y', 1, u_y.reshape((len(time), nx * ny * nz)).T)
dataset.add_field('u_z', 1, u_z.reshape((len(time), nx * ny * nz)).T)
# dataset.add_field('mode x',1, modes)
# dataset.add_field('mode y',1, modes_y)
# dataset.add_field('mode z',1, modes_z)

save_path1 = os.path.join(RESULTS_DIR, 'snapshot_0_component_1.png')
save_path2 = os.path.join(RESULTS_DIR, 'snapshot_0_component_2.png')
save_path3 = os.path.join(RESULTS_DIR, 'snapshot_0_component_3.png')
for i in range(10):
    save_path1 = os.path.join(RESULTS_DIR, f'snapshot_{i}_component_1.png')
    pyLOM.NN.plotSnapshot(mesh, dataset, vars=['u_x'], instant=i, component=0, cmap='plasma', save_path=save_path1)

pyLOM.NN.plotSnapshot(mesh, dataset, vars=['u_y'], instant=10, component=0, cmap='plasma', save_path=save_path2)
pyLOM.NN.plotSnapshot(mesh, dataset, vars=['u_z'], instant=10, component=0, cmap='plasma', save_path=save_path3)
pyLOM.cr_info()
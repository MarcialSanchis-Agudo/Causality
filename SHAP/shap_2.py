"""
Evaluate the performance of beta-VAE
"""
import os
import sys
import  h5py
import  numpy as np 
import pandas as pd
import mat73
import matplotlib.ticker as ticker
import  matplotlib.pyplot as plt 
sys.path.append(os.path.abspath('/mimer/NOBACKUP/groups/kthmech/sanchis/scratch/Beta-VAE-combined-with-Transformers-for-ROM/'))
from    utils.VAE.AutoEncoder import BetaVAE
from utils.VAE.AutoSHAP import BetaVAE as BetaVAESHAP
# from    utils.configs         import VAE_custom as args, Name_Costum_VAE
from    utils.post         import VAE_custom as args, Name_Costum_VAE
import  torch
from    torch.utils.data      import DataLoader, TensorDataset
from    tqdm                  import tqdm
from utils.plot               import colorplate as cc 
from utils.pp import *
import time
from utils.pp import Power_Spectrum_Density
import shap
import seaborn as sns
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable

torch.manual_seed(1024)
device = ("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Unlock the h5py file

# Function to print all datasets in the HDF5 file
def print_h5_datasets(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return

    try:
        with h5py.File(file_path, 'r') as hf:
            def print_name(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"Dataset: {name}, Shape: {obj.shape}, Data Type: {obj.dtype}")
            
            print("---- Datasets in the HDF5 file ----")
            hf.visititems(print_name)
    except OSError as e:
        print(f"Error opening file: {e}")

batch_size = args.batch_size
beta = args.beta
latent_dim = args.latent_dim
epochs = args.epoch
split_ratio = args.train_split
lr = args.lr
es =  args.earlystop
model_type = args.model

# base_dir        = os.getcwd()
base_dir = "/mimer/NOBACKUP/groups/kthmech/sanchis/scratch/Beta-VAE-combined-with-Transformers-for-ROM"

base_dir        += "/"
print(f"Current dir : {base_dir}")
#datafile        = '../../Abhijeet2DobsData/OneObs2D-05k_z0_test-v2.h5'
datafile        = '../../Abhijeet2DobsData/OneObs2D-25k_z0_train-v2.h5'
new_path = '/mimer/NOBACKUP/groups/kthmech/abhvis/Projects/UFLOW_data/obstacleNOTRIP_T_233_283_N5e/DATASET37e/U.mat'
CheckPoint_path = "results/"
csv_file        = "vae_results.csv"

with h5py.File(datafile, 'r') as f:
    u_keras   = np.array(f['u_fluc'][:],dtype=np.float32)
    nt,nx,ny  = f['t'][()], f['x'][()],f['y'][()]
    u_mean    = f['means'][:]
    u_std     = np.array(f['u_fluc'][:],dtype=np.float32)
    u_v = np.array(f['v_fluc'][:],dtype=np.float32)

    new_data = np.zeros((len(nt[:,0]),304,112))
    new_data[:,:301,:101] = u_keras 
    u_keras = np.nan_to_num(u_keras)
    u_v = np.nan_to_num(u_v)
    u_keras = np.transpose(u_keras[:, :288, :96], (0, 2, 1))
    u_v = np.transpose(u_v[:, :288, :96], (0, 2, 1))
    u_std = np.stack([u_keras, u_v], axis=1)

print(f"The shape of data: {u_std.shape}")
# Paths
base_path = "/mimer/NOBACKUP/groups/kthmech/abhvis/Projects/UFLOW_data"
output_h5_path = "../../Abhijeet2DobsData/OneObs2D-{snapshots}_z0_train-v2.h5"

# Folder names
folders = [
    "obstacleNOTRIP_T_183_233_N5e",
    "obstacleNOTRIP_T_233_283_N5e",
]

for folder_index, folder in enumerate(folders):
    folder_path = os.path.join(base_path, folder)
    datasets = [d for d in os.listdir(folder_path) if d.startswith("DATASET")]
    
    folder_u_data = []
    folder_v_data = []
    folder_time_data = []
    folder_snapshots = 0  # Counter for snapshots in the current folder
    
    for dataset in datasets:
        dataset_path = os.path.join(folder_path, dataset)
        
        # Load U.mat and V.mat
        U_mat_path = os.path.join(dataset_path, "U.mat")
        V_mat_path = os.path.join(dataset_path, "V.mat")
        
        if not (os.path.exists(U_mat_path) and os.path.exists(V_mat_path)):
            print(f"Skipping {dataset} due to missing U.mat or V.mat.")
            continue
        
        print(f"Loading {U_mat_path} and {V_mat_path}")
        U_dict = mat73.loadmat(U_mat_path)
        V_dict = mat73.loadmat(V_mat_path)
        
        # Extract fields
        U = np.array(U_dict["U"], dtype=np.float32)
        V = np.array(V_dict["V"], dtype=np.float32)
        
        x = np.array(U_dict["x"], dtype=np.float32)
        y = np.array(U_dict["y"], dtype=np.float32)
        z = np.array(U_dict["z"], dtype=np.float32)
        t = np.array(U_dict["t"], dtype=np.float32)


        ######## RESHAPE #############
        U = U.reshape((np.unique(t)).shape[0], (np.unique(x)).shape[0], (np.unique(y)).shape[0], (np.unique(z)).shape[0])
        V = V.reshape((np.unique(t)).shape[0], (np.unique(x)).shape[0], (np.unique(y)).shape[0], (np.unique(z)).shape[0])

        # Identify the z = 0 plane
        z_index = np.where(np.isclose(z, 0))[0]
        if z_index.size == 0:
            print(f"No z=0 plane found in {dataset}. Skipping.")
            continue
        print('TIME', t.shape)
        # Extract the 2D plane
        U_2d = U[:, :, :, z_index[0]].squeeze()
        V_2d = V[:, :, :, z_index[0]].squeeze()
        time = t.squeeze()
        
        # Collect time snapshots and velocity data
        folder_u_data.append(U_2d)
        folder_v_data.append(V_2d)
        folder_time_data.append(time)  # Tile time for each snapshot
        folder_snapshots += U_2d.shape[0]
    
    # Concatenate folder-specific data
    folder_u_data = np.concatenate(folder_u_data, axis=0)
    folder_v_data = np.concatenate(folder_v_data, axis=0)
    folder_time_data = np.concatenate(folder_time_data, axis=0)  # Concatenate time data
    print('L', folder_time_data.shape)
    # Define output file name
    folder_id = folder_index + 1  # Use 1-based indexing
    output_h5 = f"../../Abhijeet2DobsData/OneObs2D-{folder_snapshots}_z0_train-v2_{folder_id}.h5"
    print(f"Saving data for folder {folder} to {output_h5}")
    
    # Save data to an HDF5 file
    with h5py.File(output_h5, "w") as h5f:
        h5f.create_dataset("u_fluc", data=folder_u_data, dtype=np.float32)
        h5f.create_dataset("v_fluc", data=folder_v_data, dtype=np.float32)
        h5f.create_dataset("time", data=folder_time_data, dtype=np.float32)  # Save time
        h5f.create_dataset("x", data=x, dtype=np.float32)
        h5f.create_dataset("y", data=y, dtype=np.float32)
        h5f.create_dataset("t", data=t, dtype=np.float32)
    
    print(f"Data for folder {folder} saved. Total snapshots: {folder_snapshots}")
Ntrain      = int(args.test_split* len(nt[:,0]))

# # We treat VAE as a method of modal decomposition,so we use whole dataset for test
# u_t     = torch.tensor(u_std[:,:,:,:])
# u       = TensorDataset(u_t, u_t)
# dl      = DataLoader(u ,batch_size = 1)
# fileID  = Name_Costum_VAE(args, 25000)
# print(f"The fileID will be {fileID}")

# ckpt    = torch.load(base_dir + CheckPoint_path +fileID+".pt",map_location=device)
# model_shap    = BetaVAESHAP(    
#                     zdim         = args.latent_dim, 
#                     knsize       = args.knsize, 
#                     beta         = args.beta, 
#                     filters      = args.filters,
#                     block_type   = args.block_type,
#                     lineardim    = args.linear_dim,
#                     lineardim2    = args.linear_dim2,
#                     act_conv     = args.act_conv,
#                     act_linear   = args.act_linear)

# # model    = BetaVAE(    
# #                     zdim         = args.latent_dim, 
# #                     knsize       = args.knsize, 
# #                     beta         = args.beta, 
# #                     filters      = args.filters,
# #                     block_type   = args.block_type,
# #                     lineardim    = args.linear_dim,
# #                     lineardim2    = args.linear_dim2,
# #                     act_conv     = args.act_conv,
# #                     act_linear   = args.act_linear)

# # model.load_state_dict(ckpt["model"])
# # model.to(device)
# # model.eval()
# print("INFO: Model has been correctly loaded")


# # def print_layer_shapes(model, model_name):
# #     print(f"\n{model_name} Layer Shapes:")
# #     for name, param in model.named_parameters():
# #         print(f"{name}: {param.shape}")

# # # Print shapes for both models
# # print_layer_shapes(model, "Original BetaVAE")
# # print_layer_shapes(model_shap, "SHAP BetaVAE")

# model_shap.load_state_dict(ckpt["model"])

# # Step 4: Move the new model to the appropriate device (e.g., GPU if available)
# model_shap.to(device)
# model_shap.eval()

# ######
# #Temporal Modes
# ######
# print(f"INFO: Generating temporal modes")

# Z_tot = []; Pred = []; Z_vector = []
# for x,y in tqdm(dl):
#     x   = x.float().to(device)
#     z_tot, pred = model_shap(x)

#     Z_tot.append(z_tot.detach().cpu().numpy())
#     Pred.append(pred.detach().cpu().numpy())

# Z_tot  = np.array(Z_tot).squeeze()
# Pred    = np.array(Pred).squeeze()
# print(f"Shape of z_tot: {Z_tot.shape}")
# print(f"Shape of out: {Pred.shape}")


# #######################33 SHAP <3333333333333333333333
# shap_base_path = "shap_values"

# # Define the length of each batch and the number of sets
# length = 500
# sets = int(25000 / length)
# print('DATABASE:', sets, length)

# # Move the tensor to the device and set the background
# u_t = torch.tensor(u_std).to(device)
# Background = u_t[:7000, :, :, :]
# print('Background prepared')

# for i in range(sets):
#     # Define the test batch
#     test = u_t[i * length:(i + 1) * length, :, :, :]
    
#     # Define the shap_save_path for each iteration
#     shap_save_path = f"{shap_base_path}_{i}.h5"
    
#     # Instantiate the SHAP explainer and compute SHAP values
#     explainer = shap.GradientExplainer(model_shap.encoder, Background)
#     print('Explaining batch:', i)
    
#     # Calculate SHAP values for the current batch
#     shap_val = explainer.shap_values(test)
    
#     # Save SHAP values in the respective HDF5 file for the current batch
#     with h5py.File(shap_save_path, 'w') as hf:  # 'w' mode ensures each file is newly created
#         # Saving SHAP values for channel 0 (u) and channel 1 (v)
#         hf.create_dataset(f'u_shap_{i}K', data=shap_val[:, 0, :, :, :])  # Optional compression can be added
#         hf.create_dataset(f'v_shap_{i}K', data=shap_val[:, 1, :, :, :])
    
#     print(f"SHAP values saved successfully to {shap_save_path}")
#     print_h5_datasets(shap_save_path)  # Optional: Print datasets for verification

# # Reshape SHAP values to 5D (Samples, Channels, Y, X, Latent Modes)
# # Ensure SHAP values are absolute before processing
# shap_values = np.abs(shap_val)

# # Define a threshold based on the mean or a specific percentile
# threshold_percentile = 95  # Keep the top 20% of SHAP values globally across all samples
# threshold = np.percentile(shap_values, threshold_percentile)

# # Mask SHAP values below the threshold globally
# shap_values_masked = np.where(shap_values > threshold, shap_values, 0)

# # Print shapes for verification
# print(f"Shape of SHAP values: {shap_values.shape}")
# print(f"Shape of threshold: {threshold}")

# # Visualization of SHAP value distribution
# num_channels = shap_values.shape[1]
# num_modes = shap_values.shape[-1]

# # Visualize SHAP values distribution for each channel and mode
# for j in range(num_channels):
#     for i in range(num_modes):
#         plt.figure(figsize=(8, 4))  # Create a new figure for each channel and mode

#         # Plot histogram of SHAP values for the current channel and mode
#         plt.hist(shap_values[:, j, :, :, i].flatten(), bins=50, alpha=0.75, color='blue', label=f'Mode {i + 1}')

#         # Plot the global threshold
#         plt.axvline(threshold, color='red', linestyle='dashed', linewidth=1, label='Global Threshold')

#         plt.title(f'SHAP Value Distribution for Channel {j + 1}, Mode {i + 1}')
#         plt.xlabel('SHAP Value')
#         plt.ylabel('Frequency')
#         plt.legend()

#         # Save the histogram figure for the current channel and mode
#         plt.tight_layout()
#         plt.savefig(f'SHAP_Value_Distribution_Channel_{j + 1}_Mode_{i + 1}.png', dpi=300)
#         plt.close()  # Close the figure to free memory

# # Heatmap plotting
# nyy, nxx = 96, 288
# xb = np.array([-0.125, -0.125, 0.25, 0.25])
# yb = np.array([0.0, 1.0, 1.0, 0.0])

# # Assuming x and y are the grid coordinates for the data
# x = np.linspace(nx.min(), nx.max(), nxx)
# y = np.linspace(ny.min(), ny.max(), nyy)
# x, y = np.meshgrid(x, y)

# # Set colormap
# plt.set_cmap('plasma')

# # Preparing SHAP values and ensuring they are absolute values
# shap_values_masked = np.abs(shap_values_masked)
# print('BEFORE MASKING:',shap_values_masked.shape)  # Use absolute values of SHAP
# shap_values_mean = shap_values_masked.mean(axis=0)  # Average SHAP values over the time axis

# # 1. Calculate global min and max SHAP values across all channels and modes
# global_min = np.min(shap_values_masked)
# global_max = np.max(shap_values_masked)

# print(f"Global Min SHAP Value: {global_min}")
# print(f"Global Max SHAP Value: {global_max}")

# # 2. Plot SHAP heatmaps with the same scale for each channel and mode
# for j in range(num_channels):  # Loop over channels
#     for i in range(num_modes):
#         fig, ax = plt.subplots(figsize=(8, 6))

#         # Extract SHAP values for the current latent mode and channel
#         shap_data = shap_values_masked[0,j,:,:,i]

#         # Plot SHAP mode heatmap with a global scale using vmin and vmax
#         cb1 = ax.contourf(x, y, shap_data, levels=500, vmin=global_min, vmax=global_max)
#         ax.fill(xb, yb, c='w', zorder=3)  # Fill object in white
#         ax.plot(xb, yb, c='k', lw=1, zorder=5)  # Object boundary in black
#         ax.set_aspect('equal')
#         ax.set_title(f"SHAP Mode {i + 1}, Channel {j + 1}", fontdict={"size": 18})

#         # Create a colorbar for SHAP mode with a consistent scale
#         divider1 = make_axes_locatable(ax)
#         cax1 = divider1.append_axes("right", size="5%", pad=0.05)
#         cbar1 = plt.colorbar(cb1, cax=cax1, orientation='vertical', format="%.2f")
#         cbar1.locator = ticker.MaxNLocator(nbins=5)  # Adjust number of ticks
#         cbar1.update_ticks()

#         # Save the figure
#         plt.tight_layout()
#         plt.savefig(f"SHAP_Mode_Heatmap_Channel_{j + 1}_Mode_{i + 1}.png", dpi=300)
#         plt.close()



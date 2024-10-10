"""
Evaluate the performance of beta-VAE
"""
import os
import sys
import  h5py
import  numpy as np 
import pandas as pd
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


Ntrain      = int(args.test_split* len(nt[:,0]))

# We treat VAE as a method of modal decomposition,so we use whole dataset for test
u_t     = torch.tensor(u_std[:,:,:,:])
u       = TensorDataset(u_t, u_t)
dl      = DataLoader(u ,batch_size = 1)
fileID  = Name_Costum_VAE(args, 25000)
print(f"The fileID will be {fileID}")

ckpt    = torch.load(base_dir + CheckPoint_path +fileID+".pt",map_location=device)
model_shap    = BetaVAESHAP(    
                    zdim         = args.latent_dim, 
                    knsize       = args.knsize, 
                    beta         = args.beta, 
                    filters      = args.filters,
                    block_type   = args.block_type,
                    lineardim    = args.linear_dim,
                    lineardim2    = args.linear_dim2,
                    act_conv     = args.act_conv,
                    act_linear   = args.act_linear)

# model    = BetaVAE(    
#                     zdim         = args.latent_dim, 
#                     knsize       = args.knsize, 
#                     beta         = args.beta, 
#                     filters      = args.filters,
#                     block_type   = args.block_type,
#                     lineardim    = args.linear_dim,
#                     lineardim2    = args.linear_dim2,
#                     act_conv     = args.act_conv,
#                     act_linear   = args.act_linear)

# model.load_state_dict(ckpt["model"])
# model.to(device)
# model.eval()
print("INFO: Model has been correctly loaded")


# def print_layer_shapes(model, model_name):
#     print(f"\n{model_name} Layer Shapes:")
#     for name, param in model.named_parameters():
#         print(f"{name}: {param.shape}")

# # Print shapes for both models
# print_layer_shapes(model, "Original BetaVAE")
# print_layer_shapes(model_shap, "SHAP BetaVAE")

model_shap.load_state_dict(ckpt["model"])

# Step 4: Move the new model to the appropriate device (e.g., GPU if available)
model_shap.to(device)
model_shap.eval()

######
#Temporal Modes
######
print(f"INFO: Generating temporal modes")

Z_tot = []; Pred = []; Z_vector = []
for x,y in tqdm(dl):
    x   = x.float().to(device)
    z_tot, pred = model_shap(x)

    Z_tot.append(z_tot.detach().cpu().numpy())
    Pred.append(pred.detach().cpu().numpy())

Z_tot  = np.array(Z_tot).squeeze()
Pred    = np.array(Pred).squeeze()
print(f"Shape of z_tot: {Z_tot.shape}")
print(f"Shape of out: {Pred.shape}")


#######################33 SHAP <3333333333333333333333
u_t = torch.tensor(u_std).to(device)
Background = u_t[:1000,:,:,:]
print('BACkground')
test = u_t[24000:240001,:,:,:]
explainer = shap.GradientExplainer(model_shap.encoder,Background)
print('Explain')
shap_val = explainer.shap_values(test)
print('SHAPES')

# Reshape SHAP values to 5D (Samples, Channels, Y, X, Latent Modes)
# Ensure SHAP values are absolute before processing
shap_values = np.abs(shap_val)

# Define a threshold based on the mean or a specific percentile
threshold_percentile = 95  # Keep the top 20% of SHAP values globally across all samples
threshold = np.percentile(shap_values, threshold_percentile)

# Mask SHAP values below the threshold globally
shap_values_masked = np.where(shap_values > threshold, shap_values, 0)

# Print shapes for verification
print(f"Shape of SHAP values: {shap_values.shape}")
print(f"Shape of threshold: {threshold}")

# Visualization of SHAP value distribution
num_channels = shap_values.shape[1]
num_modes = shap_values.shape[-1]

# Visualize SHAP values distribution for each channel and mode
for j in range(num_channels):
    for i in range(num_modes):
        plt.figure(figsize=(8, 4))  # Create a new figure for each channel and mode

        # Plot histogram of SHAP values for the current channel and mode
        plt.hist(shap_values[:, j, :, :, i].flatten(), bins=50, alpha=0.75, color='blue', label=f'Mode {i + 1}')

        # Plot the global threshold
        plt.axvline(threshold, color='red', linestyle='dashed', linewidth=1, label='Global Threshold')

        plt.title(f'SHAP Value Distribution for Channel {j + 1}, Mode {i + 1}')
        plt.xlabel('SHAP Value')
        plt.ylabel('Frequency')
        plt.legend()

        # Save the histogram figure for the current channel and mode
        plt.tight_layout()
        plt.savefig(f'SHAP_Value_Distribution_Channel_{j + 1}_Mode_{i + 1}.png', dpi=300)
        plt.close()  # Close the figure to free memory

# Heatmap plotting
nyy, nxx = 96, 288
xb = np.array([-0.125, -0.125, 0.25, 0.25])
yb = np.array([0.0, 1.0, 1.0, 0.0])

# Assuming x and y are the grid coordinates for the data
x = np.linspace(nx.min(), nx.max(), nxx)
y = np.linspace(ny.min(), ny.max(), nyy)
x, y = np.meshgrid(x, y)

# Set colormap
plt.set_cmap('plasma')

# Preparing SHAP values and ensuring they are absolute values
shap_values_masked = np.abs(shap_values_masked)
print('BEFORE MASKING:',shap_values_masked.shape)  # Use absolute values of SHAP
shap_values_mean = shap_values_masked.mean(axis=0)  # Average SHAP values over the time axis

# Plot SHAP heatmaps
for j in range(num_channels):  # Loop over channels
    for i in range(num_modes):
        fig, ax = plt.subplots(figsize=(8, 6))

        # Extract SHAP values for the current latent mode and channel
        # shap_data = shap_values_mean[j, :, :, i]
        shap_data = shap_values_masked[0,j,:,:,i]

        # Plot SHAP mode heatmap
        cb1 = ax.contourf(x, y, shap_data, levels=500)
        ax.fill(xb, yb, c='w', zorder=3)  # Fill object in white
        ax.plot(xb, yb, c='k', lw=1, zorder=5)  # Object boundary in black
        ax.set_aspect('equal')
        ax.set_title(f"SHAP Mode {i + 1}, Channel {j + 1}", fontdict={"size": 18})

        # Create a colorbar for SHAP mode
        divider1 = make_axes_locatable(ax)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        cbar1 = plt.colorbar(cb1, cax=cax1, orientation='vertical', format="%.2f")
        cbar1.locator = ticker.MaxNLocator(nbins=5)  # Adjust number of ticks
        cbar1.update_ticks()

        # Save the figure
        plt.tight_layout()
        plt.savefig(f"SHAP_Mode_Heatmap_Channel_{j + 1}_Mode_{i + 1}.png", dpi=300)
        plt.close()


import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import seaborn as sns
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter1d


# Set a minimalistic, elegant, space-like style for all plots
plt.rc('font', family='monospace')    # Classic mathematical font
plt.rc('axes', labelsize=18)                 # Axis label size
plt.rc('font', size=22)                      # General font size
plt.rc('legend', fontsize=18)                # Legend font size
plt.rc('xtick', labelsize=16)                # X-tick label size
plt.rc('ytick', labelsize=16)                # Y-tick label size
plt.rcParams['mathtext.fontset'] = 'stix'   # Elegant math fonts
colors = plt.cm.plasma(np.linspace(0, 1, 13))


datafile = '../../Abhijeet2DobsData/OneObs2D-25k_z0_train-v2.h5'

with h5py.File(datafile, 'r') as f:
    u_keras = np.array(f['u_fluc'][:], dtype=np.float32)
    nt, nx, ny = f['t'][()], f['x'][()], f['y'][()]
    u_v = np.array(f['v_fluc'][:],dtype=np.float32)

    u_keras = np.transpose(u_keras[:, :288, :96], (0, 2, 1))
    u_v = np.transpose(u_v[:, :288, :96], (0, 2, 1))
print('X', nx.shape, nx)
# Define the path to the saved structures file
S_i = 1
perc = 15
# shap_save_path = f'/mimer/NOBACKUP/groups/deepmechalvis/marcial/structures_time_{s_i}.h5'
local = 0
steps_ = 10
steps_1 = 1000
size = 25000
steps = steps_1
vars = 10
values = 10
sigma = 1
all_structures = []
shap_name = f'shap_values_{S_i}_{perc}'
output_dir = f'testing/{shap_name}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
ranks = [10,7,4,8,1,2,6,3,9,5]
# Initialize the main list to collect structures from all files
all_structures = []
print('VELOCITY FIELD', u_keras.shape)
pdf_y = True
pdf_V = True
# Loop over the range of s_i values (from 0 to 10)
for s_i in range(S_i):  # 11 because we want to include 10 (0 to 10)
    #shap_save_path = f'/mimer/NOBACKUP/groups/kthmech/sanchis/scratch/STRUC/structures_time_{s_i}.h5'
    shap_save_path = f'/mimer/NOBACKUP/groups/kthmech/sanchis/scratch/STRUC/structures_time_{perc}_{s_i}.h5'

    print('FILES:',s_i)
    # Open each HDF5 file and read the structures
    with h5py.File(shap_save_path, 'r') as hf:
        if s_i == 0:
            for j in range(len(hf.keys())):
                all_structures.append([])
        for j in range(len(hf.keys())):
            var_group = hf[f'var_{j}']
            structures_for_var = []
            for t in range(len(var_group.keys())):#for t in range(steps):
                time_group = var_group[f'time_{t}']
                time_structures = []
                for i in range(len(time_group.keys()) // 2):
                    structure_y = time_group[f'structure_{i}_v'][:]
                    structure_x = time_group[f'structure_{i}_u'][:]
                    time_structures.append([structure_x, structure_y])
                #structures_for_var.append(time_structures)
                all_structures[j].append(time_structures)
# Initialize matrices
matrix_t = np.zeros((steps, 96, 288))
frequency_matrix_l = np.zeros((10, 96, 288))
u_x = np.zeros((steps,vars,96,288))
u_v_y = np.zeros((steps,vars,96,288))
matrix_t_l = np.zeros((10,steps, 96, 288))
x = np.linspace(nx.min(), nx.max(), len(nx))  # Spatial coordinates for x
print('Loaded data')
# Populate matrices
for variable in range(vars):
    for t in range(len(all_structures[variable])):
        for i in range(len(all_structures[variable][t])):
            structure_x = all_structures[variable][t][i][0]
            structure_y = all_structures[variable][t][i][1]
            for j in range(len(structure_x)):
                y_coord = structure_y[j]  # Note: structure_x represents y
                x_coord = structure_x[j]  # Note: structure_y represents x
                matrix_t[t, x_coord,y_coord] = i + 1
                frequency_matrix_l[variable, x_coord,y_coord] += 1
                matrix_t_l[variable,t,x_coord,y_coord] = 1



    for t in range(steps):  # Loop over time steps
        high_frequency_indices = np.where(matrix_t_l[variable,t,:,:] > 0)
        # print('HIGH', high_frequency_indices[0].shape)
        u_x[t,variable, high_frequency_indices[0], high_frequency_indices[1]] = u_keras[t, high_frequency_indices[0], high_frequency_indices[1]]
        u_v_y[t, variable, high_frequency_indices[0], high_frequency_indices[1]] = u_v[t, high_frequency_indices[0], high_frequency_indices[1]]
####################
def load_all_shap_values(start_index, end_index):
    shap_u = []
    shap_v = []

    for i in range(start_index, end_index + 1):
        file_name = f"SHAPS/shap_values_{i}.h5"
        
        # Open the HDF5 file
        with h5py.File(file_name, 'r') as hf:
            # Load the datasets and append them to the lists
            u_shap_data = hf[f'u_shap_{i}K'][:]  # Adjust the name based on your file structure
            v_shap_data = hf[f'v_shap_{i}K'][:]
            
            shap_u.append(u_shap_data)
            shap_v.append(v_shap_data)
    
    # Convert lists to numpy arrays if you need them in array form
    shap_u = np.concatenate(shap_u, axis=0)  # Concatenate along the appropriate axis
    shap_v = np.concatenate(shap_v, axis=0)
    
    return shap_u, shap_v
# Load shap values
shap_name = "shap_values_25K"
shap_file = "shap_values_1K.h5"
datafile = '../../Abhijeet2DobsData/OneObs2D-25k_z0_train-v2.h5'
save_path = 'geometry'
# Load shap values
# with h5py.File(shap_file, 'r') as hf:
#     shap_u = np.array(hf['u'])
#     shap_v = np.array(hf['v'])
shap_u, shap_v = load_all_shap_values(0, 49)
# --- Geom#3 lOAD SHAPS ############etric Analysis of High-SHAP Regions ---
print('PREE-SMOOTHING')
shap_u_smooth = gaussian_filter1d(shap_u, sigma=sigma)
shap_v_smooth = gaussian_filter1d(shap_v, sigma=sigma)
print('MSE')
mse_u = np.mean(shap_u_smooth**2,axis=(0,1,2))
mse_v = np.mean(shap_v_smooth**2,axis=(0,1,2))
mse = np.sqrt((mse_u + mse_v))
H_values = [1.11,1.17,1.22,1,1.06,1,1.06,1.11,1.22,1.11]
# Assuming you have defined 'steps', 'nx', 'ny', and 'vars'
binary_total = np.zeros((steps, 96, 288, vars), dtype=np.int32)  # Initialize binary_total to the required shape
print('H_values')
# print(f'MSE for shap_u: {mse_u}, MSE for shap_v: {mse_v}, MSE total: {mse}, {shap_u.shape}, {mse.shape},{mse_u.shape}')
# Calculate the Mean Square Error (MSE) for both channels
# H = 0.1
#print(f'MSE for shap_u: {mse_u}, MSE for shap_v: {mse_v}, {shap_u.shape}, {mse.shape},{mse_u.shape}')
for i in [3,9]:
    # binary_mask_u = np.where(shap_u_smooth[:,:,3:,i] > H_values[np.argmax(total[i,:])] * mse, 1, 0)
    # binary_mask_v = np.where(shap_v_smooth > H_values[np.argmax(total[i,:])] * mse, 1, 0)
    # binary_total[:,:,:,i] = np.where(np.sqrt(shap_u_smooth[:,:,:,i] **2 + shap_v_smooth[:,:,:,i] **2) > H_values[np.argmax(total[i,:])]*mse[i] , 1, 0)
    print('Hs', H_values[i])
    binary_total[:,:,:,i] = np.where(np.sqrt(shap_u_smooth[:steps,:,:,i] **2 + shap_v_smooth[:steps,:,:,i] **2) > H_values[i]*mse[i] , 1, 0)
###################################
def compute_curvature(field):
    grad_x = np.gradient(field, axis=0)
    grad_y = np.gradient(field, axis=1)
    curvature = np.abs(np.gradient(grad_x, axis=0) + np.gradient(grad_y, axis=1))
    return curvature
def compute_vorticity(field):
    grad_x = np.gradient(field, axis=0)
    grad_y = np.gradient(field, axis=1)
    return grad_y - grad_x
print('SHAPES', matrix_t_l.shape,u_x.shape,binary_total.shape)
for var in [3,9]:
    shap_curvature = compute_curvature(matrix_t_l[var,:,:,:].mean(axis=0))
    field_curvature = compute_curvature(u_x[:,var,:,:].mean(axis=0))
    shap_vorticity = compute_vorticity(matrix_t_l[var,:,:,:].mean(axis=0))
    field_vorticity = compute_vorticity(u_x[:,var,:,:].mean(axis=0))
    imp_cur = compute_curvature(binary_total[:,:,:,var].mean(axis=0))
    imp_vorticity = compute_vorticity(binary_total[:,:,:,var].mean(axis=0))
    

    energy_transfer = matrix_t_l[var,:,:,:].mean(axis=0) * shap_curvature
    energy_transfer_field = u_x[var,:,:,:].mean(axis=0) * field_curvature
    energy_imp = binary_total[:,:,:,var].mean(axis=0) * imp_cur

    ##################33
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(matrix_t_l[var,:,:,:].mean(axis=0), extent=[nx[:288].min(), nx[:288].max(), ny[:96].min(), ny[:96].max()], origin='lower',cmap='plasma')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04,  shrink=0.7)       

    # Overlay obstacle
    xb = np.array([-0.125, -0.125, 0.25, 0.25])  # Obstacle x-coordinates
    yb = np.array([0.0, 1.0, 1.0, 0.0])  # Obstacle y-coordinates
    ax.fill(xb, yb, 'white')  # No legend, just visual

    # Style adjustments
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$y$', fontsize=14)
    ax.set_xticks(np.linspace(0, nx[:288].max(), 5))
    ax.set_yticks(np.linspace( ny[:96].min(),  ny[:96].max(), 5))
    # Add legend
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{save_path}/mean_struc_shap_{1}_{var}.png', dpi=300, bbox_inches='tight')
    plt.close()
    ##############################
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(binary_total[:,:,:,var].mean(axis=0), extent=[nx[:288].min(), nx[:288].max(), ny[:96].min(), ny[:96].max()], origin='lower',cmap='plasma')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04,  shrink=0.7)       

    # Overlay obstacle
    xb = np.array([-0.125, -0.125, 0.25, 0.25])  # Obstacle x-coordinates
    yb = np.array([0.0, 1.0, 1.0, 0.0])  # Obstacle y-coordinates
    ax.fill(xb, yb, 'white')  # No legend, just visual

    # Style adjustments
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$y$', fontsize=14)
    ax.set_xticks(np.linspace(0, nx[:288].max(), 5))
    ax.set_yticks(np.linspace( ny[:96].min(),  ny[:96].max(), 5))
    # Add legend
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{save_path}/mean_struc_shap_{2}_{var}.png', dpi=300, bbox_inches='tight')
    plt.close()
    ########################

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(shap_curvature, extent=[nx[:288].min(), nx[:288].max(), ny[:96].min(), ny[:96].max()], origin='lower',cmap='plasma')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04,  shrink=0.7)       

    # Overlay obstacle
    xb = np.array([-0.125, -0.125, 0.25, 0.25])  # Obstacle x-coordinates
    yb = np.array([0.0, 1.0, 1.0, 0.0])  # Obstacle y-coordinates
    ax.fill(xb, yb, 'white')  # No legend, just visual

    # Style adjustments
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$y$', fontsize=14)
    ax.set_xticks(np.linspace(0, nx[:288].max(), 5))
    ax.set_yticks(np.linspace( ny[:96].min(),  ny[:96].max(), 5))
    # Add legend
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{save_path}/mean_struc_shap_{3}_{var}.png', dpi=300, bbox_inches='tight')
    plt.close()
    ############33
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(field_curvature, extent=[nx[:288].min(), nx[:288].max(), ny[:96].min(), ny[:96].max()], origin='lower',cmap='plasma')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04,  shrink=0.7)       

    # Overlay obstacle
    xb = np.array([-0.125, -0.125, 0.25, 0.25])  # Obstacle x-coordinates
    yb = np.array([0.0, 1.0, 1.0, 0.0])  # Obstacle y-coordinates
    ax.fill(xb, yb, 'white')  # No legend, just visual

    # Style adjustments
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$y$', fontsize=14)
    ax.set_xticks(np.linspace(0, nx[:288].max(), 5))
    ax.set_yticks(np.linspace( ny[:96].min(),  ny[:96].max(), 5))
    # Add legend
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{save_path}/mean_struc_shap_{4}_{var}.png', dpi=300, bbox_inches='tight')
    plt.close()
    #############3

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(imp_cur, extent=[nx[:288].min(), nx[:288].max(), ny[:96].min(), ny[:96].max()], origin='lower',cmap='plasma')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04,  shrink=0.7)       

    # Overlay obstacle
    xb = np.array([-0.125, -0.125, 0.25, 0.25])  # Obstacle x-coordinates
    yb = np.array([0.0, 1.0, 1.0, 0.0])  # Obstacle y-coordinates
    ax.fill(xb, yb, 'white')  # No legend, just visual

    # Style adjustments
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$y$', fontsize=14)
    ax.set_xticks(np.linspace(0, nx[:288].max(), 5))
    ax.set_yticks(np.linspace( ny[:96].min(),  ny[:96].max(), 5))
    # Add legend
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{save_path}/mean_struc_shap_{5}_{var}.png', dpi=300, bbox_inches='tight')
    plt.close()
    #######################

    # plt.figure(figsize=(10, 8))
    # plt.imshow(field_vorticity, cmap='plasma')
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    # plt.title(f'Vorticity shap field Mode{[ranks[var]]}')
    # plt.savefig(f'{save_path}/mean_struc_shap_{6}_{var}.png', dpi=300, bbox_inches='tight')
    # plt.close()

    # plt.figure(figsize=(10, 8))
    # plt.imshow(energy_transfer, cmap='plasma')
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    # plt.title(f'Energy shap field Mode{[ranks[var]]}')
    # plt.savefig(f'{save_path}/mean_struc_shap_{7}_{var}.png', dpi=300, bbox_inches='tight')
    # plt.close()

    # plt.figure(figsize=(10, 8))
    # plt.imshow(energy_transfer_field, cmap='plasma')
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    # plt.title(f'Energy shap field Mode{[ranks[var]]}')
    # plt.savefig(f'{save_path}/mean_struc_shap_{8}_{var}.png', dpi=300, bbox_inches='tight')
    # plt.close()

    # plt.figure(figsize=(10, 8))
    # plt.imshow(imp_vorticity, cmap='plasma')
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    # plt.title(f'Vorticity bin field Mode{[ranks[var]]}')
    # plt.savefig(f'{save_path}/mean_struc_shap_{9}_{var}.png', dpi=300, bbox_inches='tight')
    # plt.close()

    # plt.figure(figsize=(10, 8))
    # plt.imshow(imp_cur, cmap='plasma')
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    # plt.title(f'Curvature bin field Mode{[ranks[var]]}')
    # plt.savefig(f'{save_path}/mean_struc_shap_{10}_{var}.png', dpi=300, bbox_inches='tight')
    # plt.close()

    # plt.figure(figsize=(10, 8))
    # plt.imshow(energy_imp, cmap='plasma')
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    # plt.title(f'Energy bin Mode{[ranks[var]]}')
    # plt.savefig(f'{save_path}/mean_struc_shap_{11}_{var}.png', dpi=300, bbox_inches='tight')
    # plt.close()
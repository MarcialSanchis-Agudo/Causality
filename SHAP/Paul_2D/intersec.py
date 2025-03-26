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
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter1d
import imageio

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
S_i = 5
perc = 15
# shap_save_path = f'/mimer/NOBACKUP/groups/deepmechalvis/marcial/structures_time_{s_i}.h5'
local = 0
steps_ = 10
steps_1 = 5000
size = 25000
steps = steps_1
vars = 3
values = 10
all_structures = []
shap_name = f'shap_values_{S_i}_{perc}_3'
output_dir = f'testing/{shap_name}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# ranks = [10,7,4,8,1,2,6,3,9,5]
ranks = [1,3,2]
# Initialize the main list to collect structures from all files
all_structures = []
print('VELOCITY FIELD', u_keras.shape)
pdf_y = True
pdf_V = True
# Loop over the range of s_i values (from 0 to 10)
for s_i in range(S_i):  # 11 because we want to include 10 (0 to 10)
    #shap_save_path = f'/mimer/NOBACKUP/groups/kthmech/sanchis/scratch/STRUC/structures_time_{s_i}.h5'
    shap_save_path = f'/mimer/NOBACKUP/groups/kthmech/sanchis/scratch/STRUC/lat_3_m/structures_time_{s_i}.h5'

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
matrix_g = np.zeros((steps, 96, 288))
matrix_t = np.zeros((steps, 96, 288))
frequency_matrix_l = np.zeros((10, 96, 288))
u_x = np.zeros((steps,vars,96,288))
u_v_y = np.zeros((steps,vars,96,288))
u_x_u = np.zeros((steps,vars,96,288))
u_v_u = np.zeros((steps,vars,96,288))
space = np.zeros((steps,vars,96,288,2))
appear = np.zeros((steps,vars))
appear_1 = np.zeros((steps,vars))
total_appear = np.zeros((steps))


matrix_t_l = np.zeros((vars,steps, 96, 288))
matrix_causal = np.zeros((vars,steps, 96, 288))
matrix_cau_g = np.zeros((steps, 96, 288))
matrix_cau_t = np.zeros((steps, 96, 288))

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
                matrix_t[t, x_coord,y_coord] = 1
                frequency_matrix_l[variable, x_coord,y_coord] += 1
                matrix_t_l[variable,t,x_coord,y_coord] = 3
                matrix_g[t, x_coord,y_coord] = matrix_t[t, x_coord,y_coord] + matrix_g[t, x_coord,y_coord]
                matrix_causal[variable,t,x_coord,y_coord] = -8
                if variable == 0:
                    matrix_cau_t[t, x_coord,y_coord] = 1
                elif variable == 1:
                    matrix_cau_t[t, x_coord,y_coord] = 2
                else:
                    matrix_cau_t[t, x_coord,y_coord] = 4
                matrix_cau_g[t, x_coord,y_coord] = matrix_cau_t[t, x_coord,y_coord] + matrix_cau_g[t, x_coord,y_coord]

for variable in range(vars):
    matrix_causal[variable,:,:,:] = matrix_causal[variable,:,:,:] + matrix_cau_g[:,:,:]

    for t in range(steps):  # Loop over time steps
        high_frequency_indices = np.where(matrix_t_l[variable, t, :, :] > 0)
        if variable == 0:
            unique_indices = np.where(matrix_causal[variable, t, :, :] == -7)
        elif variable == 1:
            unique_indices = np.where(matrix_causal[variable, t, :, :] == -6)
        else:
            unique_indices = np.where(matrix_causal[variable, t, :, :] == -4)

        full_in = np.where(
            (matrix_causal[variable, t, :, :] == -7) | 
            (matrix_causal[variable, t, :, :] == -4) | 
            (matrix_causal[variable, t, :, :] == -6)
        )

        # Extract index arrays
        hf_x, hf_y = high_frequency_indices
        u_x_idx, u_y_idx = unique_indices   
        x_l, y_l = full_in

        # Assign values
        space[t, variable, hf_x, hf_y, 0] = nx[hf_y]  # x-coordinates
        space[t, variable, hf_x, hf_y, 1] = ny[hf_x]  # y-coordinates        
        u_x[t, variable, hf_x, hf_y] = u_keras[t, hf_x, hf_y]
        u_x_u[t, variable, u_x_idx, u_y_idx] = u_keras[t, u_x_idx, u_y_idx]
        u_v_u[t, variable, u_x_idx, u_y_idx] = u_v[t, u_x_idx, u_y_idx]
        # u_x_u[t, variable, x_l, y_l] = u_keras[t, x_l, y_l]
        # u_v_u[t, variable, x_l, y_l] = u_v[t, x_l, y_l]
        u_v_y[t, variable, hf_x, hf_y] = u_v[t, hf_x, hf_y]

        # Corrected shape calculation
        appear[t, variable] = len(np.unique(u_x_idx))*len(np.unique(u_y_idx))
        appear_1[t,variable] = len(np.unique(x_l))*len(np.unique(y_l))
def plot_smooth_intersection(var, var_1, perc, matrix_t_l, steps, x_vals, save_path='testing/intersections/'):
    """
    Creates a smooth line plot with shaded areas for the contributions of Var, Var_1, and their intersection.
    
    Parameters:
    - var (int): First variable index.
    - var_1 (int): Second variable index.
    - matrix_t_l (numpy array): 4D array with shape (vars, steps, ?, x_dim) containing intersection data.
    - steps (int): Number of time steps.
    - x_vals (list or numpy array): x-coordinates for plotting.
    - save_path (str, optional): Path to save the generated plots.
    """

    x_dim = matrix_t_l.shape[3]  # Number of x-coordinates
    var_contributions = np.zeros(x_dim)
    var_1_contributions = np.zeros(x_dim)
    var_2_contributions = np.zeros(x_dim)
    intersection_contributions = np.zeros(x_dim)
    x_vals = np.ravel(x_vals)

    # Compute contributions
    for x_index in range(x_dim):
        total_var, total_var_1, total_var_2,total_intersection = [], [], [],[]

        for t in range(steps): 
            field_var = matrix_t_l[var, t, :, x_index]
            field_var_1 = matrix_t_l[var_1, t, :, x_index]
            field_val_mixed = field_var + field_var_1 

            total_var.append(np.sum(field_var > 0)/len(field_val_mixed))
            total_var_1.append(np.sum(field_var_1 > 0)/len(field_val_mixed))
            total_intersection.append(np.sum(field_val_mixed == 6)/len(field_val_mixed))

        # Compute mean values
        var_contributions[x_index] = np.mean(total_var)
        var_1_contributions[x_index] = np.mean(total_var_1)
        intersection_contributions[x_index] = np.mean(total_intersection)

    # Apply smoothing using Gaussian filter
    smooth_var = gaussian_filter1d(var_contributions, sigma=3)
    smooth_var_1 = gaussian_filter1d(var_1_contributions, sigma=3)
    smooth_intersection = gaussian_filter1d(intersection_contributions, sigma=3)


    # Define Plasma colormap colors
    plasma_colors = cm.plasma(np.linspace(0, 1, 5))
    color_var = plasma_colors[0]
    color_var_1 = plasma_colors[1]
    color_intersection = plasma_colors[2]


    # Plot with soft colors and no black lines
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.fill_between(x_vals, smooth_var, color=color_var, alpha=0.4, label=f'Mode {ranks[var]} Contribution')
    ax.fill_between(x_vals, smooth_var_1, color=color_var_1, alpha=0.4, label=f'Mode {ranks[var_1]} Contribution')
    ax.fill_between(x_vals, smooth_intersection, color=color_intersection, alpha=0.6, label='Intersection')


    # Minimalist aesthetics
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('Mean Contribution', fontsize=14)
    ax.set_title(f'Contributions Modes ${ranks[var]},{ranks[var_1]}$ and ${ranks[var]}\cap{ranks[var_1]}$ ', fontsize=10)
    ax.set_xlim([0,5])
    ax.legend(fontsize=12, loc='upper right', frameon=False)
    
    # Remove all spines (box lines)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Remove grid
    ax.grid(False)

    # Save the figure
    plt.tight_layout()
    plt.savefig(f'{save_path}/mean_intersection_smooth_{var}_vs_{var_1}_{perc}.png', dpi=300, bbox_inches='tight')
    plt.close()        
def plot_smooth_intersection_4(var, var_1,var2,var3, perc, matrix_t_l, steps, x_vals, save_path='testing/intersections/'):
    """
    Creates a smooth line plot with shaded areas for the contributions of Var, Var_1, and their intersection.
    
    Parameters:
    - var (int): First variable index.
    - var_1 (int): Second variable index.
    - matrix_t_l (numpy array): 4D array with shape (vars, steps, ?, x_dim) containing intersection data.
    - steps (int): Number of time steps.
    - x_vals (list or numpy array): x-coordinates for plotting.
    - save_path (str, optional): Path to save the generated plots.
    """

    x_dim = matrix_t_l.shape[3]  # Number of x-coordinates
    var_contributions = np.zeros(x_dim)
    var_1_contributions = np.zeros(x_dim)
    var_2_contributions = np.zeros(x_dim)
    var_3_contributions = np.zeros(x_dim)
    intersection_contributions = np.zeros(x_dim)
    x_vals = np.ravel(x_vals)

    # Compute contributions
    for x_index in range(x_dim):
        total_var, total_var_1, total_var_2, total_var_3,total_intersection = [], [], [],[], []

        for t in range(steps): 
            field_var = matrix_t_l[var, t, :, x_index]
            field_var_1 = matrix_t_l[var_1, t, :, x_index]
            field_var_2 = matrix_t_l[var2, t, :, x_index]
            field_var_3 = matrix_t_l[var3, t, :, x_index]
            field_val_mixed = field_var + field_var_1 + field_var_2 + field_var_3

            total_var.append(np.sum(field_var > 0)/len(field_val_mixed))
            total_var_1.append(np.sum(field_var_1 > 0)/len(field_val_mixed))
            total_var_2.append(np.sum(field_var_2 > 0)/len(field_val_mixed))
            total_var_3.append(np.sum(field_var_3 > 0)/len(field_val_mixed))
            total_intersection.append(np.sum(field_val_mixed > 3)/len(field_val_mixed))

        # Compute mean values
        var_contributions[x_index] = np.mean(total_var)
        var_1_contributions[x_index] = np.mean(total_var_1)
        var_2_contributions[x_index] = np.mean(total_var_2)
        var_3_contributions[x_index] = np.mean(total_var_3)
        intersection_contributions[x_index] = np.mean(total_intersection)

    # Apply smoothing using Gaussian filter
    smooth_var = gaussian_filter1d(var_contributions, sigma=3)
    smooth_var_1 = gaussian_filter1d(var_1_contributions, sigma=3)
    smooth_var_2 = gaussian_filter1d(var_2_contributions, sigma=3)
    smooth_var_3 = gaussian_filter1d(var_3_contributions, sigma=3)
    smooth_intersection = gaussian_filter1d(intersection_contributions, sigma=3)


    # Define Plasma colormap colors
    plasma_colors = cm.plasma(np.linspace(0, 1, 5))
    color_var = plasma_colors[0]
    color_var_1 = plasma_colors[1]
    color_var_2 = plasma_colors[3]
    color_var_3 = plasma_colors[4]
    color_intersection = plasma_colors[2]


    # Plot with soft colors and no black lines
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.fill_between(x_vals, smooth_var, color=color_var, alpha=0.4, label=f'Mode {ranks[var]} Contribution')
    ax.fill_between(x_vals, smooth_var_1, color=color_var_1, alpha=0.4, label=f'Mode {ranks[var_1]} Contribution')
    ax.fill_between(x_vals, smooth_var_2, color=color_var_2, alpha=0.4, label=f'Mode {ranks[var2]} Contribution')
    ax.fill_between(x_vals, smooth_var_3, color=color_var_3, alpha=0.4, label=f'Mode {ranks[var3]} Contribution')
    ax.fill_between(x_vals, smooth_intersection, color=color_intersection, alpha=0.6, label='Intersection')


    # Minimalist aesthetics
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('Mean Contribution', fontsize=14)
    ax.set_title(f'Contributions Modes ${ranks[var]},{ranks[var_1]},{ranks[var2]},{ranks[var3]}$ and ${ranks[var]}\cap{ranks[var_1]}\cap{ranks[var2]}\cap {ranks[var3]}$ ', fontsize=10)
    ax.set_xlim([0,5])
    ax.legend(fontsize=12, loc='upper right', frameon=False)
    
    # Remove all spines (box lines)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Remove grid
    ax.grid(False)

    # Save the figure
    plt.tight_layout()
    plt.savefig(f'{save_path}/mean_intersection_smooth_{var}_vs_{var_1}_vs_{var2}_vs_{var3}_{perc}.png', dpi=300, bbox_inches='tight')
    plt.close()
def plot_smooth_intersection_3(var, var_1,var2, perc, matrix_t_l, steps, x_vals, save_path='testing/intersections/'):
    """
    Creates a smooth line plot with shaded areas for the contributions of Var, Var_1, and their intersection.
    
    Parameters:
    - var (int): First variable index.
    - var_1 (int): Second variable index.
    - matrix_t_l (numpy array): 4D array with shape (vars, steps, ?, x_dim) containing intersection data.
    - steps (int): Number of time steps.
    - x_vals (list or numpy array): x-coordinates for plotting.
    - save_path (str, optional): Path to save the generated plots.
    """

    x_dim = matrix_t_l.shape[3]  # Number of x-coordinates
    var_contributions = np.zeros(x_dim)
    var_1_contributions = np.zeros(x_dim)
    var_2_contributions = np.zeros(x_dim)
    intersection_contributions = np.zeros(x_dim)
    x_vals = np.ravel(x_vals)

    # Compute contributions
    for x_index in range(x_dim):
        total_var, total_var_1, total_var_2,total_intersection = [], [], [],[]

        for t in range(steps): 
            field_var = matrix_t_l[var, t, :, x_index]
            field_var_1 = matrix_t_l[var_1, t, :, x_index]
            field_var_2 = matrix_t_l[var2, t, :, x_index]
            field_val_mixed = field_var + field_var_1 + field_var_2 

            total_var.append(np.sum(field_var > 0)/len(field_val_mixed))
            total_var_1.append(np.sum(field_var_1 > 0)/len(field_val_mixed))
            total_var_2.append(np.sum(field_var_2 > 0)/len(field_val_mixed))
            total_intersection.append(np.sum(field_val_mixed > 3)/len(field_val_mixed))

        # Compute mean values
        var_contributions[x_index] = np.mean(total_var)
        var_1_contributions[x_index] = np.mean(total_var_1)
        var_2_contributions[x_index] = np.mean(total_var_2)
        intersection_contributions[x_index] = np.mean(total_intersection)

    # Apply smoothing using Gaussian filter
    smooth_var = gaussian_filter1d(var_contributions, sigma=3)
    smooth_var_1 = gaussian_filter1d(var_1_contributions, sigma=3)
    smooth_var_2 = gaussian_filter1d(var_2_contributions, sigma=3)
    smooth_intersection = gaussian_filter1d(intersection_contributions, sigma=3)


    # Define Plasma colormap colors
    plasma_colors = cm.plasma(np.linspace(0, 1, 5))
    color_var = plasma_colors[0]
    color_var_1 = plasma_colors[1]
    color_var_2 = plasma_colors[3]
    color_intersection = plasma_colors[2]


    # Plot with soft colors and no black lines
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.fill_between(x_vals, smooth_var, color=color_var, alpha=0.4, label=f'Mode {ranks[var]} Contribution')
    ax.fill_between(x_vals, smooth_var_1, color=color_var_1, alpha=0.4, label=f'Mode {ranks[var_1]} Contribution')
    ax.fill_between(x_vals, smooth_var_2, color=color_var_2, alpha=0.4, label=f'Mode {ranks[var2]} Contribution')
    ax.fill_between(x_vals, smooth_intersection, color=color_intersection, alpha=0.6, label='Intersection')


    # Minimalist aesthetics
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('Mean Contribution', fontsize=14)
    ax.set_title(f'Contributions Modes ${ranks[var]},{ranks[var_1]},{ranks[var2]}$ and ${ranks[var]}\cap{ranks[var_1]}\cap{ranks[var2]}$ ', fontsize=10)
    ax.set_xlim([0,5])
    ax.legend(fontsize=12, loc='upper right', frameon=False)
    
    # Remove all spines (box lines)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Remove grid
    ax.grid(False)

    # Save the figure
    plt.tight_layout()
    plt.savefig(f'{save_path}/mean_intersection_smooth_{var}_vs_{var_1}_vs_{var2}_{perc}.png', dpi=300, bbox_inches='tight')
    plt.close()
# Example use

def plot_2d_shap_fields(var, var_1,perc, matrix_t_l, steps, x_vals, y_vals, save_path='testing/intersections/'):
    """
    Plots the 2D spatial field with SHAP values for two latent vectors over time using Plasma colormap.

    Parameters:
    - var (int): First variable index.
    - var_1 (int): Second variable index.
    - matrix_t_l (numpy array): 4D array (vars, steps, y_dim, x_dim) containing SHAP data.
    - steps (int): Number of time steps.
    - x_vals (numpy array): x-coordinates for the spatial field.
    - y_vals (numpy array): y-coordinates for the spatial field.
    - save_path (str, optional): Path to save the generated plots.
    """

    # Define Magma colormap
    magma_colors = cm.magma(np.linspace(0, 1, 9))
    color_var = magma_colors[5]  # Mode var (dark purple)
    color_var_1 = magma_colors[7]  # Mode var_1 (orange-yellow)
    color_intersection = magma_colors[8]  # Intersection (bright yellow)

    os.makedirs(save_path, exist_ok=True)  # Ensure save path exists

    # Define the legend handles
    legend_elements = [
        Line2D([0], [0], color=color_var, lw=2, label=f'Mode ${ranks[var]}$'),
        Line2D([0], [0], color=color_var_1, lw=2, label=f'Mode ${ranks[var_1]}$'),
        Line2D([0], [0], color=color_intersection, lw=2, label=f'${ranks[var]}\cap{ranks[var_1]}$')
    ]

    for t in range(10):
        # Extract SHAP values at time `t`
        field_var = matrix_t_l[var, t*100, :, :]
        field_var_1 = matrix_t_l[var_1, t*100, :, :]

        # Normalize fields for color mapping
        norm_var = field_var / np.max(field_var) if np.max(field_var) > 0 else field_var
        norm_var_1 = field_var_1 / np.max(field_var_1) if np.max(field_var_1) > 0 else field_var_1
        norm_intersection = np.minimum(field_var, field_var_1) / np.max(np.minimum(field_var, field_var_1)) if np.max(np.minimum(field_var, field_var_1)) > 0 else np.minimum(field_var, field_var_1)

        # Create color image
        img = np.zeros((*field_var.shape, 3))
        img[..., 0] = norm_var * color_var[0] + norm_var_1 * color_var_1[0] + norm_intersection * color_intersection[0]
        img[..., 1] = norm_var * color_var[1] + norm_var_1 * color_var_1[1] + norm_intersection * color_intersection[1]
        img[..., 2] = norm_var * color_var[2] + norm_var_1 * color_var_1[2] + norm_intersection * color_intersection[2]

        # Set up the figure
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(img, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower')

        # Add colorbar
        # cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        # cbar.set_label('SHAP Intensity', fontsize=12)

        # Overlay obstacle
        xb = np.array([-0.125, -0.125, 0.25, 0.25])  # Obstacle x-coordinates
        yb = np.array([0.0, 1.0, 1.0, 0.0])  # Obstacle y-coordinates
        ax.fill(xb, yb, 'gray', alpha=0.5)  # No legend, just visual

        # Style adjustments
        ax.set_xlabel('$x$', fontsize=14)
        ax.set_ylabel('$y$', fontsize=14)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_yticks(np.linspace(y_vals.min(), y_vals.max(), 5))
        ax.set_title(f'SHAP Fields at t={ t*100}', fontsize=10)

        # Add legend
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

        # Save the plot
        plt.tight_layout()
        plt.savefig(f'{save_path}/shap_2d_field_t{t}_{perc}.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_2d_fields( u_keras, u_v, steps, x_vals, y_vals, save_path='testing/intersections/'):
    """
    Plots the 2D spatial field with SHAP values for two latent vectors over time using Plasma colormap.

    Parameters:
    - var (int): First variable index.
    - var_1 (int): Second variable index.
    - matrix_t_l (numpy array): 4D array (vars, steps, y_dim, x_dim) containing SHAP data.
    - steps (int): Number of time steps.
    - x_vals (numpy array): x-coordinates for the spatial field.
    - y_vals (numpy array): y-coordinates for the spatial field.
    - save_path (str, optional): Path to save the generated plots.
    """

    # Define Magma colormap
    magma_colors = cm.magma(np.linspace(0, 1, 9))
    color_var = magma_colors[5]  # Mode var (dark purple)
    color_var_1 = magma_colors[7]  # Mode var_1 (orange-yellow)
    color_intersection = magma_colors[8]  # Intersection (bright yellow)

    os.makedirs(save_path, exist_ok=True)  # Ensure save path exists

    frames = []
    for t in range(8):
        # Extract SHAP values at time `t`
        field_var = u_keras[t*steps,:,:]

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(field_var, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower',cmap='plasma')

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04,  shrink=0.7)       

        # Overlay obstacle
        xb = np.array([-0.125, -0.125, 0.25, 0.25])  # Obstacle x-coordinates
        yb = np.array([0.0, 1.0, 1.0, 0.0])  # Obstacle y-coordinates
        ax.fill(xb, yb, 'white')  # No legend, just visual

        # Style adjustments
        ax.set_xlabel('$x$', fontsize=14)
        ax.set_ylabel('$y$', fontsize=14)
        ax.set_xticks(np.linspace(0, x_vals.max(), 5))
        ax.set_yticks(np.linspace(y_vals.min(), y_vals.max(), 5))
        ax.set_title(f'$u$ at t={t*steps}', fontsize=10)

        # Add legend
        # Save the plot
        plt.tight_layout()
        plt.savefig(f'{save_path}/2d_field_t{t}_{perc}.png', dpi=300, bbox_inches='tight')
        plt.close()
    for t in range(steps_1):
        # Extract SHAP values at time t (adjust if your time indexing differs)
        field_var = u_keras[t, :, :]

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot the field using imshow with the desired extent
        im = ax.imshow(field_var, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], 
                    origin='lower', cmap='plasma')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04, shrink=0.7)
        
        # Overlay obstacle (filled with white)
        ax.fill(xb, yb, 'white')
        
        # Style adjustments
        ax.set_xlabel('$x$', fontsize=14)
        ax.set_ylabel('$y$', fontsize=14)
        ax.set_xticks(np.linspace(0, x_vals.max(), 5))
        ax.set_yticks(np.linspace(y_vals.min(), y_vals.max(), 5))
        ax.set_title(f'$u$ at t={t}', fontsize=10)
        
        # Render the figure canvas and capture as a numpy array
        fig.canvas.draw()
        ncols, nrows = fig.canvas.get_width_height()
        # Get ARGB buffer
        buf = fig.canvas.tostring_argb()
        # Convert to a NumPy array with shape (nrows, ncols, 4)
        buf = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 4)
        # Rearrange from ARGB to RGBA: channels [A, R, G, B] -> [R, G, B, A]
        buf = buf[:, :, [1, 2, 3, 0]]
        # Take only the RGB channels (drop alpha)
        frame = buf[:, :, :3]
        frames.append(frame)
        
        plt.close(fig)
    # Now, create a video from the frames (adjust fps as needed)
    video_filename = os.path.join(save_path, 'video_10.mp4')
    imageio.mimwrite(video_filename, frames, fps=10, format='ffmpeg')
    print("Video saved as:", video_filename)
    print('stats')
    field_u2 = np.mean(u_keras**2, axis=0)
    field_uv = np.mean(u_keras*u_v, axis=0)
    field_v2 = np.mean(u_v**2, axis=0)
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(field_u2, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower',cmap='plasma')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04,  shrink=0.7)    # cbar.set_label('SHAP Intensity', fontsize=8)

    # Overlay obstacle
    xb = np.array([-0.125, -0.125, 0.25, 0.25])  # Obstacle x-coordinates
    yb = np.array([0.0, 1.0, 1.0, 0.0])  # Obstacle y-coordinates
    ax.fill(xb, yb, 'white')  # No legend, just visual

    # Style adjustments
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$y$', fontsize=14)
    ax.set_xticks(np.linspace(0, x_vals.max(), 5))
    ax.set_yticks(np.linspace(y_vals.min(), y_vals.max(), 5))
    ax.set_title('$\overline{u^2}$ Field', fontsize=10)

    # Add legend
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{save_path}/2d_field_u2_{perc}.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(field_uv, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower',cmap='plasma')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04,  shrink=0.7)    # cbar.set_label('SHAP Intensity', fontsize=12)

    # Overlay obstacle
    xb = np.array([-0.125, -0.125, 0.25, 0.25])  # Obstacle x-coordinates
    yb = np.array([0.0, 1.0, 1.0, 0.0])  # Obstacle y-coordinates
    ax.fill(xb, yb, 'white')  # No legend, just visual

    # Style adjustments
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$y$', fontsize=14)
    ax.set_xticks(np.linspace(0, x_vals.max(), 5))
    ax.set_yticks(np.linspace(y_vals.min(), y_vals.max(), 5))
    ax.set_title('$\overline{uv}$ Field', fontsize=10)

    # Add legend
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{save_path}/2d_field_uv_{perc}.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(field_v2, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower',cmap='plasma')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04,  shrink=0.7)    # cbar.set_label('SHAP Intensity', fontsize=12)

    # Overlay obstacle
    xb = np.array([-0.125, -0.125, 0.25, 0.25])  # Obstacle x-coordinates
    yb = np.array([0.0, 1.0, 1.0, 0.0])  # Obstacle y-coordinates
    ax.fill(xb, yb, 'white')  # No legend, just visual

    # Style adjustments
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$y$', fontsize=14)
    ax.set_xticks(np.linspace(0, x_vals.max(), 5))
    ax.set_yticks(np.linspace(y_vals.min(), y_vals.max(), 5))
    ax.set_title('$\overline{v^2}$ Field', fontsize=10)

    # Add legend
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{save_path}/2d_field_v2_{perc}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_super(var, var_1,perc, matrix_t_l,u_keras, steps, x_vals, y_vals, save_path='testing/intersections/'):
    """
    Plots the 2D spatial field with SHAP values for two latent vectors over time using Plasma colormap.

    Parameters:
    - var (int): First variable index.
    - var_1 (int): Second variable index.
    - matrix_t_l (numpy array): 4D array (vars, steps, y_dim, x_dim) containing SHAP data.
    - steps (int): Number of time steps.
    - x_vals (numpy array): x-coordinates for the spatial field.
    - y_vals (numpy array): y-coordinates for the spatial field.
    - save_path (str, optional): Path to save the generated plots.
    """

    # Define Magma colormap
    magma_colors = cm.magma(np.linspace(0, 1, 9))
    color_var = magma_colors[5]  # Mode var (dark purple)
    color_var_1 = magma_colors[7]  # Mode var_1 (orange-yellow)
    color_intersection = magma_colors[2]  # Intersection (bright yellow)

    os.makedirs(save_path, exist_ok=True)  # Ensure save path exists

    # Define the legend handles
    legend_elements = [
        Line2D([0], [0], color=color_var, lw=2, label=f'Mode ${ranks[var]}$'),
        Line2D([0], [0], color=color_var_1, lw=2, label=f'Mode ${ranks[var_1]}$'),
        Line2D([0], [0], color=color_intersection, lw=2, label=f'${ranks[var]}\cap{ranks[var_1]}$')
    ]

    for t in range(8):
        # Extract SHAP values at time `t`
        # Extract SHAP values at time `t`
        field_var = matrix_t_l[var, t*steps, :, :]
        field_var_1 = matrix_t_l[var_1, t*steps, :, :]
        original = u_keras[t*steps,:,:]

        # Normalize fields for color mapping
        norm_var = field_var / np.max(field_var) if np.max(field_var) > 0 else field_var
        norm_var_1 = field_var_1 / np.max(field_var_1) if np.max(field_var_1) > 0 else field_var_1
        norm_intersection = np.minimum(field_var, field_var_1) / np.max(np.minimum(field_var, field_var_1)) if np.max(np.minimum(field_var, field_var_1)) > 0 else np.minimum(field_var, field_var_1)

        # Create color image
        img = np.zeros((*field_var.shape, 3))
        img[..., 0] = norm_var * color_var[0] + norm_var_1 * color_var_1[0] + norm_intersection * color_intersection[0]
        img[..., 1] = norm_var * color_var[1] + norm_var_1 * color_var_1[1] + norm_intersection * color_intersection[1]
        img[..., 2] = norm_var * color_var[2] + norm_var_1 * color_var_1[2] + norm_intersection * color_intersection[2]

        # Set up the figure
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(original,cmap='Greys' ,extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower')
        img[img==  0]=np.nan
        im = ax.imshow(img, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower',alpha=0.7)
        # Add colorbar
        # cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        # cbar.set_label('SHAP Intensity', fontsize=12)

        # Overlay obstacle
        xb = np.array([-0.125, -0.125, 0.125, 0.125])  # Obstacle x-coordinates
        yb = np.array([0.0, 1.0, 1.0, 0.0])  # Obstacle y-coordinates
        ax.fill(xb, yb, 'white')  # No legend, just visual

        # Style adjustments
        ax.set_xlabel('$x$', fontsize=14)
        ax.set_ylabel('$y$', fontsize=14)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_yticks(np.linspace(y_vals.min(), y_vals.max(), 5))
        # ax.set_title(f'Q events classifaction in SHAP values for Modes {ranks[var]} and  {ranks[var_1]} at t={t*steps}', fontsize=10)

        # Add legend
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        # Save the plot
        plt.tight_layout()
        plt.savefig(f'{save_path}/super_{t}_{perc}.png', dpi=300, bbox_inches='tight')
        plt.close()


import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

def plot_super_3(var, var_1, var_2, perc, matrix_t_l, u_keras, steps, x_vals, y_vals, save_path='testing/intersections/'):
    """
    Improved 2D spatial field plotting with SHAP values for three latent vectors over time.
    Uses distinct colormap overlays for clarity.
    """
    os.makedirs(save_path, exist_ok=True)  # Ensure save path exists
    
    # Define distinct colors
    color_var = 'blue'  # Mode var
    color_var_1 = 'red'  # Mode var_1
    color_var_2 = 'green'  # Mode var_2
    color_intersection = 'white'  # Intersection color for visibility
    
    # Legend handles
    legend_elements = [
        Line2D([0], [0], color=color_var, lw=2, label=f'Mode {var}'),
        Line2D([0], [0], color=color_var_1, lw=2, label=f'Mode {var_1}'),
        Line2D([0], [0], color=color_var_2, lw=2, label=f'Mode {var_2}'),
        Line2D([0], [0], color=color_intersection, lw=2, linestyle='--', label='Intersection')
    ]
    
    for t in range(steps):
        # Extract SHAP values at time `t`
        field_var = matrix_t_l[var, t, :, :]
        field_var_1 = matrix_t_l[var_1, t, :, :]
        field_var_2 = matrix_t_l[var_2, t, :, :]
        original = u_keras[t, :, :]
        
        # Normalize fields
        max_val = max(np.max(field_var), np.max(field_var_1), np.max(field_var_2), 1e-6)
        norm_var = field_var / max_val
        norm_var_1 = field_var_1 / max_val
        norm_var_2 = field_var_2 / max_val
        
        # Compute intersection
        intersection = (field_var > 0) & (field_var_1 > 0) & (field_var_2 > 0)
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(original, cmap='Greys', extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower')
        
        # Overlay color maps for each mode
        ax.imshow(norm_var, cmap='Blues', alpha=0.5, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower')
        ax.imshow(norm_var_1, cmap='Reds', alpha=0.5, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower')
        ax.imshow(norm_var_2, cmap='Greens', alpha=0.5, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower')
        
        # Overlay intersection as contours
        ax.contour(intersection, colors=color_intersection, linewidths=1.5, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()])
        
        # Overlay obstacle
        xb = np.array([-0.125, -0.125, 0.25, 0.25])
        yb = np.array([0.0, 1.0, 1.0, 0.0])
        ax.fill(xb, yb, 'gray', alpha=0.5)
        
        # Style adjustments
        ax.set_xlabel('$x$', fontsize=14)
        ax.set_ylabel('$y$', fontsize=14)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_yticks(np.linspace(y_vals.min(), y_vals.max(), 5))
        ax.set_title(f'SHAP Fields at t={t}', fontsize=10)
        
        # Add legend
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(f'{save_path}/super_3_{t}_{perc}.png', dpi=300, bbox_inches='tight')
        plt.close()



def classify_events(u_keras, u_v):
    """
    Classifies events based on conditions:
    - 0: Back Up (u_keras < 0, u_v < 0)
    - 1: Back Down (u_keras < 0, u_v > 0)
    - 2: Forward Up (u_keras > 0, u_v > 0)
    - 3: Forward Down (u_keras > 0, u_v < 0)
    """
    event_map = np.zeros_like(u_keras, dtype=int)
    event_map[(u_keras < 0) & (u_v < 0)] = 3  # Back DOwn
    event_map[(u_keras < 0) & (u_v > 0)] = 2  # Back Up
    event_map[(u_keras > 0) & (u_v > 0)] = 1  # Forward Up
    event_map[(u_keras > 0) & (u_v < 0)] = 4  # Forward Down
    return event_map

def plot_event_overlay(var, var_1, u, u_keras, u_v, steps,x_vals, y_vals, save_path='testing/intersections/'):
    os.makedirs(save_path, exist_ok=True)

    # Define a color palette for the events
    color_map = {
        0: (0.6, 0.6, 1.0),  # Back DOwn - Soft Blue
        1: (0.6, 1.0, 0.6),  # Back Up - Soft Green
        2: (1.0, 0.6, 1.0),  # Forward Up - Soft Purple
        3: (1.0, 1.0, 0.6),  # Forward Down - Soft Yellow
    }

    for t in range(8):
        # Extract SHAP values at time `t`
        field_var = classify_events(u_keras[t*steps, var, :, :], u_v[t*steps, var, :, :])
        print('U',u_keras)
        field_var_1 = classify_events(u_keras[t*steps, var_1, :, :], u_v[t*steps, var_1, :, :])
        original = u[t*steps, :, :]

        # Debugging prints
        print(f'Time {t}: Unique event values in field_var:', np.unique(field_var), field_var[:10,:10])
        print(f'Time {t}: Unique event values in field_var_1:', np.unique(field_var_1))
        if t==0:
            print('classify',field_var)
        # Create color image
        img = np.zeros((*field_var.shape, 3))  # RGB channels only

        for event in range(1,5):
            mask = (field_var == event) | (field_var_1 == event)
            img[mask, :] = color_map[event-1]  # Assign correct color

        # Debugging check: If no color is applied, print warning
        if np.all(img == 0):
            print(f"Warning: No classification applied at t={t}!")

        # Set up the figure
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(original,cmap='Greys' ,extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower')
        img[img==  0]=np.nan
        # Plot classification overlay
        im = ax.imshow(img, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower', alpha=0.7)

        # Overlay obstacle
        xb = np.array([-0.125, -0.125, 0.25, 0.25])  # Obstacle x-coordinates
        yb = np.array([0.0, 1.0, 1.0, 0.0])  # Obstacle y-coordinates
        ax.fill(xb, yb, 'white')

        # Style adjustments
        ax.set_xlabel('$x$', fontsize=14)
        ax.set_ylabel('$y$', fontsize=14)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_yticks(np.linspace(y_vals.min(), y_vals.max(), 5))
        ax.set_title(f'Q events classifaction in SHAP regions for Modes {ranks[var]} and  {ranks[var_1]} at t={t*steps}', fontsize=10)

        # Define the legend handles
        legend_elements = [
            Line2D([0], [0], color=color_map[0], lw=2, label='Q3'),
            Line2D([0], [0], color=color_map[1], lw=2, label='Q2'),
            Line2D([0], [0], color=color_map[2], lw=2, label='Q1'),
            Line2D([0], [0], color=color_map[3], lw=2, label='Q4'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        # Save the plot
        plt.tight_layout()
        plt.savefig(f'{save_path}/overlay_{t}.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_event_overlay_4(var, var_1,var2,var3, u, u_keras, u_v, x_vals, y_vals, save_path='testing/intersections/'):
    os.makedirs(save_path, exist_ok=True)

    # Define a color palette for the events
    color_map = {
        0: (0.6, 0.6, 1.0),  # Back Up - Soft Blue
        1: (0.6, 1.0, 0.6),  # Back Down - Soft Green
        2: (1.0, 0.6, 1.0),  # Forward Up - Soft Purple
        3: (1.0, 1.0, 0.6),  # Forward Down - Soft Yellow
    }

    for t in range(10):
        # Extract SHAP values at time `t`
        field_var = classify_events(u_keras[t, var, :, :], u_v[t, var, :, :])
        print('U',u_keras)
        field_var_1 = classify_events(u_keras[t, var_1, :, :], u_v[t, var_1, :, :])
        field_var_2 = classify_events(u_keras[t, var2, :, :], u_v[t, var2, :, :])
        field_var_3 = classify_events(u_keras[t, var3, :, :], u_v[t, var3, :, :])

        original = u[t, :, :]

        # Debugging prints
        print(f'Time {t}: Unique event values in field_var:', np.unique(field_var), field_var[:10,:10])
        print(f'Time {t}: Unique event values in field_var_1:', np.unique(field_var_1))
        if t==0:
            print('classify',field_var)
        # Create color image
        img = np.zeros((*field_var.shape, 5))  # RGB channels only

        for event in range(1,5):
            mask = (field_var == event) | (field_var_1 == event) | (field_var_2 == event) | (field_var_3 == event)
            img[mask, :] = color_map[event-1]  # Assign correct color

        # Debugging check: If no color is applied, print warning
        if np.all(img == 0):
            print(f"Warning: No classification applied at t={t}!")

        # Set up the figure
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(original,cmap='Greys' ,extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower')

        # Plot classification overlay
        ax.imshow(img, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower', alpha=0.7)

        # Overlay obstacle
        xb = np.array([-0.125, -0.125, 0.25, 0.25])  # Obstacle x-coordinates
        yb = np.array([0.0, 1.0, 1.0, 0.0])  # Obstacle y-coordinates
        ax.fill(xb, yb, 'white')

        # Style adjustments
        ax.set_xlabel('$x$', fontsize=14)
        ax.set_ylabel('$y$', fontsize=14)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_yticks(np.linspace(y_vals.min(), y_vals.max(), 5))
        ax.set_title(f'Event Classification at t={t}', fontsize=10)

        # Define the legend handles
        # Define the legend handles
        legend_elements = [
            Line2D([0], [0], color=color_map[0], lw=2, label='Q3'),
            Line2D([0], [0], color=color_map[1], lw=2, label='Q2'),
            Line2D([0], [0], color=color_map[2], lw=2, label='Q1'),
            Line2D([0], [0], color=color_map[3], lw=2, label='Q4'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        # Save the plot
        plt.tight_layout()
        plt.savefig(f'{save_path}/overlay_{t}_{var_1}_{var}_{var2}_{var3}.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_event_contributions_on_intersection(var, var_1, matrix_t_l, steps, x_vals, y_vals, u_keras, u_v, save_path='testing/intersections/'):
    """
    Creates a smooth line plot of event contributions in the intersection subdomain based on matrix_t_l equality.
    
    Parameters:
    - var (int): First variable index.
    - var_1 (int): Second variable index.
    - matrix_t_l (numpy array): 4D array with shape (vars, steps, ?, x_dim) containing intersection data.
    - steps (int): Number of time steps.
    - x_vals (list or numpy array): x-coordinates for plotting.
    - y_vals (list or numpy array): y-coordinates for plotting.
    - u_keras (numpy array): Data for classification (u_keras).
    - u_v (numpy array): Data for classification (u_v).
    - save_path (str, optional): Path to save the generated plots.
    """
    x_dim = matrix_t_l.shape[3]  # Number of x-coordinates
    y_dim = matrix_t_l.shape[2]  # Number of y-coordinates
    event_contributions = {1: np.zeros(x_dim), 2: np.zeros(x_dim), 3: np.zeros(x_dim), 4: np.zeros(x_dim)}

    x_vals = np.ravel(x_vals)

    # Loop over each x-coordinate and compute event contributions in the intersection
    for x_index in range(x_dim):
        total_event_contributions = {1: [], 2: [], 3: [], 4: []}

        for t in range(steps): 
            field_var = matrix_t_l[var, t, :, x_index]
            field_var_1 = matrix_t_l[var_1, t, :, x_index]

            # Compute intersection subdomain where matrix_t_l values are equal
            intersection_mask = (field_var == field_var_1)

            # Classify events for both variables in the intersection subdomain
            classified_var = classify_events(u_keras[t, var, :, x_index], u_v[t, var, :, x_index])
            classified_var_1 = classify_events(u_keras[t, var_1, :, x_index], u_v[t, var_1, :, x_index])

            # Track the events that occur within the intersection subdomain
            for event in range(1,5):  # 0 = Back Up, 1 = Back Down, 2 = Forward Up, 3 = Forward Down
                # Count how many times the event occurs in the intersection subdomain
                total_event_contributions[event].append(np.sum((classified_var == event) & (classified_var_1 == event) & intersection_mask) / np.sum(intersection_mask))

        # Store the average contributions across time steps
        for event in range(1,5):
            event_contributions[event][x_index] = np.mean(total_event_contributions[event])

    # Apply smoothing using Gaussian filter
    smooth_event_contributions = {event: gaussian_filter1d(contrib, sigma=3) for event, contrib in event_contributions.items()}

    # Define Plasma colormap colors for event contributions
    colors = {
        1: (0.6, 0.6, 1.0),  # Back Up - Soft Blue
        2: (0.6, 1.0, 0.6),  # Back Down - Soft Green
        3: (1.0, 0.6, 1.0),  # Forward Up - Soft Purple
        4: (1.0, 1.0, 0.6),  # Forward Down - Soft Yellow
    }
    # colors = {event: plasma_colors[event-1] for event in range(1,5)}

    # Plot with soft colors and no black lines
    fig, ax = plt.subplots(figsize=(12, 6))
    legends = ['Q3', 'Q2', 'Q1','Q4']
    # Plot each event contribution with different colors
    for event in range(1,5):
        ax.fill_between(x_vals, smooth_event_contributions[event], color=colors[event], alpha=0.4, label=f'{legends[event-1]}')

    # Minimalist aesthetics
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('Event Contribution', fontsize=14)
    ax.set_title(f'Event Contributions on Intersection Subdomain for Modes {ranks[var]} and {ranks[var_1]}', fontsize=10)
    ax.set_xlim([x_vals.min(), x_vals.max()])
    ax.legend(fontsize=12, loc='upper right', frameon=False)

    # Remove all spines (box lines)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Remove grid
    ax.grid(False)

    # Save the figure
    plt.tight_layout()
    plt.savefig(f'{save_path}/event_contributions_intersection_{var}_vs_{var_1}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_event_contributions_on_intersection_4(var, var_1, var2,var3, matrix_t_l, steps, x_vals, y_vals, u_keras, u_v, save_path='testing/intersections/'):
    """
    Creates a smooth line plot of event contributions in the intersection subdomain based on matrix_t_l equality.
    
    Parameters:
    - var (int): First variable index.
    - var_1 (int): Second variable index.
    - matrix_t_l (numpy array): 4D array with shape (vars, steps, ?, x_dim) containing intersection data.
    - steps (int): Number of time steps.
    - x_vals (list or numpy array): x-coordinates for plotting.
    - y_vals (list or numpy array): y-coordinates for plotting.
    - u_keras (numpy array): Data for classification (u_keras).
    - u_v (numpy array): Data for classification (u_v).
    - save_path (str, optional): Path to save the generated plots.
    """
    x_dim = matrix_t_l.shape[3]  # Number of x-coordinates
    y_dim = matrix_t_l.shape[2]  # Number of y-coordinates
    event_contributions = {1: np.zeros(x_dim), 2: np.zeros(x_dim), 3: np.zeros(x_dim), 4: np.zeros(x_dim)}

    x_vals = np.ravel(x_vals)

    # Loop over each x-coordinate and compute event contributions in the intersection
    for x_index in range(x_dim):
        total_event_contributions = {1: [], 2: [], 3: [], 4: []}

        for t in range(steps): 
            field_var = matrix_t_l[var, t, :, x_index]
            field_var_1 = matrix_t_l[var_1, t, :, x_index]
            field_var_2 = matrix_t_l[var2, t, :, x_index]
            field_var_3 = matrix_t_l[var3, t, :, x_index]


            # Compute intersection subdomain where matrix_t_l values are equal
            intersection_mask = (field_var == field_var_1) & (field_var == field_var_2) & (field_var == field_var_3)

            # Classify events for both variables in the intersection subdomain
            classified_var = classify_events(u_keras[t, var, :, x_index], u_v[t, var, :, x_index])
            classified_var_1 = classify_events(u_keras[t, var_1, :, x_index], u_v[t, var_1, :, x_index])
            classified_var_2 = classify_events(u_keras[t, var2, :, x_index], u_v[t, var2, :, x_index])
            classified_var_3 = classify_events(u_keras[t, var3, :, x_index], u_v[t, var3, :, x_index])


            # Track the events that occur within the intersection subdomain
            for event in range(1,5):  # 0 = Back Up, 1 = Back Down, 2 = Forward Up, 3 = Forward Down
                # Count how many times the event occurs in the intersection subdomain
                total_event_contributions[event].append(np.sum((classified_var == event) & (classified_var_1 == event) & (classified_var_2 == event) & (classified_var_3 == event) & intersection_mask) / np.sum(intersection_mask))

        # Store the average contributions across time steps
        for event in range(1,5):
            event_contributions[event][x_index] = np.mean(total_event_contributions[event])

    # Apply smoothing using Gaussian filter
    smooth_event_contributions = {event: gaussian_filter1d(contrib, sigma=3) for event, contrib in event_contributions.items()}

    # Define Plasma colormap colors for event contributions
    plasma_colors = cm.plasma(np.linspace(0, 1, 4))
    colors = {
        1: (0.6, 0.6, 1.0),  # Back Up - Soft Blue
        2: (0.6, 1.0, 0.6),  # Back Down - Soft Green
        3: (1.0, 0.6, 1.0),  # Forward Up - Soft Purple
        4: (1.0, 1.0, 0.6),  # Forward Down - Soft Yellow
    }

    # Plot with soft colors and no black lines
    fig, ax = plt.subplots(figsize=(12, 6))
    legends = ['Q3', 'Q2', 'Q1','Q4']
    # Plot each event contribution with different colors
    for event in range(1,5):
        ax.fill_between(x_vals, smooth_event_contributions[event], color=colors[event], alpha=0.4, label=f'{legends[event-1]}')

    # Minimalist aesthetics
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('Event Contribution', fontsize=14)
    ax.set_title(f'Event Contributions on ${ranks[var]}\cap{ranks[var_1]}\cap{ranks[var2]}\cap {ranks[var3]}$', fontsize=10)
    ax.set_xlim([x_vals.min(), x_vals.max()])
    ax.legend(fontsize=12, loc='upper right', frameon=False)

    # Remove all spines (box lines)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Remove grid
    ax.grid(False)

    # Save the figure
    plt.tight_layout()
    plt.savefig(f'{save_path}/event_contributions_intersection_{var}_vs_{var_1}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_temporal_event_contributions(var, var_1, matrix_t_l, steps, x_vals,y_vals, u_keras, u_v, save_path='testing/intersections/'):
    """
    Plots the temporal evolution of event contributions in the intersection subdomain 
    for two selected variables.

    Parameters:
    - var (int): Index of the first variable.
    - var_1 (int): Index of the second variable.
    - matrix_t_l (numpy array): 4D array with shape (vars, steps, ?, x_dim) containing intersection data.
    - steps (int): Number of time steps.
    - x_vals (numpy array): x-coordinates for plotting.
    - u_keras (numpy array): Data for classification (u_keras).
    - u_v (numpy array): Data for classification (u_v).
    - save_path (str, optional): Path to save the generated plot.
    """
    
    x_dim = matrix_t_l.shape[3]  # Number of x-coordinates
    event_contributions = {1: np.zeros((steps, x_dim)), 
                           2: np.zeros((steps, x_dim)), 
                           3: np.zeros((steps, x_dim)), 
                           4: np.zeros((steps, x_dim))}

    x_vals = np.ravel(x_vals)  # Flatten x-coordinates
    print('VELuOS',u_keras.shape)
    for x_index in range(x_dim):
        for t in range(steps): 
            field_var = matrix_t_l[var, t, :, x_index]
            field_var_1 = matrix_t_l[var_1, t, :, x_index]

            # Compute intersection mask where matrix_t_l values are equal
            intersection_mask = (field_var == field_var_1)

            # Classify events at the intersection subdomain
            classified_var = classify_events(u_keras[t, var, :, x_index], u_v[t, var, :, x_index])
            classified_var_1 = classify_events(u_keras[t, var_1, :, x_index], u_v[t, var_1, :, x_index])

            # Count occurrences of each event in the intersection zone
            for event in range(1, 5):  # 1 = Q4, 2 = Q2, 3 = Q1, 4 = Q3
                event_count = np.sum((classified_var == event) & (classified_var_1 == event) & intersection_mask)
                if np.sum(intersection_mask) > 0:
                    event_contributions[event][t, x_index] = event_count / np.sum(intersection_mask)
                else:
                    event_contributions[event][t, x_index] = 0

    # Select specific time steps for plotting
    time_indices = np.linspace(0, steps - 1, 5, dtype=int)  # Select 4 time instants
    time_colors = ['b', 'g', 'r', 'm']  # Colors for different time instants

    # Define colors for each event
    event_colors = {
        1: 'blue',   # Q4
        2: 'green',  # Q2
        3: 'red',    # Q1
        4: 'purple'  # Q3
    }

    # Plot event contributions over time
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for event in range(1, 5):  # Loop over events (Q4, Q2, Q1, Q3)
        for i, t in enumerate(time_indices):
            ax.plot(x_vals, event_contributions[event][t, :], color=event_colors[event], 
                    linestyle='-', alpha=0.7, label=f'Q{event} (t={t})' if i == 0 else "")

    # Minimalist aesthetics
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('Event Contribution', fontsize=14)
    ax.set_title(f'Temporal Evolution of Event Contributions in Intersection Subdomain for Modes {var} and {var_1}', fontsize=10)
    ax.set_xlim([x_vals.min(), x_vals.max()])
    ax.legend(fontsize=10, loc='upper right', frameon=False)

    # Remove spines and grid
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    # Save the figure
    plt.tight_layout()
    plt.savefig(f'{save_path}/event_contributions_temporal_{var}_vs_{var_1}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_temporal_dominant_events(var, var_1, matrix_t_l, steps, x_vals, u_keras, u_v, save_path='testing/intersections/'):
    x_dim = matrix_t_l.shape[3]  # Number of x-coordinates
    event_colors = plt.cm.plasma(np.linspace(0, 1, steps // 100))  # Color transition from purple to red
    
    # Quadrant mapping based on event classification
    quadrant_mapping = {3: (1, 1), 4: (1, -1), 2: (-1, 1), 1: (-1, -1)}
    
    # Prepare lists for storing event data
    dominant_data, second_data = [], []
    
    for t in range(0, steps, 10):  # Sampling every 10 steps
        event_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        intersection_size = 0
        
        for x_index in range(x_dim):
            field_var = matrix_t_l[var, t, :, x_index]
            field_var_1 = matrix_t_l[var_1, t, :, x_index]
            intersection_mask = (field_var == field_var_1)
            
            if np.sum(intersection_mask) == 0:
                continue
            
            classified_var = classify_events(u_keras[t, var, :, x_index], u_v[t, var, :, x_index])
            classified_var_1 = classify_events(u_keras[t, var_1, :, x_index], u_v[t, var_1, :, x_index])
            
            for event in range(1, 5):
                event_counts[event] += np.sum((classified_var == event) & (classified_var_1 == event) & intersection_mask)
            
            intersection_size += np.sum(intersection_mask)
        
        sorted_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_events) < 2 or intersection_size == 0:
            continue
        
        dominant_event, dominant_count = sorted_events[0]
        second_event, second_count = sorted_events[1]
        
        dominant_data.append((dominant_count, intersection_size, dominant_event, t))
        second_data.append((second_count, intersection_size, second_event, t))
    
    # Normalize values
    def normalize_data(data):
        x, y, events, time = zip(*data)
        max_x = max(x) if max(x) != 0 else 1
        max_y = max(y) if max(y) != 0 else 1
        return np.array(x) / max_x, np.array(y) / max_y, events, time
    
    dominant_x, dominant_y, dominant_quadrants, dominant_time = normalize_data(dominant_data)
    second_x, second_y, second_quadrants, second_time = normalize_data(second_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    titles = ['Dominant Event Evolution', 'Second Dominant Event Evolution']
    data_sets = [(dominant_x, dominant_y, dominant_quadrants, dominant_time), (second_x, second_y, second_quadrants, second_time)]
    
    for ax, (x_vals, y_vals, quadrants, times), title in zip(axes, data_sets, titles):
        for i, (x, y, event, t) in enumerate(zip(x_vals, y_vals, quadrants, times)):
            qx, qy = quadrant_mapping[event]
            ax.scatter(qx * x, qy * y, color=event_colors[i % len(event_colors)], alpha=0.7, label=f"t={t}")
        
        ax.set_xlabel('Normalized Number of Appearances', fontsize=12)
        ax.set_ylabel('Normalized Intersection Size', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        # Add quadrant labels
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)
        ax.text(0.1, 0.1, 'Q3', fontsize=12, ha='center', va='center', transform=ax.transAxes)
        ax.text(0.1, 0.9, 'Q2', fontsize=12, ha='center', va='center', transform=ax.transAxes)
        ax.text(0.9, 0.9, 'Q1', fontsize=12, ha='center', va='center', transform=ax.transAxes)
        ax.text(0.9, 0.1, 'Q4', fontsize=12, ha='center', va='center', transform=ax.transAxes)
        
        ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/temporal_dominant_events.png', dpi=300)
    plt.close()


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
import seaborn as sns

def plot_power_spectrum(u_x, u_v_y, high_freq_coords, nx, ny, steps, output_dir, variable, perc, spectrum=True):
    # Grid parameters
    Nx, Ny, Nt = 288, 96, steps
    dx, dy, dt = nx[200] - nx[199], ny[91] - ny[90], 0.005  

    # Reshape velocity data (Nt, Ny, Nx) → (Nx, Ny, Nt)
    u_x = np.transpose(u_x[:, variable, :, :], (2, 1, 0))  
    u_v_y = np.transpose(u_v_y[:, variable, :, :], (2, 1, 0))  

    # Compute spatial FFT for velocity fields
    fft_u_x = fftshift(fft2(u_x.mean(axis=-1)))  
    fft_u_v = fftshift(fft2(u_v_y.mean(axis=-1)))  

    # Compute spatial FFT for high-frequency coordinates
    high_freq_x = np.transpose(high_freq_coords[:,variable,:,:, 0] ,(2,1,0)) # Extract x-coordinates
    high_freq_y =  np.transpose(high_freq_coords[:,variable,:,:, 1] ,(2,1,0))  # Extract y-coordinates
    fft_high_freq_x = fftshift(fft2(high_freq_x.mean(axis=-1)))  
    fft_high_freq_y = fftshift(fft2(high_freq_y.mean(axis=-1)))  

    # Compute power spectra
    power_u_x = np.abs(fft_u_x) ** 2 + 1e-10  
    power_u_v = np.abs(fft_u_v) ** 2 + 1e-10  
    power_high_freq = np.abs(fft_high_freq_x) ** 2 + np.abs(fft_high_freq_y) ** 2 + 1e-10  

    # Normalize
    power_u_x /= np.max(power_u_x)
    power_u_v /= np.max(power_u_v)
    power_high_freq /= np.max(power_high_freq)

    # Compute frequency axes
    kx = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
    ky = np.fft.fftshift(np.fft.fftfreq(Ny, d=dy))

    # Apply log transformation
    log_power_u_x = np.log10(power_u_x)
    log_power_u_v = np.log10(power_u_v)
    log_power_high_freq = np.log10(power_high_freq)

    # Adaptive thresholding: Set a percentile-based threshold to remove noise
    threshold_x = np.percentile(log_power_u_x, 5)
    threshold_v = np.percentile(log_power_u_v, 5)
    threshold_high_freq = np.percentile(log_power_high_freq, 5)

    log_power_u_x[log_power_u_x < threshold_x] = np.nan  
    log_power_u_v[log_power_u_v < threshold_v] = np.nan  
    log_power_high_freq[log_power_high_freq < threshold_high_freq] = np.nan  

    # Define contour levels dynamically
    levels_x = np.linspace(threshold_x, np.nanmax(log_power_u_x), 10)
    levels_v = np.linspace(threshold_v, np.nanmax(log_power_u_v), 10)
    levels_freq = np.linspace(threshold_high_freq, np.nanmax(log_power_high_freq), 12)

    # Create spatial power spectrum plots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # u_x spectrum
    im1 = axes[0].imshow(log_power_u_x.T, extent=[kx.min(), kx.max(), ky.min(), ky.max()],
                          origin='lower', aspect='auto', cmap='coolwarm', vmin=threshold_x)
    contour1 = axes[0].contour(kx, ky, log_power_u_x.T, levels=levels_x, colors='black', linewidths=0.5)
    fig.colorbar(im1, ax=axes[0], label='Log Power Spectrum')
    axes[0].set_title(f'Spatial Power Spectrum - $u_x$')
    axes[0].set_ylabel('$k_y$')

    # v_y spectrum
    im2 = axes[1].imshow(log_power_u_v.T, extent=[kx.min(), kx.max(), ky.min(), ky.max()],
                          origin='lower', aspect='auto', cmap='coolwarm', vmin=threshold_v)
    contour2 = axes[1].contour(kx, ky, log_power_u_v.T, levels=levels_v, colors='black', linewidths=0.5)
    fig.colorbar(im2, ax=axes[1], label='Log Power Spectrum')
    axes[1].set_title(f'Spatial Power Spectrum - $v_y$')
    axes[1].set_ylabel('$k_y$')

    # High-frequency coordinate spectrum
    im3 = axes[2].imshow(log_power_high_freq.T, extent=[kx.min(), kx.max(), ky.min(), ky.max()],
                          origin='lower', aspect='auto', cmap='magma', vmin=threshold_high_freq)
    contour3 = axes[2].contour(kx, ky, log_power_high_freq.T, levels=levels_freq, colors='black', linewidths=0.5)
    fig.colorbar(im3, ax=axes[2], label='Log Power Spectrum')
    axes[2].set_title(f'Spatial Power Spectrum - High-Frequency Structures')
    axes[2].set_xlabel('$k_x$')
    axes[2].set_ylabel('$k_y$')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/power_space_{variable}_{perc}.png', bbox_inches='tight')
    plt.close()



def plot_dominant_event_evolution(var, var_1, matrix_t_l, steps, x_vals, u_keras, u_v, save_path='testing/intersections/', sigma=1):
    """
    Plots the temporal evolution of the dominant event at each x-location for two variables.
    - X-axis: Spatial coordinate x
    - Y-axis: Time (ordered correctly from 0 to max steps)
    - Color: Dominant event of var (heatmap)
    - Contour lines (dashed): Dominant event of var_1, using same colors
    - Background: Pure white
    - sigma: Standard deviation for Gaussian smoothing
    """
    x_dim = matrix_t_l.shape[3]
    
    # Compute dominant events for var and var_1
    dominant_events_var = np.zeros((steps, x_dim), dtype=int)
    dominant_events_var1 = np.zeros((steps, x_dim), dtype=int)

    for t in range(steps):
        for x_index in range(x_dim):
            classified_var = classify_events(u_keras[t, var, :, x_index], u_v[t, var, :, x_index])
            classified_var1 = classify_events(u_keras[t, var_1, :, x_index], u_v[t, var_1, :, x_index])

            event_counts_var = np.array([np.sum(classified_var == event) for event in range(1, 5)])
            event_counts_var1 = np.array([np.sum(classified_var1 == event) for event in range(1, 5)])

            dominant_events_var[t, x_index] = np.argmax(event_counts_var) + 1  # Shift index to match event labels
            dominant_events_var1[t, x_index] = np.argmax(event_counts_var1) + 1  # Shift index to match event labels

    # Smooth both matrices
    dominant_events_var_smooth = gaussian_filter1d(dominant_events_var, sigma=sigma, axis=0)
    dominant_events_var1_smooth = gaussian_filter1d(dominant_events_var1, sigma=sigma, axis=0)

    # Define color map for both var and var_1 (same colors)
    color_map = {
        3: (0.6, 0.6, 1.0),  # Q3 - Soft Blue
        2: (0.6, 1.0, 0.6),  # Q2 - Soft Green
        1: (1.0, 0.6, 1.0),  # Q1 - Soft Purple
        4: (1.0, 1.0, 0.6),  # Q4 - Soft Yellow
    }
    cmap = plt.cm.colors.ListedColormap([color_map[i] for i in range(1, 5)])

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set background to white
    ax.set_facecolor('white')

    # Plot heatmap for var
    cax = ax.imshow(dominant_events_var_smooth.T, aspect='auto', cmap=cmap, interpolation='nearest', origin='lower')

    # Overlay dashed contour lines for var_1 using same colors
    for event in range(1, 5):
        ax.contour(
            dominant_events_var1_smooth.T, 
            levels=[event], 
            colors=[color_map[event]], 
            linestyles='dashdot', 
            linewidths=1.5
        )

    # Set labels and title
    ax.set_ylabel('$x$', fontsize=14)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylim([50,288])
    ax.set_title(f'Temporal Evolution of Dominant Events (var={var}, var_1={var_1})', fontsize=14)

    # Keep the original legend (no extra elements)
    legend_elements = [
        Line2D([0], [0], color=color_map[3], lw=6, label='Q1'),
        Line2D([0], [0], color=color_map[2], lw=6, label='Q2'),
        Line2D([0], [0], color=color_map[1], lw=6, label='Q3'),
        Line2D([0], [0], color=color_map[4], lw=6, label='Q4'),
    ]
    # ax.legend(handles=legend_elements, loc='lower right', fontsize=12, frameon=False)

    # Add colorbar for dominant events of var
    cbar = fig.colorbar(cax, ax=ax, ticks=[3, 2, 1, 4])
    cbar.set_label('Event Type (var)', fontsize=12)
    cbar.set_ticks([3, 2, 1, 4])

    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{save_path}/dominant_event_evolution_var_{var}_vs_{var_1}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def normalize_field(field):
    return (field - np.min(field)) / (np.max(field) - np.min(field))

def plot_unique(u_keras, u_v, var, steps, x_vals, y_vals, save_path='geometry'):
    """
    Plots the 2D spatial field with SHAP values for two latent vectors over time using Plasma colormap.
    """
    os.makedirs(save_path, exist_ok=True)  # Ensure save path exists

    for t in range(8):
        field_var = normalize_field(u_keras[t*steps,:,:])

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(field_var, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower', cmap='plasma')
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04, shrink=0.7)
        
        # Set background to white
        ax.set_facecolor('white')
        
        xb = np.array([-0.125, -0.125, 0.25, 0.25])
        yb = np.array([0.0, 1.0, 1.0, 0.0])
        ax.fill(xb, yb, 'white')

        ax.set_xlabel('$x$', fontsize=14)
        ax.set_ylabel('$y$', fontsize=14)
        ax.set_title(f'$u$ at t={t*steps}', fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{save_path}/2d_field_t{t}_{var}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Compute mean fields and normalize them
    field_u2 = normalize_field(np.mean(u_keras**2, axis=0))
    field_uv = normalize_field(np.mean(u_keras*u_v, axis=0))
    field_v2 = normalize_field(np.mean(u_v**2, axis=0))
    
    for field, title, filename in zip([field_u2, field_uv, field_v2],
                                       ['$\overline{u^2}$ Field', '$\overline{uv}$ Field', '$\overline{v^2}$ Field'],
                                       ['2d_field_u2', '2d_field_uv', '2d_field_v2']):
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(field, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower', cmap='plasma')
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04, shrink=0.7)
        
        # Set background to white
        ax.set_facecolor('white')
        
        xb = np.array([-0.125, -0.125, 0.25, 0.25])
        yb = np.array([0.0, 1.0, 1.0, 0.0])
        ax.fill(xb, yb, 'white')

        ax.set_xlabel('$x$', fontsize=14)
        ax.set_ylabel('$y$', fontsize=14)
        ax.set_title(title, fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{save_path}/{filename}_{var}.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_temporal_signal(Modes, x_idx, y_idx, variables, time_steps, save_path="temporal_signal.png"):
    """
    Plots the temporal evolution of a selected point (x_idx, y_idx) for each variable.
    
    Parameters:
    Modes : np.ndarray
        The dataset of modes with shape (variables, time, x, y).
    x_idx : int
        The x-coordinate index of the selected point.
    y_idx : int
        The y-coordinate index of the selected point.
    variables : int
        Number of variables (latent modes) to plot.
    time_steps : int
        Number of time steps.
    save_path : str, optional
        Path to save the figure (default is "temporal_signal.png").
    """
    temporal_evolution = np.zeros((variables, time_steps))
    
    # Extract temporal evolution for the selected point
    for t in range(time_steps):
        for v in range(variables):
            temporal_evolution[v, t] = Modes[v, t, x_idx, y_idx]
    
    # Plot the temporal evolution
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for v in range(variables):
        ax.plot(range(time_steps), temporal_evolution[v], label=f'Latent {v+1}')
    
    ax.set_xlabel("Time Step", fontsize=14)
    ax.set_ylabel("Mode Value", fontsize=14)
    ax.set_title("Temporal Evolution of Selected Point", fontsize=16)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"INFO: Temporal signal plot saved to {save_path}")


def plot_shap_video(var, var_1, matrix_t_l, steps, x_vals, y_vals, save_path="output_video.mp4"):
    """
    Generates a video of 2D spatial SHAP fields over time using Plasma colormap.

    Parameters:
    - var (int): First variable index.
    - var_1 (int): Second variable index.
    - matrix_t_l (numpy array): 4D array (vars, steps, y_dim, x_dim) containing SHAP data.
    - steps (int): Number of time steps.
    - x_vals (numpy array): x-coordinates for the spatial field.
    - y_vals (numpy array): y-coordinates for the spatial field.
    - save_path (str, optional): Path to save the generated video.
    """

    # Define Magma colormap
    magma_colors = cm.magma(np.linspace(0, 1, 9))
    color_var = magma_colors[5]  # Mode var (dark purple)
    color_var_1 = magma_colors[7]  # Mode var_1 (orange-yellow)
    color_intersection = magma_colors[2]  # Intersection (bright yellow)

    # Define the legend handles
    legend_elements = [
        Line2D([0], [0], color=color_var, lw=2, label=f'Mode {var}'),
        Line2D([0], [0], color=color_var_1, lw=2, label=f'Mode {var_1}'),
        Line2D([0], [0], color=color_intersection, lw=2, label=f'Mode {var} ∩ Mode {var_1}')
    ]

    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Initialize image
    img_display = ax.imshow(np.zeros_like(matrix_t_l[var, 0]), 
                            extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], 
                            origin='lower', alpha=0.7)

    # Add obstacle
    xb = np.array([-0.125, -0.125, 0.125, 0.125])  # Obstacle x-coordinates
    yb = np.array([0.0, 1.0, 1.0, 0.0])  # Obstacle y-coordinates
    ax.fill(xb, yb, 'white')  # No legend, just visual

    # Style adjustments
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$y$', fontsize=14)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks(np.linspace(y_vals.min(), y_vals.max(), 5))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Function to update each frame
    def update(t):
        field_var = matrix_t_l[var, t, :, :]
        field_var_1 = matrix_t_l[var_1, t, :, :]

        norm_var = field_var / np.max(field_var) if np.max(field_var) > 0 else field_var
        norm_var_1 = field_var_1 / np.max(field_var_1) if np.max(field_var_1) > 0 else field_var_1
        norm_intersection = np.minimum(field_var, field_var_1) / np.max(np.minimum(field_var, field_var_1)) if np.max(np.minimum(field_var, field_var_1)) > 0 else np.minimum(field_var, field_var_1)

        # Create color image
        img = np.zeros((*field_var.shape, 3))
        img[..., 0] = norm_var * color_var[0] + norm_var_1 * color_var_1[0] + norm_intersection * color_intersection[0]
        img[..., 1] = norm_var * color_var[1] + norm_var_1 * color_var_1[1] + norm_intersection * color_intersection[1]
        img[..., 2] = norm_var * color_var[2] + norm_var_1 * color_var_1[2] + norm_intersection * color_intersection[2]

        img_display.set_array(img)
        return img_display,

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=steps, blit=True)

    # Save as MP4
    writer = animation.FFMpegWriter(fps=10)
    ani.save(save_path, writer=writer, dpi=300)
    plt.close()

    print(f"Video saved at {save_path}")

def plot_shap_contour(shap_values, nx, steps, output_path):
    """
    Plot a contour map of SHAP evolution in space-time.

    Parameters:
        shap_values (np.array): Shape (Nt, Ny, Nx), SHAP values over time.
        nx (array): X-axis coordinates.
        steps (int): Total number of time steps.
        output_path (str): File path for saving the plot.
    """

    # Average SHAP values over Y
    shap_x_t = np.mean(shap_values, axis=1)

    plt.figure(figsize=(8, 6))
    T, X = np.meshgrid(np.arange(steps),nx)
    contour = plt.contourf(T, X, shap_x_t.T, levels=50, cmap='plasma')

    plt.colorbar(label="SHAP Value Intensity")
    plt.ylabel("X")
    plt.xlabel("Time Step")
    plt.title("SHAP Spatiotemporal Contour Map")
    
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_shap_contour_comparison(shap_values1, shap_values2, nx, steps, output_path):
    """
    Plot a contour map comparing the evolution of two SHAP fields in space-time.
    The first variable is plotted with the 'plasma' colormap, and the second with 'magma' at lower opacity.

    Parameters:
        shap_values1 (np.array): Shape (Nt, Ny, Nx), first SHAP field.
        shap_values2 (np.array): Shape (Nt, Ny, Nx), second SHAP field.
        nx (array): X-axis coordinates.
        steps (int): Total number of time steps.
        output_path (str): File path for saving the plot.
    """

    # Average SHAP values over Y
    shap_x_t1 = np.mean(shap_values1, axis=1)  # Shape: (Nt, Nx)
    shap_x_t2 = np.mean(shap_values2, axis=1)  # Shape: (Nt, Nx)

    plt.figure(figsize=(8, 6))
    T, X = np.meshgrid(np.arange(steps), nx)

    # Contour plot for the first variable (Plasma colormap)
    contour1 = plt.contourf(T, X, shap_x_t1.T, levels=15, cmap='plasma')#, alpha=1)

    # Contour plot for the second variable (Magma colormap, more transparent)
    contour2 = plt.contourf(T, X, shap_x_t2.T, levels=15, cmap='viridis', alpha=0.5)

    # Colorbar for first variable
    cbar = plt.colorbar(contour1)
    cbar.set_label("SHAP Value Intensity (Variable 1)")

    plt.ylabel("X")
    plt.xlabel("Time Step")
    plt.title("SHAP Spatiotemporal Contour Map Comparison")

    plt.savefig(output_path, dpi=200)
    plt.close()
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import numpy as np

def plot_appear_matrix_1(appear, path, tol=0.01):
    """
    Plots the number of unique indices (appear[t, variable]) over time and analyzes cyclic behavior
    by extracting trajectories between consecutive global maxima. For each variable:
    
      - Identifies global maximum points (using a tolerance relative to the maximum value).
      - Uses these points as cycle boundaries (global maxima repeated due to periodicity).
      - Extracts the trajectory from one global maximum to the next and computes the period
        (i.e., the time difference between them).
      - Creates a main plot showing the overall time series with global maxima annotated.
      - Creates an external plot that overlays the normalized cycle trajectories for comparison.
    
    Parameters:
      - appear: 2D numpy array of shape (time_steps, variables).
      - path: String used for saving the resulting plot images.
      - tol: Tolerance fraction relative to the maximum value to decide if a point qualifies as a global maximum.
             Default is 0.01 (i.e. within 1% of the maximum).
    """
    time_steps = appear.shape[0]   # Total number of time steps
    variables = appear.shape[1]    # Total number of variables

    # ---- Main Plot: Overall Time Series with Global Maximum Markers ----
    plt.figure(figsize=(12, 8))
    for var in range(variables):
        series = appear[:, var]
        time = np.arange(time_steps)
        plt.plot(time, series, label=f'Variable {var+1}')
        
        # Identify the maximum value and then the indices that are within tolerance of that max
        max_val = series.max()
        tolerance = tol * max_val
        candidate_indices = np.where(np.abs(series - max_val) < tolerance)[0]
        
        # Filter out consecutive indices; we only want one representative per cycle boundary.
        global_maxima = []
        if candidate_indices.size > 0:
            global_maxima.append(candidate_indices[0])
            for idx in candidate_indices[1:]:
                if idx != global_maxima[-1] + 1:
                    global_maxima.append(idx)
        global_maxima = np.array(global_maxima)
        
        # Plot the global maximum points on the main time series.
        plt.plot(time[global_maxima], series[global_maxima], "o", color='black')
        
        # If we have at least two global maxima, compute and annotate the period.
        if len(global_maxima) >= 2:
            periods = np.diff(global_maxima)
            avg_period = np.mean(periods)
            plt.annotate(f'Avg period: {avg_period:.2f}', 
                         xy=(time[global_maxima[0]], series[global_maxima[0]]),
                         xytext=(time[global_maxima[0]] + 5, series[global_maxima[0]] + 0.1 * max_val),
                         arrowprops=dict(arrowstyle='->', color='black'),
                         fontsize=9)
            print(f"Variable {var+1}: Global maxima at time steps: {global_maxima}")
            print(f"Periods between global maxima: {periods}")
            print(f"Average period: {avg_period:.2f}\n")
        else:
            print(f"Variable {var+1}: Not enough global maxima detected for cycle analysis.\n")
    
    plt.xlabel('Time step')
    plt.ylabel('Number of unique indices')
    plt.title('Unique Indices Over Time with Global Maximum Analysis')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"testing/intersections/{path}_appear.png", dpi=200)
    plt.close()

    # ---- External Plot: Cycle Trajectories Comparison ----
    for var in range(variables):
        series = appear[:, var]
        time = np.arange(time_steps)
        max_val = series.max()
        tolerance = tol * max_val
        candidate_indices = np.where(np.abs(series - max_val) < tolerance)[0]
        global_maxima = []
        if candidate_indices.size > 0:
            global_maxima.append(candidate_indices[0])
            for idx in candidate_indices[1:]:
                if idx != global_maxima[-1] + 1:
                    global_maxima.append(idx)
        global_maxima = np.array(global_maxima)
        
        if len(global_maxima) < 2:
            print(f"Variable {var+1}: Not enough global maxima for trajectory comparison.")
            continue
        
        plt.figure(figsize=(10, 6))
        # For each cycle (trajectory from one global maximum to the next)
        for i in range(len(global_maxima) - 1):
            start = global_maxima[i]
            end = global_maxima[i+1]
            traj = series[start:end+1]
            period = end - start
            # Normalize the time axis to compare cycles on a common [0, 1] scale.
            norm_time = np.linspace(0, 1, len(traj))
            plt.plot(norm_time, traj, label=f'Cycle {i+1} (Period: {period})')
        
        plt.xlabel('Normalized Time')
        plt.ylabel('Unique Indices')
        plt.title(f'Cycle Trajectories for Variable {var+1}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"testing/intersections/{path}_trajectories_var{var+1}.png", dpi=200)
        plt.close()

import matplotlib.pyplot as plt
import numpy as np

def plot_appear_matrix(appear, path, tol=0.01):
    """
    Plots the number of unique indices (appear[t, variable]) over time for each variable,
    annotating each absolute (global) maximum with the time difference from the previous one.
    
    For each variable, the function:
      - Determines the absolute global maximum value.
      - Finds all time steps where the value is within a specified tolerance of that maximum.
      - Filters out consecutive points so that only one representative is taken per cycle.
      - Annotates each global maximum (except the first) with the time difference from its previous global maximum.
      - Plots a distinct point at every time step that is a multiple of 485, as this lag seems to be the most causal.
      
    It also extracts the cycle trajectories (segments between consecutive global maxima)
    and creates an external plot comparing the normalized cycles.
    
    Parameters:
      - appear: 2D numpy array of shape (time_steps, variables) with the measured indices.
      - path: String used for saving the resulting plot images.
      - tol: Tolerance fraction relative to the global maximum to determine if a value qualifies as a global maximum.
             (Default: 0.01, i.e. within 1% of the maximum.)
    """
    time_steps = appear.shape[0]   # Total number of time steps
    variables = appear.shape[1]    # Total number of variables

    # ---- Main Plot: Overall Time Series with Global Maximum Annotations and Multiples of 485 ----
    plt.figure(figsize=(12, 8))
    for var in range(variables):
        series = appear[:, var]
        time = np.arange(time_steps)
        plt.plot(time, series, label=f'Variable {var+1}')
        
        # Plot points for multiples of 485.
        # These indices are considered important as they represent the most causal lag.
        multiples = np.arange(0, time_steps, 485)
        # Only add the label for the first variable to avoid duplicate legend entries.
        plt.scatter(time[multiples], series[multiples], marker='D', color='magenta', zorder=10,
                    label='Multiples of 485' if var == 0 else "")
        
        # Determine the absolute global maximum for this variable.
        max_val = series.max()
        tolerance = tol * max_val
        candidate_indices = np.where(np.abs(series - max_val) < tolerance)[0]
        
        # Filter out consecutive indices so that only one representative is taken per cycle boundary.
        global_maxima = []
        if candidate_indices.size > 0:
            global_maxima.append(candidate_indices[0])
            for idx in candidate_indices[1:]:
                if idx != global_maxima[-1] + 1:
                    global_maxima.append(idx)
        global_maxima = np.array(global_maxima)
        
        # Plot the global maximum points on the main time series.
        plt.plot(time[global_maxima], series[global_maxima], "o", color='black')
        
        # Annotate each global maximum (except the first) with the time difference from the previous one.
        for i in range(1, len(global_maxima)):
            dt = global_maxima[i] - global_maxima[i-1]
            plt.text(time[global_maxima[i]], series[global_maxima[i]], f"{dt}", 
                     fontsize=9, color='red', ha='center', va='bottom')
        
        print(f"Variable {var+1}: Global maxima at time steps: {global_maxima}")
        if len(global_maxima) >= 2:
            print(f"Time differences between consecutive global maxima: {np.diff(global_maxima)}\n")
        else:
            print(f"Variable {var+1}: Not enough global maxima for cycle analysis.\n")
    plt.plot(time, np.sum(appear,axis=1), label=f'Total structures')
    plt.xlabel('Time step')
    plt.ylabel('Number of unique indices')
    plt.title('Unique Indices Over Time with Global Maximum Annotations')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"testing/intersections/{path}_appear.png", dpi=200)
    plt.close()

    # ---- External Plot: Cycle Trajectories Comparison for Each Variable ----
    for var in range(variables):
        series = appear[:, var]
        time = np.arange(time_steps)
        max_val = series.max()
        tolerance = tol * max_val
        candidate_indices = np.where(np.abs(series - max_val) < tolerance)[0]
        global_maxima = []
        if candidate_indices.size > 0:
            global_maxima.append(candidate_indices[0])
            for idx in candidate_indices[1:]:
                if idx != global_maxima[-1] + 1:
                    global_maxima.append(idx)
        global_maxima = np.array(global_maxima)
        
        if len(global_maxima) < 2:
            print(f"Variable {var+1}: Not enough global maxima for trajectory comparison.")
            continue
        
        plt.figure(figsize=(10, 6))
        # For each cycle (trajectory from one global maximum to the next)
        for i in range(len(global_maxima) - 1):
            start = global_maxima[i]
            end = global_maxima[i+1]
            traj = series[start:end+1]
            period = end - start
            # Normalize the time axis to [0, 1] for direct cycle comparison.
            norm_time = np.linspace(0, 1, len(traj))
            plt.plot(norm_time, traj, label=f'Cycle {i+1} (Δt={period})')
        
        plt.xlabel('Normalized Time')
        plt.ylabel('Unique Indices')
        plt.title(f'Cycle Trajectories for Variable {var+1}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"testing/intersections/{path}_trajectories_var{var+1}.png", dpi=200)
        plt.close()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
def plot_causal_snapshots_improved(matrix, nx, ny, time_indices, output_path_prefix, variable):
    """
    Plot and save refined 2D spatial (X-Y) maps of the causal field for a given variable at specified time steps.
    The function expects that the input 'matrix' is a 4D array with shape (n_variables, Nt, Ny, Nx) where each pixel
    has been recoded into a small number of categories:
        0   : Background (no causal event)
        11  : Uniqueness of latent 1
        12  : Uniqueness of latent 2
        13  : Uniqueness of latent 3
        21  : Redundancy S12
        22  : Redundancy S13
        23  : Redundancy S23
        31  : Synergy led by latent 1 (S1)
        32  : Synergy led by latent 2 (S2)
        33  : Synergy S12 (combination of latent 1 and 2)
        40  : Synergy of all three (S123)
    
    Additionally, an obstacle is drawn on each plot using predetermined coordinates.
    
    Parameters:
      matrix           : 4D numpy array (n_variables, Nt, Ny, Nx) of recoded causal codes.
      nx (array)       : 1D array of X-axis coordinates (length = Nx).
      ny (array)       : 1D array of Y-axis coordinates (length = Ny).
      time_indices     : List or array of time indices (0 ≤ t < Nt) to plot.
      output_path_prefix: String prefix for the saved image files.
      variable         : Index of the variable (first dimension of matrix) to be plotted.
    """
    # Define refined mapping from recoded causal codes to labels and colors.
    recoded_matrix = recode_causal_matrix_l(matrix)
    code_to_label = {
         0: "Background",
        11: "U1",
        12: "U2",
        13: "U3",
        21: "R12",
        22: "R13",
        23: "R23",
        31: "R123",
        32: "S13",
        33: "S12",
        40: "S123"
    }
    code_to_color = {
         0: "white",
        11: "#ADD8E6",   # light blue
        12: "#90EE90",   # light green
        13: "#FFFFE0",   # light yellow
        21: "#FFB6C1",   # light pink
        22: "#FFA07A",   # light salmon
        23: "#FF69B4",   # hot pink
        31: "purple",
        32: "brown",
        33: "magenta",
        40: "gold"
    }
    
    # Use only the codes that actually appear (or force the full set).
    codes = sorted(code_to_label.keys())
    # Define boundaries halfway between consecutive codes.
    boundaries = [codes[0] - 5] + [(codes[i] + codes[i+1]) / 2 for i in range(len(codes)-1)] + [codes[-1] + 5]
    
    # Create a discrete colormap.
    cmap_colors = [code_to_color[code] for code in codes]
    cmap = mcolors.ListedColormap(cmap_colors)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    
    # Define obstacle coordinates (assumed to be in the same coordinate system as nx, ny).
    xb = np.array([-0.125, -0.125, 0.125, 0.125])  # Obstacle x-coordinates
    yb = np.array([0.0, 1.0, 1.0, 0.0])             # Obstacle y-coordinates

    # Create a meshgrid for spatial coordinates.
    X, Y = np.meshgrid(nx, ny)
    
    for t in time_indices:
        snapshot = recoded_matrix[variable, t, :, :]  # shape: (Ny, Nx)
        plt.figure(figsize=(8, 6))
        contour = plt.contourf(X, Y, snapshot, levels=boundaries, cmap=cmap, norm=norm)
        cbar = plt.colorbar(contour, ticks=codes)
        cbar.ax.set_yticklabels([code_to_label[code] for code in codes], fontsize=12)
        cbar.set_label("Causal Category", fontsize=12)
        
        plt.xlabel("X", fontsize=14)
        plt.ylabel("Y", fontsize=14)
        plt.title(f"Causal Field (Variable {variable}) at Time Step {t}", fontsize=16)
        
        # Add the obstacle to the plot.
        ax = plt.gca()
        ax.fill(xb, yb, color="grey", alpha=0.8)  # Draw the obstacle in grey.
        
        plt.savefig(f"{output_path_prefix}_t{t}_impr.png", dpi=200, bbox_inches='tight')
        plt.close()
def recode_causal_matrix_l(matrix):
    """
    Recode the detailed causal codes into four broad categories:
      0  -> Background
     10  -> Redundancy (if original code in [-1, -2, -4])
     20  -> Uniqueness (if original code in [-3, -5, -6])
     30  -> Synergy (if original code in [1, 2, -7])
     
    Parameters:
        matrix (np.array): Input matrix with detailed causal codes.
        
    Returns:
        recoded (np.array): Matrix recoded into {0,10,20,30}.
    """
    recoded = np.copy(matrix)
    # Set background to 0 (we assume background is coded as 0 in the original matrix)
    recoded[matrix == 0] = 0
    
    # Redundancy: original codes -1, -2, -4.
    recoded[np.isin(matrix, [-7])] = 11
    # Uniqueness: original codes -3, -5, -6.
    recoded[np.isin(matrix, [-6])] = 12
    # Synergy: original codes 1, 2, -7.
    recoded[np.isin(matrix, [-5])] = 21
    recoded[np.isin(matrix, [-4])] = 13
    recoded[np.isin(matrix, [-2])] = 23
    recoded[np.isin(matrix, [-1])] = 31
    recoded[np.isin(matrix, [-3])] = 22
    recoded[np.isin(matrix, [1])] = 31
    recoded[np.isin(matrix, [2])] = 33
    recoded[np.isin(matrix, [-8])] = 40
    return recoded
def recode_causal_matrix(matrix):
    """
    Recode the detailed causal codes into four broad categories:
      0  -> Background
     10  -> Redundancy (if original code in [-1, -2, -4])
     20  -> Uniqueness (if original code in [-3, -5, -6])
     30  -> Synergy (if original code in [1, 2, -7])
     
    Parameters:
        matrix (np.array): Input matrix with detailed causal codes.
        
    Returns:
        recoded (np.array): Matrix recoded into {0,10,20,30}.
    """
    recoded = np.copy(matrix)
    # Set background to 0 (we assume background is coded as 0 in the original matrix)
    recoded[matrix == 0] = 0
    
    # Redundancy: original codes -1, -2, -4.
    recoded[np.isin(matrix, [-1, -2, -4])] = 10
    # Uniqueness: original codes -3, -5, -6.
    recoded[np.isin(matrix, [-3, -5, -6])] = 20
    # Synergy: original codes 1, 2, -7.
    recoded[np.isin(matrix, [1, 2, -7])] = 30
    
    return recoded

def plot_causal_snapshots_l(matrix, nx, ny, time_indices, output_path_prefix, variable):
    """
    Plot and save 2D spatial (X-Y) maps of the causal matrix for a given variable at specified time steps.
    
    Parameters:
        matrix (np.array): 4D array with shape (n_variables, Nt, Ny, Nx) containing integer causal codes.
        nx (array): 1D array of X-axis coordinates (length = Nx).
        ny (array): 1D array of Y-axis coordinates (length = Ny).
        time_indices (list or array): List of time indices (within 0...Nt-1) to plot.
        output_path_prefix (str): Prefix for the saved output image files.
        variable (int): Index of the variable (first dimension in matrix) to be plotted.
    """
    # First recode the detailed matrix into 4 broad categories.
    recoded_matrix = recode_causal_matrix(matrix)
    
    # Define a simple mapping for the recoded values.
    code_to_label = {
         0: "Background",
        10: "Redundancy",
        20: "Uniqueness",
        30: "Synergy"
    }
    code_to_color = {
         0: "white",
        10: "#FF9999",   # light red for redundancy
        20: "#99CCFF",   # light blue for uniqueness
        30: "#99FF99"    # light green for synergy
    }
    
    # The recoded matrix now only takes on values {0, 10, 20, 30}.
    codes = sorted(code_to_label.keys())  # [0, 10, 20, 30]
    # Define boundaries halfway between codes.
    boundaries = [codes[0] - 5] + [(codes[i] + codes[i+1]) / 2 for i in range(len(codes)-1)] + [codes[-1] + 5]
    
    cmap_colors = [code_to_color[code] for code in codes]
    cmap = mcolors.ListedColormap(cmap_colors)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    
    # Create a meshgrid for the spatial coordinates.
    X, Y = np.meshgrid(nx, ny)
    
    for t in time_indices:
        # Extract the 2D field (Y, X) for the given variable at time t.
        snapshot = recoded_matrix[variable, t, :, :]
        
        plt.figure(figsize=(8, 6))
        contour = plt.contourf(X, Y, snapshot, levels=boundaries, cmap=cmap, norm=norm)
        cbar = plt.colorbar(contour, ticks=codes)
        cbar.ax.set_yticklabels([code_to_label[code] for code in codes], fontsize=12)
        cbar.set_label("Causal Category", fontsize=12)
        
        plt.xlabel("X", fontsize=14)
        plt.ylabel("Y", fontsize=14)
        plt.title(f"Causal Field (Variable {variable}) at Time Step {t}", fontsize=16)
        plt.savefig(f"{output_path_prefix}_t{t}.png", dpi=200, bbox_inches='tight')
        plt.close()

def plot_appear_ener(signal, filename=None):
    """
    Plot the time evolution (appearance) of energy for one variable and save the figure if a filename is provided.
    
    Parameters:
        signal (np.array): A 1D array representing the energy time series.
        filename (str, optional): If provided, the plot will be saved to this file.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(signal, 'b-', linewidth=2)
    plt.xlabel('Time step', fontsize=14)
    plt.ylabel('Energy', fontsize=14)
    plt.title('Energy Time Signal', fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
print('NX', nx[:].shape,matrix_t_l.shape)

# plot_smooth_intersection(0, 1, 2, matrix_t_l, matrix_t_l.shape[1], nx[:288], save_path='testing/intersections/')
# plot_temporal_signal(matrix_t_l/2,48,144,3,1000,save_path="testing/intersections/temporal_signal.png")
# # plot_smooth_intersection_4(1, 2,4,7 , 15,matrix_t_l, matrix_t_l.shape[1], nx[:288], save_path='testing/intersections/')
# # plot_smooth_intersection_3(0, 3, 6, 15,matrix_t_l, matrix_t_l.shape[1], nx[:288], save_path='testing/intersections/')
# # plot_smooth_intersection_3(0, 2, 7,15, matrix_t_l, matrix_t_l.shape[1], nx[:288], save_path='testing/intersections/')
# # plot_smooth_intersection_4(2, 4, 5,7, 15,matrix_t_l, matrix_t_l.shape[1], nx[:288], save_path='testing/intersections/')
# plot_2d_shap_fields(0, 1, 2, matrix_t_l,10, nx[:288], ny[:96])
# plot_2d_fields(u_keras,u_v,100, nx[:288], ny[:100])
# plot_super(0, 1, 15, matrix_t_l,u_keras, 100, nx[:288], ny[:96])
# plot_super_3(0, 1,2, 15, matrix_t_l,u_keras, 100, nx[:288], ny[:96])# plot_shap_video(0,2, matrix_t_l, steps,nx[:288], ny[:96])
# # plot_super_4(1, 2,4 ,15, matrix_t_l,u_keras, 10, nx[:288], ny[:96])
# print('START')
# plot_event_overlay(0,2,u_keras, u_x, u_v_y,100, nx[:288], ny[:96])
# plot_event_contributions_on_intersection(0, 2, matrix_t_l,steps,nx[:288], ny[:96],u_x,u_v_y)
# # plot_event_contributions_on_intersection_4( 1, 2,4,7, matrix_t_l,steps,nx[:288], ny[:96],u_x,u_v_y)
# # print('U',u_x.shape)
# # # plot_temporal_event_contributions(3, 8, matrix_t_l,steps,nx[:288], ny[:96],u_x,u_v_y)
# # # plot_temporal_dominant_events(3, 8, matrix_t_l,steps,nx[:288],u_x,u_v_y)
# print('OCUR')
# # plot_event_occurrences(0, 2, matrix_t_l,steps,nx[:288],u_x,u_v_y)
# # plot_dominant_event_evolution(0, 1, matrix_t_l,steps,nx[:288],u_x,u_v_y)
# plot_power_spectrum(u_x, u_v_y, space,nx[:288], ny[:96], steps, 'testing/intersections/', 0, perc)
# plot_power_spectrum(u_x, u_v_y, space, nx[:288], ny[:96], steps, 'testing/intersections/',1, perc)
# plot_power_spectrum(u_x, u_v_y, space, nx[:288], ny[:96], steps, 'testing/intersections/',2, perc)
plot_shap_contour(matrix_t_l[0,:,:,:],nx[:288],steps,'spatio_0.png')
plot_shap_contour(matrix_t_l[2,:,:,:],nx[:288],steps,'spatio_2.png')
plot_shap_contour(matrix_t_l[1,:,:,:],nx[:288],steps,'spatio_1.png')

# print('COMPARISON')
# plot_shap_contour_comparison(matrix_t_l[0,:,:,:],matrix_t_l[2,:,:,:],nx[:288],steps,'spatio_mix.png')

# # plot_dominant_event_evolution(8, matrix_t_l,steps,nx[:288],u_x,u_v_y)
# # plot_dominant_event_evolution(5, matrix_t_l,steps,nx[:288],u_x,u_v_y)
# # plot_dominant_event_evolution(2, matrix_t_l,steps,nx[:288],u_x,u_v_y)
plot_unique(u_x_u[:,0,:,:],u_v_u[:,0,:,:],0,100, nx[:288], ny[:100],save_path="testing/intersections/")
plot_unique(u_x_u[:,1,:,:],u_v_u[:,1,:,:],1,100, nx[:288], ny[:100],save_path="testing/intersections/")
plot_unique(u_x_u[:,2,:,:],u_v_u[:,2,:,:],2,100, nx[:288], ny[:100],save_path="testing/intersections/")
plot_appear_matrix(appear,1)
plot_appear_matrix(appear_1,2)
plot_causal_snapshots_improved(matrix_causal, nx[:288], ny[:96], time_indices=[0, 485, 1000],output_path_prefix="causal_snapshot_variable0", variable=0)
plot_causal_snapshots_l(matrix_causal, nx[:288], ny[:96], time_indices=[0, 50, 100],output_path_prefix="causal_snapshot_variable0", variable=2)
plot_causal_snapshots_improved(matrix_causal, nx[:288], ny[:96], time_indices=[0, 485, 1000],output_path_prefix="causal_snapshot_variable1", variable=1)
plot_causal_snapshots_improved(matrix_causal, nx[:288], ny[:96], time_indices=[0, 485, 1000],output_path_prefix="causal_snapshot_variable2", variable=2)

#                       output_path_prefix="causal_snapshot_variable0", variable=0)
# plot_unique(u_x_u[:,4,:,:],u_v_u[:,4,:,:],4,100, nx[:288], ny[:100])
print('----ENERGY ------')
Er = (u_keras**2 + u_v**2)/2
plot_appear_ener(Er[:5000,48,144],filename='ener')
plot_appear_ener(u_keras[:5000,48,144]**2,filename='u_f')
plot_appear_ener(u_v[:5000,48,144]**2,filename='v_f')

domain_energy = np.mean(Er, axis=(1, 2))
domain_u_f    = np.mean(u_keras**2, axis=(1, 2))
domain_v_f    = np.mean(u_v**2, axis=(1, 2))

# Plot and save the time series for the entire domain.
plot_appear_ener(domain_energy[:], filename='domain_energy.png')
plot_appear_ener(domain_u_f[:5000], filename='domain_u_f.png')
plot_appear_ener(domain_v_f[:5000], filename='domain_v_f.png')



# plot_event_overlay_4(1, 2,4,7,u_keru_vas, u_x, u_v_y, nx[:288], ny[:96])
print('END')
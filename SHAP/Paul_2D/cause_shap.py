import numpy as np
from scipy.ndimage import gaussian_filter
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import gaussian_kde

def separate_structures(nx, ny, structure_matrix):
    """
    Function to separate the different coherent structures in a 2D matrix.

    Returns
    -------
    nodes : list of structures (list of arrays)
    """
    mat_struc_copy = structure_matrix.copy()
    dirs = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])  # Up, Left, Down, Right

    list_waiting = []
    nodes = []

    for ind_y in range(ny):
        for ind_x in range(nx):
            if mat_struc_copy[ind_y, ind_x] == 0:
                continue
            else:
                ind_yx = [ind_y, ind_x]
                list_struc = [np.array(ind_yx, dtype='int')]
                mat_struc_copy[ind_y, ind_x] = 0
                list_waiting.append(ind_yx)  # Initialize the waiting list

                while list_waiting:  # Process until the waiting list is empty
                    ind_yx = list_waiting.pop(0)  # Take the first node from the waiting list
                    dir_ind = ind_yx + dirs

                    for dir_ii in dir_ind:
                        if 0 <= dir_ii[0] < ny and 0 <= dir_ii[1] < nx:
                            if mat_struc_copy[dir_ii[0], dir_ii[1]] == 1:
                                list_struc.append(dir_ii)
                                mat_struc_copy[dir_ii[0], dir_ii[1]] = 0
                                list_waiting.append(dir_ii)
                points = np.array(list_struc)                 
                # if points.shape[0] > 8:  # Filter small structures
                #     nodes.append(np.array(list_struc).T)
                nodes.append(np.array(list_struc).T)

    return nodes  # Return the list of structures

# Function to create a structure mask based on the identified structures
def create_structure_mask(structures, mask_shape):
    """
    Create a mask from the identified structures.

    Parameters
    ----------
    structures : list of arrays
        Each array contains the points of a structure.
    mask_shape : tuple
        Shape of the mask to be created.

    Returns
    -------
    mask : np.ndarray
        Array of zeros with the shape specified, with 1s where structures are located.
    """
    # Initialize the mask with zeros
    mask = np.zeros(mask_shape, dtype=np.float32)

    # Iterate through each structure
    for structure in structures:
        for point in structure.T:  # Transpose to iterate through points
            # Check if the point is within bounds
            if 0 <= point[0] < mask_shape[0] and 0 <= point[1] < mask_shape[1]:
                mask[point[0], point[1]] = 1  # Mark points in the mask
            else:
                print(f"Point {point} is out of bounds for the mask of shape {mask_shape}")

    return mask

def count_structures(H, shap_u, shap_v, mse, nx, ny, max_structures, nota ,sigma=1):
    """
    Function to count the number of structures for a given H.
    Ensures that SHAP values are smoothed and that structure extraction respects the domain.
    """
    # Apply Gaussian smoothing to SHAP values
    # shap_u_smooth = gaussian_filter(shap_u, sigma=sigma)
    # shap_v_smooth = gaussian_filter(shap_v, sigma=sigma)

    # Create binary masks based on smoothed SHAP values exceeding the MSE
    binary_mask_u = np.where(shap_u > H * mse, 1, 0)
    binary_mask_v = np.where(shap_v > H * mse, 1, 0)

    # Convert binary masks to float32 for structure matrix 
    structure_matrix_u = binary_mask_u.astype(np.float32)
    structure_matrix_v = binary_mask_v.astype(np.float32)
    # print('STRUCTURE:',structure_matrix_u.shape,H)

    total_structures = 0

    # Loop through each time step and count the structures
    for t in range(20):  # Assuming you want to process two time steps (adjust if necessary)
        # print('-------- nOTa:', nota)
        structures_u = separate_structures(len(nx[:288]), len(ny[:96]), structure_matrix_u[t, :, :])
        structures_v = separate_structures(len(nx[:288]), len(ny[:96]), structure_matrix_v[t, :, :])
        
        # Sum the number of structures detected in both u and v components
        total_structures += len(structures_u)
        total_structures += len(structures_v)
    for t in range(20): # Assuming you want to process two time steps)   
        if total_structures > max_structures and nota == 0:
            all_structures_u[j].append(structures_u)
            all_structures_v[j].append(structures_v)
            # print(t)
        elif total_structures > max_structures and nota == 1:
            # If we've already appended once, substitute the last structures
            t_l = t - 20
            print('local', t_l, H)
            all_structures_u[j][t_l] = structures_u
            all_structures_v[j][t_l] = structures_v
    nota = 1



    return total_structures, structures_u, structures_v, nota

def load_all_shap_values(start_index, end_index):
    shap_u = []
    shap_v = []

    for i in range(start_index, end_index + 1):
        file_name = f"SHAPS/3/shap_values_{i}.h5"
        
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

# Load shap values
# with h5py.File(shap_file, 'r') as hf:
#     shap_u = np.array(hf['u'])
#     shap_v = np.array(hf['v'])
shap_u, shap_v = load_all_shap_values(0, 1)
    # structures_v = np.array(hf['structures_v'])
    # structures_u = np.array(hf['structures_u'])

# Load data
with h5py.File(datafile, 'r') as f:
    u_keras = np.array(f['u_fluc'][:], dtype=np.float32)
    nt, nx, ny = f['t'][()], f['x'][()], f['y'][()]
    u_mean = f['means'][:]
    u_std = np.array(f['u_fluc'][:], dtype=np.float32)
    u_v = np.array(f['v_fluc'][:], dtype=np.float32)

# print('---------- LOAD DATA -------------',structures_u.shape,structures_v.shape)
sigma = 1
# Check the shape of nx and ny
print(f"Shape of nx: {nx.shape}, Shape of ny: {ny.shape}, Study case {shap_name}")

# ... your other imports and definitions ...
local = 0
steps = 1000
steps_1 = 1000
size = steps_1
vars = 3
values = 15
H_values = [1.14,1.36,1.14]#,1,1.06,1,1.06,1.11,1.22,1.11]

# for s_i in range(int(size/steps)):
#     print('Pre-smoothing SHAP fields for iteration:', s_i,shap_u.shape)
#     # Smooth the SHAP fields for each channel
#     shap_u_smooth = gaussian_filter(shap_u, sigma=sigma)
#     shap_v_smooth = gaussian_filter(shap_v, sigma=sigma)

#     print('Computing MSE for thresholding')
#     # Compute the mean squared error (MSE) for each latent dimension (across space and time)
#     mse_u = np.mean(shap_u_smooth**2, axis=(0,1,2))  # shape: (vars,)
#     mse_v = np.mean(shap_v_smooth**2, axis=(0,1,2))  # shape: (vars,)
#     mse = np.sqrt(mse_u + mse_v)                     # shape: (vars,)

#     # Create a binary mask array for positive contributions only.
#     # We'll create one mask per latent variable of shape (size, 96, 288)
#     binary_total_pos = np.zeros((size, 96, 288, vars), dtype=np.int32)
    
#     for i in range(vars):
#         print('Processing latent dimension', i, 'with H value:', H_values[i])
#         # Compute the net contribution for the i-th latent variable.
#         # Here, net_shap is simply the sum of the contributions from the two channels.
#         net_shap = shap_u_smooth[:,:,:,i] + shap_v_smooth[:,:,:,i]
#         # Compute the magnitude for thresholding purposes.
#         shap_mag = np.sqrt(shap_u_smooth[:,:,:,i]**2 + shap_v_smooth[:,:,:,i]**2)
#         # Define the threshold based on H_values and the mse for this latent variable.
#         threshold = H_values[i] * mse[i]
#         # Create a binary mask: only keep pixels with net positive contribution that exceed the threshold.
#         mask = (net_shap > 0) & (shap_mag > threshold)
#         binary_total_pos[:,:,:,i] = mask.astype(np.int32)
    
#     # ---------------------------------------------------------------
#     # Now, compute pairwise synergy, redundancy, and uniqueness maps
#     # using the positive binary masks.
#     # ---------------------------------------------------------------
    
#     # Initialize arrays to hold pairwise maps.
#     # synergy_total_pos and redundancy_total_pos will have shape: (size, 96, 288, vars, vars)
#     synergy_total_pos    = np.zeros((size, 96, 288, vars, vars), dtype=np.int32)
#     redundancy_total_pos = np.zeros((size, 96, 288, vars, vars), dtype=np.int32)
#     # uniqueness_total_pos will hold two channels: [unique for latent i, unique for latent j]
#     uniqueness_total_pos = np.zeros((size, 96, 288, vars, vars, 2), dtype=np.int32)
    
#     # Loop over all unique pairs of latent variables.
#     for i in range(vars):
#         for j in range(i+1, vars):
#             mask_i = binary_total_pos[:,:,:,i]
#             mask_j = binary_total_pos[:,:,:,j]
            
#             # Redundancy: intersection of the two binary masks.
#             intersection_pos = mask_i & mask_j
#             # Synergy: union of the two binary masks.
#             union_pos = mask_i | mask_j
#             # Uniqueness for i: pixels in i that are not in the intersection.
#             unique_i = mask_i - intersection_pos
#             # Uniqueness for j: pixels in j that are not in the intersection.
#             unique_j = mask_j - intersection_pos
            
#             redundancy_total_pos[:,:,:,i,j] = intersection_pos
#             synergy_total_pos[:,:,:,i,j]    = union_pos
#             uniqueness_total_pos[:,:,:,i,j,0] = unique_i
#             uniqueness_total_pos[:,:,:,i,j,1] = unique_j

#     # --- Visualization and saving figures ---
#     # For demonstration, save a figure from the positive synergy map for a chosen latent pair at a specific time step.
#     time_idx = 0  # Modify as needed for your visualization
#     latent_i = 0
#     latent_j = 2
#     nxx, nyy = 288, 96  # Dimensions of the grid
#     x = np.linspace(nx.min(), nx.max(), nxx)
#     y = np.linspace(ny.min(), ny.max(), nyy)
#     X, Y = np.meshgrid(x, y)
#     # Plot the synergy map (union) for positive contributions
#     plt.figure(figsize=(6,4))
#     plt.contour(X,Y,synergy_total_pos[time_idx, :, :, latent_i, latent_j].T, cmap='plasma')
#     plt.title("Synergy (Positive) for latent pair ({} & {})".format(latent_i, latent_j))
#     plt.colorbar()
#     plt.savefig("synergy_pos_latent{}_{}.png".format(latent_i, latent_j))
#     plt.close()

#     # Plot the redundancy map (intersection) for positive contributions
#     plt.figure(figsize=(6,4))
#     plt.contour(X,Y,redundancy_total_pos[time_idx, :, :, latent_i, latent_j].T, cmap='plasma')
#     plt.title("Redundancy (Positive) for latent pair ({} & {})".format(latent_i, latent_j))
#     plt.colorbar()
#     plt.savefig("redundancy_pos_latent{}_{}.png".format(latent_i, latent_j))
#     plt.close()

#     # Plot the uniqueness map for latent_i (positive) for the pair (latent_i, latent_j)
#     plt.figure(figsize=(6,4))
#     plt.contour(X,Y,uniqueness_total_pos[time_idx, :, :, latent_i, latent_j, 0].T, cmap='plasma')
#     plt.title("Uniqueness (Positive) for latent {} unique vs {}".format(latent_i, latent_j))
#     plt.colorbar()
#     plt.savefig("uniqueness_pos_latent{}_unique_vs{}.png".format(latent_i, latent_j))
#     plt.close()

#     print("Finished processing iteration:", s_i)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter

def compute_causal_classification(net_shap, tol_factor=0.1):
    """
    Compute a causal classification for each pixel given the net SHAP values.

    Parameters:
        net_shap (np.array): Array of shape (N, H, W, n_vars) with net SHAP values.
        tol_factor (float): Fraction of the average positive value to use as tolerance.

    Returns:
        classification (np.array): Array of shape (N, H, W) with integer codes:
            30  : Synergy of 3 (all three latent variables are positive)
            12  : Synergy of 2 for latents (0,1) if values differ enough
            13  : Synergy of 2 for latents (0,2)
            23  : Synergy of 2 for latents (1,2)
            -12 : Redundancy of 2 for latents (0,1) if nearly equal
            -13 : Redundancy of 2 for latents (0,2)
            -23 : Redundancy of 2 for latents (1,2)
            -1  : Uniqueness of latent 0 (only latent 0 positive)
            -2  : Uniqueness of latent 1
            -3  : Uniqueness of latent 2
             0  : None positive
    """
    N, H, W, n_vars = net_shap.shape
    classification = np.zeros((N, H, W), dtype=np.int32)
    
    for s in range(N):
        for i in range(H):
            for j in range(W):
                v = net_shap[s, i, j, :]  # vector of net SHAP values for each latent at pixel (i,j)
                pos_mask = (v > 0)
                count_pos = np.sum(pos_mask)
                
                if count_pos == 3:
                    classification[s, i, j] = 30
                elif count_pos == 2:
                    pos_indices = np.where(pos_mask)[0]
                    avg_pos = np.mean(v[pos_indices])
                    tol = tol_factor * avg_pos if avg_pos != 0 else 0.01
                    # If the two positive values are nearly equal, then treat it as redundancy.
                    if abs(v[pos_indices[0]] - v[pos_indices[1]]) < tol:
                        pair = tuple(sorted(pos_indices))
                        if pair == (0, 1):
                            classification[s, i, j] = -12
                        elif pair == (0, 2):
                            classification[s, i, j] = -13
                        elif pair == (1, 2):
                            classification[s, i, j] = -23
                    else:
                        # Otherwise, classify as synergy.
                        pair = tuple(sorted(pos_indices))
                        if pair == (0, 1):
                            classification[s, i, j] = 12
                        elif pair == (0, 2):
                            classification[s, i, j] = 13
                        elif pair == (1, 2):
                            classification[s, i, j] = 23
                elif count_pos == 1:
                    pos_index = np.where(pos_mask)[0][0]
                    if pos_index == 0:
                        classification[s, i, j] = -1
                    elif pos_index == 1:
                        classification[s, i, j] = -2
                    elif pos_index == 2:
                        classification[s, i, j] = -3
                else:
                    classification[s, i, j] = 0
    return classification

# ---------------------------
# Main processing
# ---------------------------

# Assume that the following variables are already defined:
# shap_u, shap_v   : arrays of shape (size, H, W, vars) with SHAP values for channels u and v.
# sigma            : Gaussian smoothing sigma value.
# nx, ny           : 1D arrays of spatial coordinates along X and Y.
# For this example, the spatial grid is 288 points in x and 96 in y.
# Also, size and vars are defined (e.g. size = number of time steps, vars = 3).

print("Pre-smoothing SHAP fields...")
shap_u_smooth = gaussian_filter(shap_u, sigma=sigma)
shap_v_smooth = gaussian_filter(shap_v, sigma=sigma)

# Compute net SHAP (without squaring, preserving sign).
net_shap = shap_u_smooth + shap_v_smooth  # shape: (size, H, W, vars)
print('NET,', net_shap.shape)
# Compute the causal classification based on net SHAP.
classification = compute_causal_classification(net_shap, tol_factor=0.1)

# ---------------------------
# Visualization: Save three temporal snapshots
# ---------------------------

# Define time indices for snapshots (e.g., beginning, middle, end).
time_indices = [0, 485, 999]  # adjust indices if needed

# Define mapping from codes to labels.
code_to_label = {
    0: "None",
    -1: "Uniqueness 1",
    -2: "Uniqueness 2",
    -3: "Uniqueness 3",
    12: "Synergy 12",
    13: "Synergy 13",
    23: "Synergy 23",
    -12: "Redundancy 12",
    -13: "Redundancy 13",
    -23: "Redundancy 23",
    30: "Synergy 123"
}

# Define corresponding colors.
code_to_color = {
    0: "white",
    -1: "lightblue",
    -2: "lightgreen",
    -3: "lightyellow",
    12: "purple",
    13: "brown",
    23: "orange",
    -12: "blue",
    -13: "green",
    -23: "red",
    30: "pink"
}

# Prepare discrete colormap.
codes = sorted(code_to_label.keys())
boundaries = []
for i in range(len(codes)-1):
    boundaries.append((codes[i] + codes[i+1]) / 2)
boundaries = [codes[0] - 0.5] + boundaries + [codes[-1] + 0.5]

cmap_colors = [code_to_color[code] for code in codes]
cmap = mcolors.ListedColormap(cmap_colors)
norm = mcolors.BoundaryNorm(boundaries, cmap.N)

# Create a meshgrid for plotting.
# We assume that the classification has shape (size, H, W) with H=96 and W=288.
# Use the entire nx and ny arrays (length 288 and 96, respectively).
X, Y = np.meshgrid(nx[:288], ny[:96])

for t in time_indices:
    snapshot = classification[t, :, :]  # shape: (H, W)
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, snapshot, levels=boundaries, cmap=cmap, norm=norm)
    cbar = plt.colorbar(contour, ticks=codes)
    cbar.ax.set_yticklabels([code_to_label[code] for code in codes], fontsize=12)
    cbar.set_label("Causal Code", fontsize=12)
    
    plt.xlabel("X", fontsize=14)
    plt.ylabel("Y", fontsize=14)
    plt.title(f"Causal Classification at Time Step {t}", fontsize=16)
    plt.savefig(f"causal_classification_t{t}.png", dpi=200, bbox_inches='tight')
    plt.close()

print("Saved causal classification snapshots.")
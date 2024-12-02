import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter
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

# Load shap values
# with h5py.File(shap_file, 'r') as hf:
#     shap_u = np.array(hf['u'])
#     shap_v = np.array(hf['v'])
shap_u, shap_v = load_all_shap_values(0, 49)
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


# shap_u_smooth = gaussian_filter(shap_u, sigma=sigma)
# shap_v_smooth = gaussian_filter(shap_v, sigma=sigma)

# # Calculate MSE for u and v components
# mse_u = np.mean(shap_u_smooth**2, axis=(0, 1, 2))
# mse_v = np.mean(shap_v_smooth**2, axis=(0, 1, 2))
# mse = np.sqrt((mse_u + mse_v))

# print(f'MSE for shap_u: {mse_u}, MSE for shap_v: {mse_v}, MSE total: {mse}, SHAPES: {shap_u_smooth.shape}')

# # Define H values to test
# H_values = np.linspace(0.8, 1.0, 100)  # Test 100 values of H between 0 and 1
# best_H = np.zeros(10)

# # Loop through the first four modes (or any number of modes you want to process)
# for j in range(10):
#     max_structures = 0
#     appended_flag = False  # Flag to track whether structures have been appended


#     print('---------- VAR ------------:', j)
#     print('---------- MSE ------------:', mse)

#     if j==0:
#         all_structures_u = [[]]
#         all_structures_v = [[]]
#         nota = 0
#     else:
#         all_structures_u.append([])
#         all_structures_v.append([])
#         nota = 0

#     for H in H_values:
#         # Call the function to count structures using the corrected code
#         total_structures, structures_u, structures_v, nota = count_structures(H, shap_u_smooth[:, :, :, j], shap_v_smooth[:, :, :, j], mse[j], nx, ny,max_structures, nota, sigma=sigma)
#         # Check if this H produces more structures and update accordingly
#         if total_structures > max_structures:
#             max_structures = total_structures
#             best_H[j] = H
    
#     # Store the best structures for each mode

    
    
#     print(f"Best H: {best_H}, for mode {j} with Max structures: {max_structures}")
# # Create binary masks based on smoothed SHAP values exceeding the MSE

# #################################################################################
local = 0
steps = 1000
steps_1 = 1000
size = 25000
vars = 10
values = 10
total = np.zeros((vars,values))
"""
H_values = np.linspace(1, 1.5, values)
print(' INITIAL SHAPES:', shap_u.shape)
for H in H_values:
    print('H values', H)
    shap_u_smooth = gaussian_filter(shap_u, sigma=sigma)
    shap_v_smooth = gaussian_filter(shap_v, sigma=sigma)
    mse_u = np.mean(shap_u_smooth**2,axis=(0,1,2))
    mse_v = np.mean(shap_v_smooth**2,axis=(0,1,2))
    mse = np.sqrt((mse_u + mse_v))
    # print(f'MSE for shap_u: {mse_u}, MSE for shap_v: {mse_v}, MSE total: {mse}, {shap_u.shape}, {mse.shape},{mse_u.shape}')
    # Calculate the Mean Square Error (MSE) for both channels
    # H = 0.1
    #print(f'MSE for shap_u: {mse_u}, MSE for shap_v: {mse_v}, {shap_u.shape}, {mse.shape},{mse_u.shape}')
    binary_mask_u = np.where(shap_u_smooth > H * mse, 1, 0)
    binary_mask_v = np.where(shap_v_smooth > H * mse, 1, 0)
    binary_total = np.where(np.sqrt(shap_u_smooth**2 + shap_v_smooth**2) > H*mse , 1, 0)
    # print(f'------ BINARY MASK for shap_u: {binary_mask_u.shape} --------------, {H} ------- local -------, {local}')
    # Separate structure matrices for each channel
    structure_matrix_u = binary_mask_u.astype(np.float32)
    structure_matrix_v = binary_mask_v.astype(np.float32)
    structure_matrix = binary_total.astype(np.float32)

    # Print the shape of the new structure matri3ces
    # print(f'Structure matrix U shape: {structure_matrix_u.shape}')
    # print(f'Structure matrix V shape: {structure_matrix_v.shape}')

    # Initialize lists to hold structures for all time steps
    all_structures_u = []
    all_structures_v = []
    all_structures = []

    # Process each time step (first dimension of the 4D matrix)
    for j in range(vars):
        # print('VAR:',j)
        if j==0:
            all_structures_u = [[]]
            all_structures_v = [[]]
            all_structures = [[]]
        else:
            all_structures_u.append([])
            all_structures_v.append([])
            all_structures.append([])
        for t in range(steps_1):
            # Identify the structures for each time step
            # structures_u = separate_structures(len(nx[:288]), len(ny[:96]), structure_matrix_u[t,:,:,j])
            # structures_v = separate_structures(len(nx[:288]), len(ny[:96]), structure_matrix_v[t,:,:,j])
            structures = separate_structures(len(nx[:288]), len(ny[:96]), structure_matrix[t,:,:,j])

            # all_structures_u[j].append(structures_u)
            # all_structures_v[j].append(structures_v)

            all_structures[j].append(structures)

    # Use dtype=object if shapes are still irregular but you want to keep the list-like structure


    for variable in range(vars):#,9,0,1,7]:
        matrix = np.zeros((2,steps_1,96,288))
        matrix_t = np.zeros((steps_1,96,288))
        # print('----------- STRUCTURES _________ :', all_structures_u[0][0][0],all_structures_u[0][0][1],np.arange(len(all_structures_u[variable])))#, all_structures_v.shape)
        # for t in np.arange(len(all_structures_u[variable])):
        #     for i in np.arange(len(all_structures_u[variable][t])):
        #         for j in np.arange(len(all_structures_u[variable][t][i][0,:])):
        #             matrix[0,t,all_structures_u[variable][t][i][0,j],all_structures_u[variable][t][i][1,j]] = i + 1
        # for t in np.arange(len(all_structures_v[variable])):
        #     for i in np.arange(len(all_structures_v[variable][t])):
        #         for j in np.arange(len(all_structures_v[variable][t][i][0,:])):
        #             matrix[1,t,all_structures_v[variable][t][i][0,j],all_structures_v[variable][t][i][1,j]] = i + 1
        for t in np.arange(len(all_structures[variable])):
            for i in np.arange(len(all_structures[variable][t])):
                for j in np.arange(len(all_structures[variable][t][i][0,:])):
                    matrix_t[t,all_structures[variable][t][i][0,j],all_structures[variable][t][i][1,j]] = i + 1
        for t in range(steps_1):
            # total[variable,local] = len(all_structures_u[variable][t]) + len(all_structures_v[variable][t]) +  total[variable,local]
            total[variable,local] = len(all_structures[variable][t]) +  total[variable,local]


        # for m in range(2):
        #     # print(np.min(matrix),np.max(matrix))
        #     nxx, nyy = 288, 96  # Dimensions of the grid
        #     x = np.linspace(nx.min(), nx.max(), nxx)
        #     y = np.linspace(ny.min(), ny.max(), nyy)
        #     X, Y = np.meshgrid(x, y)
        #     xb = np.array([-0.125, -0.125, 0.25, 0.25])
        #     yb = np.array([0.0, 1.0, 1.0, 0.0])
        #     # Plot the matrix with physical coordinates
        #     fig, ax = plt.subplots(figsize=(10, 6))
        #     c = ax.pcolormesh(X, Y, matrix[m,0, :, :], shading='auto', cmap='plasma')

        #     # Add the object to the plot
        #     ax.fill(xb, yb, c='w', zorder=3)  # Fill object in white
        #     ax.plot(xb, yb, c='k', lw=1, zorder=5)  # Object boundary in black

        #     # Set plot limits to match the data
        #     ax.set_xlim([x.min(), x.max()])
        #     ax.set_ylim([y.min(), y.max()])

        #     # Add color bar and labels
        #     fig.colorbar(c, ax=ax, label='Structure Label')
        #     ax.set_xlabel('X Coordinate')
        #     ax.set_ylabel('Y Coordinate')
        #     ax.set_title(f'Structures for Latent Mode {variable} with H {H}')

        #     # Save and close the plot
        #     plt.savefig(f'testing/{m}/structures_plot_{m}_with_object_{variable}_perc_{H}.png', bbox_inches='tight')
        #     plt.close()
        # print(np.min(matrix),np.max(matrix))
        nxx, nyy = 288, 96  # Dimensions of the grid
        x = np.linspace(nx.min(), nx.max(), nxx)
        y = np.linspace(ny.min(), ny.max(), nyy)
        X, Y = np.meshgrid(x, y)
        xb = np.array([-0.125, -0.125, 0.25, 0.25])
        yb = np.array([0.0, 1.0, 1.0, 0.0])
        # Plot the matrix with physical coordinates
        fig, ax = plt.subplots(figsize=(10, 6))
        matrix_t[matrix_t==0]=np.nan 
        c = ax.pcolormesh(X, Y, matrix_t[0, :, :], shading='auto', cmap='plasma')

        # Add the object to the plot
        ax.fill(xb, yb, c='w', zorder=3)  # Fill object in white
        ax.plot(xb, yb, c='k', lw=1, zorder=5)  # Object boundary in black

        # Set plot limits to match the data
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([y.min(), y.max()])

        # Add color bar and labels
        fig.colorbar(c, ax=ax, label='Structure Label')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f'Structures for Latent Mode {variable} with H {H}')

        # Save and close the plot
        plt.savefig(f'testing/checks/{shap_name}/structures_plot_with_object_{variable}_perc_{H}.png', bbox_inches='tight')
        plt.close()
    local = local + 1
# Plot results after collecting data
plt.figure(figsize=(12, 6))

# Plot for individual variables
plt.figure(figsize=(12, 6))

# Plot for individual variables and mark the max H value
for var, t in enumerate(total):
    plt.plot(H_values, t, label=f'Variable {var + 1}')
    
    # Find the max value and corresponding H value for the current variable
    max_value = np.max(t)
    max_H_value = H_values[np.argmax(t)]
    
    # Plot a marker for the max value
    plt.scatter(max_H_value, max_value, marker='o', color='black')
    
    # Annotate the max value with a label
    plt.text(max_H_value, max_value, f'Max H: {max_H_value:.2f}', fontsize=9, ha='right', va='bottom')

# Labels and title
plt.xlabel('H')
plt.ylabel('Total Structures')
plt.title('Percolation')
plt.legend()
plt.savefig('perc.png', bbox_inches='tight')
plt.close()

# # Print the max H values for each variable
# for i in range(vars):
#     print(f' -------------- MAX H --------------________ VAR {i} : {np.max(total[i,:])} , {H_values[np.argmax(total[i,:])]}') 


# for i in range(vars):
#     print(f' -------------- MAX H --------------________ VAR {i} :,{np.max(total[i,:])} , {H_values[np.argmax(total[i,:])]}')
"""
#############################33 FINAL ###########################3
for s_i in range(12,int(size/steps)):
    print('PREE-SMOOTHING')
    shap_u_smooth = gaussian_filter(shap_u, sigma=sigma)
    shap_v_smooth = gaussian_filter(shap_v, sigma=sigma)
    print('MSE')
    mse_u = np.mean(shap_u_smooth**2,axis=(0,1,2))
    mse_v = np.mean(shap_v_smooth**2,axis=(0,1,2))
    mse = np.sqrt((mse_u + mse_v))
    H_values = [1.11,1.17,1.22,1,1.06,1,1.06,1.11,1.22,1.11]
    # Assuming you have defined 'steps', 'nx', 'ny', and 'vars'
    binary_total = np.zeros((size, 96, 288, vars), dtype=np.int32)  # Initialize binary_total to the required shape
    print('H_values')
    # print(f'MSE for shap_u: {mse_u}, MSE for shap_v: {mse_v}, MSE total: {mse}, {shap_u.shape}, {mse.shape},{mse_u.shape}')
    # Calculate the Mean Square Error (MSE) for both channels
    # H = 0.1
    #print(f'MSE for shap_u: {mse_u}, MSE for shap_v: {mse_v}, {shap_u.shape}, {mse.shape},{mse_u.shape}')
    for i in range(vars):
        # binary_mask_u = np.where(shap_u_smooth[:,:,:,i] > H_values[np.argmax(total[i,:])] * mse, 1, 0)
        # binary_mask_v = np.where(shap_v_smooth > H_values[np.argmax(total[i,:])] * mse, 1, 0)
        # binary_total[:,:,:,i] = np.where(np.sqrt(shap_u_smooth[:,:,:,i] **2 + shap_v_smooth[:,:,:,i] **2) > H_values[np.argmax(total[i,:])]*mse[i] , 1, 0)
        print('Hs', H_values[i])
        binary_total[:,:,:,i] = np.where(np.sqrt(shap_u_smooth[:,:,:,i] **2 + shap_v_smooth[:,:,:,i] **2) > H_values[i]*mse[i] , 1, 0)

    # Separate structure matrices for each channel
    # structure_matrix_u = binary_mask_u.astype(np.float32)
    # structure_matrix_v = binary_mask_v.astype(np.float32)
    structure_matrix = binary_total.astype(np.float32)

    # Print the shape of the new structure matri3ces
    # print(f'Structure matrix U shape: {structure_matrix_u.shape}')
    # print(f'Structure matrix V shape: {structure_matrix_v.shape}')

    # Initialize lists to hold structures for all time steps
    all_structures_u = []
    all_structures_v = []
    all_structures = []

    # Process each time step (first dimension of the 4D matrix)
    for j in range(vars):
        print('VAR:',j)
        if j==0:
            all_structures_u = [[]]
            all_structures_v = [[]]
            all_structures = [[]]
        else:
            all_structures_u.append([])
            all_structures_v.append([])
            all_structures.append([])
        for t in range(steps):
            # Identify the structures for each time step
            structures = separate_structures(len(nx[:288]), len(ny[:96]), structure_matrix[t,:,:,j])


            all_structures[j].append(structures)

    for variable in range(vars):#,9,0,1,7]
        matrix = np.zeros((2,steps,96,288))
        matrix_t = np.zeros((steps,96,288))
        frequency_matrix = np.zeros((96, 288))
        for t in np.arange(len(all_structures[variable])):
            for i in np.arange(len(all_structures[variable][t])):
                for j in np.arange(len(all_structures[variable][t][i][0,:])):
                    matrix_t[t,all_structures[variable][t][i][0,j],all_structures[variable][t][i][1,j]] = i + 1
                    frequency_matrix[all_structures[variable][t][i][0,j], all_structures[variable][t][i][1,j]] = frequency_matrix[all_structures[variable][t][i][0,j], all_structures[variable][t][i][1,j]] + 1


        nxx, nyy = 288, 96  # Dimensions of the grid
        x = np.linspace(nx.min(), nx.max(), nxx)
        y = np.linspace(ny.min(), ny.max(), nyy)
        X, Y = np.meshgrid(x, y)
        xb = np.array([-0.125, -0.125, 0.25, 0.25])
        yb = np.array([0.0, 1.0, 1.0, 0.0])
        # Plot the matrix with physical coordinates
        fig, ax = plt.subplots(figsize=(10, 6))
        matrix_t[matrix_t==0]=np.nan 
        c = ax.pcolormesh(X, Y, matrix_t[0, :, :], shading='auto', cmap='plasma')

        # Add the object to the plot
        ax.fill(xb, yb, c='w', zorder=3)  # Fill object in white
        ax.plot(xb, yb, c='k', lw=1, zorder=5)  # Object boundary in black

        # Set plot limits to match the data
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([y.min(), y.max()])

        # Add color bar and labels
        fig.colorbar(c, ax=ax, label='Structure Label')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        # ax.set_title(f'Structures for Latent Mode {variable} with H {H_values[np.argmax(total[variable,:])]}')
        ax.set_title(f'Structures for Latent Mode {variable} with H {H_values[variable]}')
        # Save and close the plot
        # plt.savefig(f'testing/best/{shap_name}/structures_plot_with_object_{variable}_perc_{ H_values[np.argmax(total[variable,:])]}.png', bbox_inches='tight')
        plt.savefig(f'testing/best/{shap_name}/structures_plot_with_object_{variable}_perc_{ H_values[variable]}.png', bbox_inches='tight')
        plt.close()

        # frequency_matrix = np.zeros((96, 288))

        # # Sum occurrences over all time steps to get the frequency of each structure
        # for t in range(steps):
        #     frequency_matrix += matrix_t[t]

        # Optionally, you might want to normalize the frequency_matrix by the number of time steps
        # frequency_matrix /= steps

        # Now you can use frequency_matrix to analyze the spatial distribution of structures
        # Plotting the frequency matrix
        nxx, nyy = 288, 96  # Dimensions of the grid
        x = np.linspace(nx.min(), nx.max(), nxx)
        y = np.linspace(ny.min(), ny.max(), nyy)
        X, Y = np.meshgrid(x, y)
        xb = np.array([-0.125, -0.125, 0.25, 0.25])
        yb = np.array([0.0, 1.0, 1.0, 0.0])
        fig, ax = plt.subplots(figsize=(10, 6))
        c = ax.pcolormesh(X, Y, frequency_matrix/np.max(frequency_matrix), shading='auto', cmap='plasma')

        ax.fill(xb, yb, c='w', zorder=3)  # Fill object in white
        ax.plot(xb, yb, c='k', lw=1, zorder=5)  # Object boundary in black
        # Set limits and labels
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([y.min(), y.max()])
        fig.colorbar(c, ax=ax, label='Frequency of Structures')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        # ax.set_title(f'Structure Frequency for Latent Mode {variable} with H {H_values[np.argmax(total[variable,:])]}')
        ax.set_title(f'Structure Frequency for Latent Mode {variable} with H {H_values[variable]}')

        # Save the plot
        plt.savefig(f'testing/pdfs/{shap_name}/frequency_matrix_{variable}.png', bbox_inches='tight')
        plt.close()




    print('SAVING',s_i)
    # # Convert the structures to 
        # Use dtype=object if shapes are still irregular but you want to keep the list-like structure
    #shap_save_path = f'/mimer/NOBACKUP/groups/deepmechalvis/marcial/structures_time_{s_i}.h5'
    shap_save_path = f'/mimer/NOBACKUP/groups/kthmech/sanchis/scratch/structures_time_{s_i}.h5'
        # Open or create the HDF5 file
    with h5py.File(shap_save_path, 'w') as hf:
        # Iterate over each variable (e.g., u, v)
        for j, structures in enumerate(all_structures):
            # Create a group for each variable
            group = hf.create_group(f'var_{j}')  # 'var_{j}' for u, v, etc.

            # Iterate over each time step (t) for the current variable
            for t, time_structures in enumerate(structures):
                # Create a group for each time step within the variable group
                time_group = group.create_group(f'time_{t}')
                
                # Iterate over each structure in the time step
                for i, structure in enumerate(time_structures):
                    # Save the u (structure[0]) and v (structure[1]) components separately
                    time_group.create_dataset(f'structure_{i}_u', data=structure[0], compression="gzip")
                    time_group.create_dataset(f'structure_{i}_v', data=structure[1], compression="gzip")

    print(f"SHAP values saved successfully to {shap_save_path}")
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde

# shap_save_path = "/mimer/NOBACKUP/groups/deepmechalvis/marcial/structures.h5"

# with h5py.File(shap_save_path, 'r') as hf:
#     all_structures = [hf[f'mode_{idx+1}'] for idx in range(len(hf.keys()))]


datafile = '../../Abhijeet2DobsData/OneObs2D-25k_z0_train-v2.h5'

with h5py.File(datafile, 'r') as f:
    u_keras = np.array(f['u_fluc'][:], dtype=np.float32)
    nt, nx, ny = f['t'][()], f['x'][()], f['y'][()]
    u_v = np.array(f['v_fluc'][:],dtype=np.float32)

    u_keras = np.transpose(u_keras[:, :288, :96], (0, 2, 1))
    u_v = np.transpose(u_v[:, :288, :96], (0, 2, 1))

# Define the path to the saved structures file
S_i = 1
# shap_save_path = f'/mimer/NOBACKUP/groups/deepmechalvis/marcial/structures_time_{s_i}.h5'
local = 0
steps = 1000
steps_1 = 1000
size = 25000
steps = size
vars = 10
values = 10
all_structures = []
shap_name = f'shap_values_{S_i}'

# Initialize the main list to collect structures from all files
all_structures = []
print('VELOCITY FIELD', u_keras.shape)
pdf_y = False
pdf_UV = True
# Loop over the range of s_i values (from 0 to 10)
for s_i in range(S_i):  # 11 because we want to include 10 (0 to 10)
    shap_save_path = f'/mimer/NOBACKUP/groups/kthmech/sanchis/scratch/STRUC/structures_time_{s_i}.h5'
    
    # Open each HDF5 file and read the structures
    with h5py.File(shap_save_path, 'r') as hf:
        for j in range(len(hf.keys())):
            var_group = hf[f'var_{j}']
            structures_for_var = []
            for t in range(len(var_group.keys())):
                time_group = var_group[f'time_{t}']
                time_structures = []
                for i in range(len(time_group.keys()) // 2):
                    structure_y = time_group[f'structure_{i}_v'][:]
                    structure_x = time_group[f'structure_{i}_u'][:]
                    time_structures.append([structure_x, structure_y])
                structures_for_var.append(time_structures)
            all_structures.append(structures_for_var)
# Initialize matrices
matrix = np.zeros((25, steps, 96, 288))
matrix_t = np.zeros((steps, 96, 288))
frequency_matrix = np.zeros((96, 288))

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
                frequency_matrix[x_coord,y_coord] += 1
    # Now that we have the structures, we can calculate the frequency matrix
    nxx, nyy = 288, 96  # Grid dimensions
    x = np.linspace(nx.min(), nx.max(), nxx)
    y = np.linspace(ny.min(), ny.max(), nyy)
    X, Y = np.meshgrid(x, y)
    xb = np.array([-0.125, -0.125, 0.25, 0.25])
    yb = np.array([0.0, 1.0, 1.0, 0.0])

    # Iterate over variables and calculate frequency matri
    # Normalize frequency matrix
    frequency_matrix /= np.max(frequency_matrix)

    # Plot the frequency matrix
    fig, ax = plt.subplots(figsize=(10, 6))
    c = ax.pcolormesh(X, Y, frequency_matrix, shading='auto', cmap='plasma')
    ax.fill(xb, yb, c='w', zorder=3)  # Fill object in white
    ax.plot(xb, yb, c='k', lw=1, zorder=5)  # Object boundary in black

    # Set plot limits and labels
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    fig.colorbar(c, ax=ax, label='Frequency of Structures')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Structure Frequency for Latent Mode {variable}')

    # Save the plot
    output_dir = f'testing/{shap_name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Now you can save the frequency matrix plot
    plt.savefig(f'{output_dir}/frequency_matrix_{variable}.png', bbox_inches='tight')
    plt.close()
    if pdf_UV == True:
        print('START UV')
        # Flatten the frequency matrix for easier indexing
        high_frequency_indices = np.where(frequency_matrix > 0.75)

        # Collect the u and v values at the selected grid points along time
        u_values = []
        v_values = []
        y_values = []

        for t in range(u_keras.shape[0]):  # Loop over time steps
            u_values.extend(u_keras[t, high_frequency_indices[0], high_frequency_indices[1]].flatten())
            v_values.extend(u_v[t, high_frequency_indices[0], high_frequency_indices[1]].flatten())
            y_values.extend(ny[high_frequency_indices[0]].flatten())

        y_bins = np.concatenate(([ny[0] - 0.5], ny[:96] + (ny[1:97]-ny[:96])/2 ))
        print('--------- FILES ---------------', ny[:96], y_bins)
        with open('output.txt', 'a') as f:
            f.write(str(np.array(ny[:96])))  # Convert the array to a string
            f.write('\n' + '-' * 100 + '\n')  # Add separators with newlines for better readability
            f.write(str(y_bins))  # Convert y_bins to a string before writing

        # Convert lists to numpy arrays
        u_values = np.array(u_values)
        v_values = np.array(v_values)
        y_values = np.array(y_values)
        hist_uy,hist_u,hist_y = np.histogram2d(u_values,y_values,bins=(144,y_bins))
        # hist_y                = hist_y[:-1]+np.diff(hist_y)/2
        hist_u                = hist_u[:-1]+np.diff(hist_u)/2
        grid_u,grid_y         = np.meshgrid(hist_u,ny[:96])
        grid_uy               = hist_uy.T.copy()

        plt.figure()
        plt.pcolor(grid_u, grid_y, grid_uy, cmap='plasma')
        plt.savefig(f'{output_dir}/joint_pdf_y_u_{variable}.png', bbox_inches='tight')
        plt.close()
        # # Calculate the joint PDF using Gaussian KDE
        # kde = gaussian_kde(np.vstack([u_values, v_values]))
        # x_min, x_max = u_values.min(), u_values.max()
        # y_min, y_max = v_values.min(), v_values.max()

        # # Generate a grid for the joint PDF
        # x_grid = np.linspace(x_min, x_max, 100)
        # y_grid = np.linspace(y_min, y_max, 100)
        # X, Y = np.meshgrid(x_grid, y_grid)
        # positions = np.vstack([X.ravel(), Y.ravel()])
        # pdf_values = kde(positions).reshape(X.shape)

        # # Plot the joint PDF as a contour map
        # fig, ax = plt.subplots(figsize=(8, 6))
        # contour = ax.contourf(X, Y, pdf_values, levels=8, cmap='plasma')
        # fig.colorbar(contour, ax=ax, label='Joint PDF')
        # ax.set_xlabel('u Velocity')
        # ax.set_ylabel('v Velocity')
        # ax.set_title(f'Joint PDF of u and v Velocities for High-Frequency Grid Points in mode {variable}')
        # plt.savefig(f'{output_dir}/joint_pdf_uv_{variable}.png', bbox_inches='tight')
        # plt.close()
    print('UV')
        # Flatten matrix_t and calculate its PDF
    if pdf_y == True:
        matrix_t_flat = matrix_t.flatten()
        kde_matrix_t = gaussian_kde(matrix_t_flat)
        pdf_matrix_t = kde_matrix_t(matrix_t_flat)

        # Identify grid points with a significant PDF value
        significant_indices = np.where(pdf_matrix_t > np.percentile(pdf_matrix_t, 5))

        # Collect the corresponding y and u values across time for these grid points
        y_values = []
        u_values = []

        for t in range(u_keras.shape[0]):  # Loop over time steps
            y_values.extend(ny[significant_indices[0]])  # Extract y values
            u_values.extend(u_keras[t, significant_indices[0], significant_indices[1]].flatten())
        print('Almost')
        # Convert lists to numpy arrays
        y_values = np.array(y_values)
        u_values = np.array(u_values)

        # Calculate the joint PDF using Gaussian KDE
        kde = gaussian_kde(np.vstack([y_values, u_values]))
        print('KDE')
        y_min, y_max = y_values.min(), y_values.max()
        u_min, u_max = u_values.min(), u_values.max()

        # Generate a grid for the joint PDF
        y_grid = np.linspace(y_min, y_max, 100)
        u_grid = np.linspace(u_min, u_max, 100)
        Y, U = np.meshgrid(y_grid, u_grid)
        positions = np.vstack([Y.ravel(), U.ravel()])
        pdf_values = kde(positions).reshape(Y.shape)

        # Plot the joint PDF as a contour map
        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(U, Y, pdf_values, levels=20, cmap='viridis')
        fig.colorbar(contour, ax=ax, label='Joint PDF')
        ax.set_xlabel('u Velocity')
        ax.set_ylabel('y Coordinate')
        ax.set_title(f'Joint PDF of y and u Velocities for Significant Grid Points {variable}')

        # Save the contour plot
        output_dir = f'testing/{shap_name}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(f'{output_dir}/joint_pdf_yu_{variable}.png', bbox_inches='tight')
        plt.close()
print(f"Frequency matrices successfully saved for all variables.")

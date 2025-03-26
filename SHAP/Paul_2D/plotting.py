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
S_i = 25
perc = 15
# shap_save_path = f'/mimer/NOBACKUP/groups/deepmechalvis/marcial/structures_time_{s_i}.h5'
local = 0
steps = 1000
steps_1 = 1000
size = 25000
steps = size
vars = 3
values = 10
all_structures = []
shap_name = f'shap_values_{S_i}_{perc}_3'

# Initialize the main list to collect structures from all files
all_structures = []
print('VELOCITY FIELD', u_keras.shape)
pdf_y = True
pdf_UV = True
pdf_yu = True
pdf_volume = True
spectrum = False
# Loop over the range of s_i values (from 0 to 10)
for s_i in range(S_i):  # 11 because we want to include 10 (0 to 10)
    #shap_save_path = f'/mimer/NOBACKUP/groups/kthmech/sanchis/scratch/STRUC/structures_time_{s_i}.h5'
    #shap_save_path = f'/mimer/NOBACKUP/groups/kthmech/sanchis/scratch/STRUC/lat_3/structures_time_{perc}_{s_i}.h5'
    shap_save_path = f'/mimer/NOBACKUP/groups/kthmech/sanchis/scratch/STRUC/lat_3/structures_time_{s_i}.h5'
    print('FILES:',s_i)
    # Open each HDF5 file and read the structures
    with h5py.File(shap_save_path, 'r') as hf:
        if s_i == 0:
            for j in range(len(hf.keys())):
                all_structures.append([])
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
                #structures_for_var.append(time_structures)
                all_structures[j].append(time_structures)
# Initialize matrices
matrix = np.zeros((25, steps, 96, 288))
volume = np.zeros((steps,96,288))
matrix_t = np.zeros((steps, 96, 288))
frequency_matrix = np.zeros((96, 288))
print('Loaded data')
# Populate matrices
for variable in range(vars):
    nxx, nyy = 288, 96  # Grid dimensions
    x = np.linspace(nx.min(), nx.max(), nxx)
    y = np.linspace(ny.min(), ny.max(), nyy)
    X, Y = np.meshgrid(x, y)
    xb = np.array([-0.125, -0.125, 0.25, 0.25])
    yb = np.array([0.0, 1.0, 1.0, 0.0])
    for t in range(len(all_structures[variable])):
        for i in range(len(all_structures[variable][t])):
            structure_x = np.array(all_structures[variable][t][i][0])  # x-coordinates
            structure_y = np.array(all_structures[variable][t][i][1])  # y-coordinates
            # print('CORRDS', x[np.max(structure_x)])
            # print('--------Y', structure_y)
            # Compute area using bounding box approach
            structure_width = y[np.max(structure_x)] - y[np.min(structure_x)]
            structure_height = x[np.max(structure_y)] - x[np.min(structure_y)] 
            structure_area = structure_width * structure_height
            # print('STRUCS',structure_area, structure_width, structure_height)
            # Store values in matrices
            for j in range(len(structure_x)):
                y_coord = structure_y[j]  # Note: structure_x represents y
                x_coord = structure_x[j]  # Note: structure_y represents x
                matrix_t[t, x_coord, y_coord] = i + 1
                volume[t, x_coord, y_coord] = structure_area
                frequency_matrix[x_coord, y_coord] += 1
    # Now that we have the structures, we can calculate the frequency matrix
    print('----------------- STRUCS END ---------------------------')


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

    c = ax.pcolormesh(X, Y, matrix_t[100,:,:], shading='auto', cmap='plasma')
    ax.fill(xb, yb, c='w', zorder=3)  # Fill object in white
    ax.plot(xb, yb, c='k', lw=1, zorder=5)  # Object boundary in black

    # Set plot limits and labels
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    fig.colorbar(c, ax=ax, label='Frequency of Structures')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Structure for Latent Mode time 100 {variable}')
    # Now you can save the frequency matrix plot
    plt.savefig(f'{output_dir}/struc_matrix_{variable}.png', bbox_inches='tight')
    plt.close()
    # Collect the u and v values at the selected grid points along time
    u_values = []
    v_values = []
    y_values = []
    volumes = []
    u_x = np.zeros((steps,96,288))
    u_v_y = np.zeros((steps,96,288))

    for t in range(u_keras.shape[0]):  # Loop over time steps
        high_frequency_indices = np.where(matrix_t[t,:,:] > 0)
        # print('HIGH', high_frequency_indices[0].shape)
        u_values.extend(u_keras[t, high_frequency_indices[0], high_frequency_indices[1]].flatten())
        u_x[t,high_frequency_indices[0], high_frequency_indices[1]] = u_keras[t, high_frequency_indices[0], high_frequency_indices[1]]
        u_v_y[t,high_frequency_indices[0], high_frequency_indices[1]] = u_v[t, high_frequency_indices[0], high_frequency_indices[1]]
        v_values.extend(u_v[t, high_frequency_indices[0], high_frequency_indices[1]].flatten())
        y_values.extend(ny[high_frequency_indices[0]].flatten())
        volumes.extend(volume[t,high_frequency_indices[0],high_frequency_indices[1]].flatten())
    
    u_values = np.array(u_values)
    v_values = np.array(v_values)
    y_values = np.array(y_values)
    volumes = np.array(volumes)

    if pdf_UV == True:
        print('START UV')
        # # Calculate the joint PDF using Gaussian KDE
        bins = 144  # Number of bins
        x_min, x_max = u_values.min(), u_values.max()
        y_min, y_max = v_values.min(), v_values.max()

        # Calculate the 2D histogram
        hist, x_edges, y_edges = np.histogram2d(u_values, v_values, bins=bins, density=True)

        # Get grid centers for plotting
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        X, Y = np.meshgrid(x_centers, y_centers)
        with h5py.File(f'/mimer/NOBACKUP/groups/kthmech/sanchis/scratch/STRUC/plot_u_v_{variable}_{perc}.h5', 'w') as hf:
            hf.create_dataset(f'u_{variable}',data=X)
            hf.create_dataset(f'v_{variable}',data=Y)
            hf.create_dataset(f'hist_{variable}',data=hist)
        print('SAVED DATA HIST2D')
        # Plot the joint PDF as a contour map
        plt.figure(figsize=(8, 6))
        contour = plt.contourf(X, Y, hist.T, levels=20, cmap='plasma')  # Transpose to align axes correctly
        plt.colorbar(contour, label='Joint PDF')
        plt.xlabel('u Velocity')
        plt.ylabel('v Velocity')
        plt.title('Joint PDF of u and v Velocities for High-Frequency Grid Points')
        output_path = f'{output_dir}/joint_pdf_uv_{perc}_{variable}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Joint PDF contour plot saved as: {output_path}")
    print('UV')
        # Flatten matrix_t and calculate its PDF
    if pdf_y == True:
        # Flatten the frequency matrix for easier indexing
        print('START Y vs U')
        y_bins = np.concatenate(([ny[0] - 0.5], ny[:96] + (ny[1:97]-ny[:96])/2 ))
        print('--------- FILES ---------------', ny[:96], y_bins)
        with open('output.txt', 'a') as f:
            f.write(str(np.array(ny[:96])))  # Convert the array to a string
            f.write('\n' + '-' * 100 + '\n')  # Add separators with newlines for better readability
            f.write(str(y_bins))  # Convert y_bins to a string before writing

        # Convert lists to numpy arrays
        hist_uy,hist_u,hist_y = np.histogram2d(v_values,y_values,bins=(144,y_bins))
        # hist_y                = hist_y[:-1]+np.diff(hist_y)/2
        hist_u                = hist_u[:-1]+np.diff(hist_u)/2
        grid_u,grid_y         = np.meshgrid(hist_u,ny[:96])
        grid_uy               = hist_uy.T.copy()
        grid_uy /= np.max(grid_uy)
        with h5py.File(f'/mimer/NOBACKUP/groups/kthmech/sanchis/scratch/STRUC/plot_y_v_{variable}_{perc}.h5', 'w') as hf:
            hf.create_dataset(f'grid_u_{variable}',data=grid_u)
            hf.create_dataset(f'grid_y_{variable}',data=grid_y)
            hf.create_dataset(f'grid_uy_{variable}',data=grid_uy)
        print('SAVED DATA HIST2D')

        plt.figure()
        plt.pcolormesh(grid_u, grid_y, np.log10(grid_uy), cmap='plasma')
        plt.colorbar(label='log10(PDF)')
        plt.xlabel('Grid U')
        plt.ylabel('Grid Y')
        plt.title(f'Joint PDF (log scale) for {variable}')
        plt.savefig(f'{output_dir}/joint_pdf_y_v_{variable}_{perc}.png', bbox_inches='tight')
        plt.close()
        
        
        print('START Y vs V')
    
    if pdf_yu == True:
         # Flatten the frequency matrix for easier indexing
        print('START Y vs U')
        y_bins = np.concatenate(([ny[0] - 0.5], ny[:96] + (ny[1:97]-ny[:96])/2 ))
        print('--------- FILES ---------------', ny[:96], y_bins)
        with open('output.txt', 'a') as f:
            f.write(str(np.array(ny[:96])))  # Convert the array to a string
            f.write('\n' + '-' * 100 + '\n')  # Add separators with newlines for better readability
            f.write(str(y_bins))  # Convert y_bins to a string before writing

        # Convert lists to numpy arrays
        hist_uy,hist_u,hist_y = np.histogram2d(u_values,y_values,bins=(144,y_bins))
        # hist_y                = hist_y[:-1]+np.diff(hist_y)/2
        hist_u                = hist_u[:-1]+np.diff(hist_u)/2
        grid_u,grid_y         = np.meshgrid(hist_u,ny[:96])
        grid_uy               = hist_uy.T.copy()
        grid_uy /= np.max(grid_uy)
        with h5py.File(f'/mimer/NOBACKUP/groups/kthmech/sanchis/scratch/STRUC/plot_y_u_{variable}_{perc}.h5', 'w') as hf:
            hf.create_dataset(f'grid_u_{variable}',data=grid_u)
            hf.create_dataset(f'grid_y_{variable}',data=grid_y)
            hf.create_dataset(f'grid_uy_{variable}',data=grid_uy)
        print('SAVED DATA HIST2D')

        plt.figure()
        plt.pcolormesh(grid_u, grid_y, np.log10(grid_uy), cmap='plasma')
        plt.colorbar(label='log10(PDF)')
        plt.xlabel('Grid U')
        plt.ylabel('Grid Y')
        plt.title(f'Joint PDF (log scale) for {variable}')
        plt.savefig(f'{output_dir}/joint_pdf_y_u_{variable}_{perc}.png', bbox_inches='tight')
        plt.close()
        
    if pdf_volume == True:
        print('VOLUMES')
        y_bins = np.concatenate(([ny[0] - 0.5], ny[:96] + (ny[1:97]-ny[:96])/2 ))
        print('--------- FILES ---------------', ny[:96], y_bins)
        with open('output.txt', 'a') as f:
            f.write(str(np.array(ny[:96])))  # Convert the array to a string
            f.write('\n' + '-' * 100 + '\n')  # Add separators with newlines for better readability
            f.write(str(y_bins))  # Convert y_bins to a string before writing

        # Convert lists to numpy arrays
        hist_uy,hist_u,hist_y = np.histogram2d(volumes,y_values,bins=(144,y_bins))
        # hist_y                = hist_y[:-1]+np.diff(hist_y)/2
        hist_u                = hist_u[:-1]+np.diff(hist_u)/2
        grid_u,grid_y         = np.meshgrid(hist_u,ny[:96])
        grid_uy               = hist_uy.T.copy()
        grid_uy /= np.max(grid_uy)
        with h5py.File(f'/mimer/NOBACKUP/groups/kthmech/sanchis/scratch/STRUC/plot_y_vol_{variable}_{perc}.h5', 'w') as hf:
            hf.create_dataset(f'grid_u_{variable}',data=grid_u)
            hf.create_dataset(f'grid_y_{variable}',data=grid_y)
            hf.create_dataset(f'grid_uy_{variable}',data=grid_uy)
        print('SAVED DATA HIST2D')

        plt.figure()
        plt.pcolormesh(grid_u, grid_y, np.log10(grid_uy), cmap='plasma')
        plt.colorbar(label='log10(PDF)')
        plt.xlabel('Grid U')
        plt.ylabel('Grid Y')
        plt.title(f'Joint PDF (log scale) for {variable}')
        plt.savefig(f'{output_dir}/joint_pdf_y_vol_{variable}_{perc}.png', bbox_inches='tight')
        plt.close()

    if spectrum == True:
        import matplotlib.ticker as ticker
        from scipy.fftpack import fft2, fftshift, fft

        # Assume u_x and u_v_y are your velocity fields (Nx, Ny, Nt)
        Nx, Ny, Nt = 288, 96, steps  
        dx, dy, dt = nx[200] - nx[199], ny[91] - ny[90], 0.005  

        # Transpose to ensure the correct shape (Nt, Ny, Nx)
        u_x = np.transpose(u_x, (2, 1, 0))
        u_v_y = np.transpose(u_v_y, (2, 1, 0))

        # Create masks for nonzero values
        mask_u = np.any(u_x != 0, axis=0)  # (Ny, Nx) mask for nonzero values
        mask_v = np.any(u_v_y != 0, axis=0)

        # Apply masks to keep only relevant values
        u_x_masked = np.where(mask_u, u_x, 0)
        u_v_y_masked = np.where(mask_v, u_v_y, 0)

        # Compute spatial power spectrum only for selected structures
        kx = np.fft.fftfreq(Nx, d=dx)
        ky = np.fft.fftfreq(Ny, d=dy)
        kx, ky = np.meshgrid(kx, ky)
        k_mag = np.sqrt(kx**2 + ky**2)

        fft_u_keras = fftshift(fft2(u_x_masked.mean(axis=-1)))
        fft_u_v = fftshift(fft2(u_v_y_masked.mean(axis=-1)))

        power_spectrum_keras = np.abs(fft_u_keras) ** 2
        power_spectrum_v = np.abs(fft_u_v) ** 2

        # Velocity power spectrum at a fixed point within structures
        valid_indices = np.argwhere(mask_u)
        x_idx, y_idx = valid_indices[len(valid_indices) // 2]  # Pick middle structure

        fft_time_keras = np.abs(fft(u_x[x_idx, y_idx, :])) ** 2
        fft_time_v = np.abs(fft(u_v_y[x_idx, y_idx, :])) ** 2
        freqs = np.fft.fftfreq(Nt, d=dt)

        # Normalize and Offset to avoid log issues
        power_spectrum_keras_offset = power_spectrum_keras + 1e-10
        power_spectrum_v_offset = power_spectrum_v + 1e-10
        power_spectrum_keras_offset /= np.max(power_spectrum_keras_offset)
        power_spectrum_v_offset /= np.max(power_spectrum_v_offset)

        # Define fixed contour levels
        max_contour_value = np.log10(np.max(power_spectrum_keras_offset))
        contour_levels = np.linspace(-1.5, max_contour_value, 6)

        # Plot Contour
        fig, (ax1, ax2) = plt.subplots(2, 2, sharex=True, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1]})

        # --- Plot contour for 'u' ---
        contour1 = ax1[0].contour(kx, ky, np.log10(power_spectrum_keras_offset.T), levels=contour_levels, cmap='plasma')
        fig.colorbar(contour1, ax=ax1[0]).set_label('Log Power Spectrum')
        ax1[0].set_title('Spatial Power Spectrum - $u$')
        ax1[0].set_ylabel('$k_y$')

        contour2 = ax2[0].contour(kx, ky, np.log10(power_spectrum_v_offset.T), levels=contour_levels, cmap='plasma')
        fig.colorbar(contour2, ax=ax2[0]).set_label('Log Power Spectrum')
        ax2[0].set_title('Spatial Power Spectrum - $v$')

        plt.savefig(f'{output_dir}/power_space_{variable}_{perc}.png', bbox_inches='tight')
        plt.close()

        # Velocity Power Spectrum at Fixed Point
        plt.figure(figsize=(6, 5))
        plt.plot(freqs[:Nt // 2], fft_time_keras[:Nt // 2], label='$u$')
        plt.plot(freqs[:Nt // 2], fft_time_v[:Nt // 2], label='$v$')
        plt.yscale('log')
        plt.xlabel('Frequency')
        plt.ylabel('Power Spectrum')
        plt.title('Velocity Power Spectrum at Fixed Point')
        plt.legend()
        plt.savefig(f'{output_dir}/power_velocity_{variable}_{perc}.png', bbox_inches='tight')
        plt.close()


print(f"Frequency matrices successfully saved for all variables.")
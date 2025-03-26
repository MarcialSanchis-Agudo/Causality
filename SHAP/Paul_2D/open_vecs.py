import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.ticker as ticker
plt.rc('font', family='monospace')    # Classic mathematical font
plt.rc('axes', labelsize=18)                 # Axis label size
plt.rc('font', size=22)                      # General font size
plt.rc('legend', fontsize=18)                # Legend font size
plt.rc('xtick', labelsize=16)                # X-tick label size
plt.rc('ytick', labelsize=16)                # Y-tick label size
plt.rcParams['mathtext.fontset'] = 'stix'   # Elegant math fonts
colors = plt.cm.plasma(np.linspace(0, 1, 13))
# Variables to compare
vars = [2,3]
d = 'u'
perc = 15
ranks = [1,3,2]
for var in range(3):
# File path to the HDF5 file
    #hdf5_file_path = f'/mimer/NOBACKUP/groups/kthmech/sanchis/scratch/STRUC/global/plot_y_{d}_{var}_lag.h5'
    hdf5_file_path = f'/mimer/NOBACKUP/groups/kthmech/sanchis/scratch/STRUC/plot_y_{d}_{var}_{perc}.h5'
    # Open the HDF5 file
    with h5py.File(hdf5_file_path, 'r') as hf:
        # Load datasets
        grid_u = hf[f'grid_u_{var}'][1:,:]
        grid_y = hf[f'grid_y_{var}'][1:,:]
        grid_uy = hf[f'grid_uy_{var}'][1:,:]

    # Limit the maximum value for the contour levels
    # max_contour_value = 0
    max_contour_value = 1
    
    # Add a small value to avoid log10(0)
    grid_uy_offset = grid_uy + 1e-100  # Adjust this offset as needed
    grid_uy_offset /= np.max(grid_uy_offset)
    # Create the contour levels
    # contour_levels = np.linspace(-1, max_contour_value, 10)  # Adjust contour levels for log scale
    contour_levels = np.linspace(0.01, max_contour_value, 10)  # Adjust contour levels for log scale


    # Plot and save individual contours
    plt.figure(figsize=(10, 8))
    #contour = plt.contourf(grid_u, grid_y, np.log10(grid_uy_offset), levels=contour_levels, cmap='plasma')
    contour = plt.contourf(grid_u, grid_y, grid_uy_offset, levels=contour_levels, cmap='plasma')
    cb = plt.colorbar(contour)
    cb.set_label('PDF Value')  # Change label to show real values
    cb.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1e}'))
    # plt.colorbar(pow(10,contour), label='log10(PDF Value)')
    plt.xlabel(f'${d}$',fontsize=18)
    plt.ylabel('$y/h$',fontsize=18)
    plt.title(f'Contour Map of Joint PDF for Mode {ranks[var]}',fontsize=18)
    plt.tight_layout()
    output_path = f'testing/shap_values_25_15_3/joint_pdf_contour_map_{var}_{d}.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Contour map saved as: {output_path}")

    # plt.figure(figsize=(10, 8))
    # contour = plt.pcolor(grid_u, grid_y, np.log10(grid_uy_offset), vmin = -1, vmax = -0.1) 
    # contour = plt.contourf(grid_u, grid_y, np.log10(grid_uy_offset), levels=contour_levels, cmap='plasma')
    # cb = plt.colorbar(contour)
    # cb.set_label('PDF Value')  # Change label to show real values
    # plt.xlabel('Grid U')
    # plt.ylabel('Grid Y')
    # plt.title(f'Contour Map of Joint PDF (log scale) for Var {var}')
    # plt.tight_layout()
    # output_path = f'testing/joint_pdf_contour_map_{var}_{d}.png'
    # plt.savefig(output_path, dpi=300)
    # plt.close()
    # print(f"Contour map saved as: {output_path}")


    # Plot comparison for specific variables
    if var in vars:
        if 'axs' not in locals():  # Initialize subplots if not already created
            fig, axs = plt.subplots(1, 1, figsize=(10, 8))
            axs.set_xlabel('$ \overline{v} $ (x-axis)', fontsize=14)
            axs.set_ylabel('$y/h$', fontsize=14)
            axs.set_title('Overlayed Contour Map of Joint PDFs')

        # Overlay contour maps on the same axis
        contour_1 = axs.contour(
            grid_u, grid_y, np.log10(grid_uy_offset),
            levels=contour_levels, linestyles ='solid' if var == vars[0] else 'dashed',cmap='tab10',# if var == vars[0] else 'viridis',
            alpha=1, label=f'Var {var}'
        )

    # Add colorbar and save overlayed plot
    if 'axs' in locals():
        fig.colorbar(contour_1, ax=axs, label='log10(PDF Value)')
        output_path_overlay = f'testing/shap_values_25_15_3/joint_pdf_overlayed_contour_map_{d}.png'
        plt.savefig(output_path_overlay, dpi=300)
        plt.close()
        print(f"Overlayed contour map saved as: {output_path_overlay}")

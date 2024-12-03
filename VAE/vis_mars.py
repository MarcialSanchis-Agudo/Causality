import sys 
sys.path.insert(0,'../Plots/')
sys.path.insert(0,'../../reactor/lib/')
import matplotlib.pyplot as plt 
import scipy.io as sio 
import numpy as np 
import pandas as pd  
import plotly.graph_objects as go
import pp_time

font_dict = {'size':25}

rootdir = '../'

#-----------------------------
# SetUp the name of file 
#-----------------------------

import pyLOM
import torch

## Set device
class colorplate:
    red = "#D23918" # luoshenzhu
    blue = "#2E59A7" # qunqing
    yellow = "#E5A84B" # huanghe liuli
    cyan = "#5DA39D" # er lv
    black = "#151D29" # lanjian
    green = "#2A6E3F" # guan lv
    brown = "#9F6027" # huang liu 
    purple = "#A76283" # zi jing pin feng 
    orange = "#EA5514" # huang dan

class PlasmaColorPlate:
    def __init__(self, num_colors=9):
        # Generate a list of colors from the 'plasma' colormap
        plasma_cmap = plt.cm.plasma
        self.colors = [plasma_cmap(i / (num_colors - 1)) for i in range(num_colors)]
        
        # Assign colors to attributes
        self.color1 = self.colors[0]
        self.color2 = self.colors[1]
        self.color3 = self.colors[2]
        self.color4 = self.colors[3]
        self.color5 = self.colors[4]
        self.color6 = self.colors[5]
        self.color7 = self.colors[6]
        self.color8 = self.colors[7]
        self.color9 = self.colors[8]

def plt_setup():
    import matplotlib.pyplot as plt
    plt.rc("font",family = "serif")
    plt.rc("font",size = 20)
    plt.rc("axes",labelsize = 16, linewidth = 2)
    plt.rc("legend",fontsize= 12, handletextpad = 0.3)
    plt.rc("xtick",labelsize = 16)
    plt.rc("ytick",labelsize = 16)
    return
plt_setup()

def load_data(case_name):
    """Load the temporal predcitions"""
    dmat    = sio.loadmat( rootdir + pathsBib.res_temp + "pred_Time_" + case_name + '.mat')
    p       = dmat['p']
    g       = dmat['g']

    nt      = len(p)
    time    = np.linspace(0, 219220,nt)
    
    ts      = time[-nt:]

    return p, g, ts

def attractor(x,y,z,fig=None, axs = None):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Example temporal signals
    # Replace these with your actual data

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    axs = fig.add_subplot(111, projection='3d')

    # Plot the attractor
    axs.plot(x, y, z, lw=0.8, color='b')

    # Add labels and a title
    axs.set_xlabel('X Signal')
    axs.set_ylabel('Y Signal')
    axs.set_zlabel('Z Signal')
    axs.set_title('3D Attractor Plot')   

    return fig, axs



def vis_Series(featureList:list,
                p,
                g,
                t,
                fig=None, axs = None,color = None):
    """Visualisation of the predictions"""
    numFeature = len(featureList)
    ncol        = 2
    nrow        = int(np.ceil(numFeature / ncol))
    if fig == None and axs == None:
        fig, axs = plt.subplots(nrow, ncol,figsize=(20, 18),sharex=True)
    if color == None: color = colorplate.red
    axss    = axs.flatten()
    for i, lable in enumerate(featureList):
        axss[i].plot(t,g[:,i],
                    lw=1.5,c = colorplate.black, 
                    # label='Reference',
                    )
        axss[i].plot(t,p[:,i],
                    lw=1.5,c = color,alpha = 0.95,
                    # label='Prediction',
                    )
        axss[i].set_title(lable,font_dict)
    k = 0 
    for il in range(len(featureList)+1, len(axss)):
        axss[il].axis('off')
        k+=1
    axs[-1,-1].axis('off')
    axs[-1,0].set_xlabel('t',font_dict)
    axss[k-2].set_xlabel('t',font_dict)
    fig.subplots_adjust(hspace= 0.3)
    return fig, axs 


def vis_Single_PSD(featureName,
                p,
                if_note=False,
                fig=None, axs = None,color = None):
    """Visualisation of the predictions"""
    from pp_time import Power_Specturm_Density
    fs = 1
    window_size = 912
    if fig == None: fig,axs= plt.subplots(1,1,figsize=(7,6))
    fig,axs = Power_Specturm_Density(p,
                                    fs=fs,
                                    window_size=window_size,
                                    fig = fig, axs = axs,
                                    color= color,
                                    if_note=if_note
                                    )
                
    # axs.set_title(featureName)
    axs.set_xlim(0,0.03)
    axs.set_ylabel("PSD",font_dict)
    return fig, axs 

def vis_Single_Series(
                p,
                g,
                t,
                fig=None, axs = None,color = None):
    """Visualisation of the predictions"""
    
    if fig == None and axs == None:
        fig, axs = plt.subplots(1, 1,figsize=(12,4),sharex=True)
    if color == None: color = colorplate.red
    axs.plot(t,g,
            lw=1.5,c = colorplate.black, 
            )
    axs.plot(t,p,
            lw=1.5,c = color,alpha = 1,
            )
    #axs.set_title(featureName,font_dict)
    # axs.set_ylim(0.9 * g.min(), 1.1 * g.max())
    k = 0 
    axs.set_xlabel('t [s]',font_dict)
    return fig, axs 


def vis_smos_single(featureName, p, g, t, fig=None, axs=None, color=None):
    import numpy as np
    from scipy.stats import gaussian_kde
    from matplotlib import pyplot as plt

    if fig is None and axs is None:
        fig, axs = plt.subplots(1, 1, figsize=(7, 6), sharex=True)
    if color is None:
        color = colorplate.red

    kde1 = gaussian_kde(g, bw_method=0.2)
    values1 = np.linspace(min(g), max(g), 2000)
    probabilities1 = kde1.evaluate(values1)

    kde2 = gaussian_kde(p, bw_method=0.2)
    values2 = np.linspace(min(p), max(p), 2000)
    probabilities2 = kde2.evaluate(values2)

    axs.plot(values1, probabilities1, lw=1.5, c=colorplate.black)
    axs.plot(values2, probabilities2, lw=1.5, c=color, alpha=1)

    #axs.set_title(featureName,y=1.025)

    # Calculate safe margins
    x_range = max(g) - min(g)
    y_range = max(probabilities1) - min(probabilities1)
    x_margin = 0.3 * x_range  # 10% margin
    y_margin = 0.1 * y_range  # 10% margin

    # Set x and y limits with safe margins
    x_min, x_max = min(g) - x_margin, max(g) + x_margin
    y_min, y_max = min(probabilities1) - y_margin, max(probabilities1) + y_margin
    axs.set_xlim(x_min, x_max)
    axs.set_ylim(y_min, y_max)
    axs.set_xticks([x_min, x_max])
    axs.set_ylabel("PDF",font_dict)
    axs.set_xlabel(f'{featureName}',font_dict)
    axs.tick_params(axis='both', which='major', labelsize=font_dict['size'])



    return fig, axs



def vis_smos(featureName,
                p,
                if_note=False,
                fig=None, axs = None,color = None):
    """Visualisation of the predictions"""
    from lib.pp_time import Power_Specturm_Density
    fs = 1
    window_size = 912
    if fig == None: fig,axs= plt.subplots(1,1,figsize=(7,6))
    fig,axs = smos(p,
                    fs=fs,
                    window_size=window_size,
                    fig = fig, axs = axs,
                    color= color,
                    if_note=if_note
                    )
    axs.set_xlim(33000,55000)
    axs.set_ylim(3.5*(10**(-5)),1.3*(10**(-5)))            
    # axs.set_title(featureName)
    axs.set_ylabel("PDF")
    axs.set_xticks([33000, 55000])
    return fig, axs 

def vis_Single_PDF(featureName,
                p,
                if_note=False,
                fig=None, axs = None,color = None):
    """Visualisation of the predictions"""
    from lib.pp_time import vis_single_PDF
    fs = 1
    window_size = 912
    if fig == None: fig,axs= plt.subplots(1,1,figsize=(7,6))
    fig,axs = vis_single_PDF(p,
                                    fs=fs,
                                    window_size=window_size,
                                    fig = fig, axs = axs,
                                    color= color,
                                    if_note=if_note
                                    )
                
    # axs.set_title(featureName)
    axs.set_xlim(0,0.03)
    axs.set_ylabel("PSD")
    return fig, axs 

def smos(signals,
        fs,
        window_size,
        fig = None,
        axs = None,
        color = None,
        if_note = True):

    import numpy as np
    from numpy.random import normal
    from numpy import hstack
    from numpy import asarray
    from numpy import exp
    from scipy.stats import gaussian_kde
    from matplotlib import pyplot as plt
    if fig == None and axs == None: fig, axs = plt.subplots(1,1,figsize=(6,4))
    if color == None: color = colorplate.black

    

    kde1 = gaussian_kde(signals.T)
    values1 = np.linspace(min(signals), max(signals), 1000)
    probabilities1 = kde1.evaluate(values1).reshape(values1.shape)

    """
    # Fit density for data2
    kde2 = gaussian_kde(sample2, bw_method=0.2)
    values2 = np.linspace(min(sample2), max(sample2), 1000)
    probabilities2 = kde2.evaluate(values2)
    """

    # Plot KDE for data1
    axs.plot(values1,probabilities1,
            lw=1.5,c = color, 
            )
    #axs.set_xlabel(r"$Hz$")

    return fig, axs

def plot_3d_joint_pdf(data1, data2, data3, title1, title2, title3, color2, color3):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Fit kernel density estimators for the three datasets
    kde1 = gaussian_kde(data1.T)
    kde2 = gaussian_kde(data2.T)
    kde3 = gaussian_kde(data3.T)

    # Define the range for the grid
    x_min, x_max = np.min([data1[:, 0], data2[:, 0], data3[:, 0]]), np.max([data1[:, 0], data2[:, 0], data3[:, 0]])
    y_min, y_max = np.min([data1[:, 1], data2[:, 1], data3[:, 1]]), np.max([data1[:, 1], data2[:, 1], data3[:, 1]])
    z_min, z_max = np.min([data1[:, 2], data2[:, 2], data3[:, 2]]), np.max([data1[:, 2], data2[:, 2], data3[:, 2]])
    
    x_grid, y_grid, z_grid = np.mgrid[x_min:x_max:50j, y_min:y_max:50j, z_min:z_max:50j]
    positions = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])

    # Evaluate the PDFs at each point on the grid
    pdf1 = kde1(positions).reshape(x_grid.shape)
    pdf2 = kde2(positions).reshape(x_grid.shape)
    pdf3 = kde3(positions).reshape(x_grid.shape)

    # Plot the 3D surfaces
    ax.contour3D(x_grid, y_grid, z_grid, pdf1, levels=6, cmap='viridis', alpha=0.6)
    ax.contour3D(x_grid, y_grid, z_grid, pdf2, levels=6, colors=color2, linestyles="dashed")
    ax.contour3D(x_grid, y_grid, z_grid, pdf3, levels=6, colors=color3, linestyles="dotted")

    # Set axis labels and title
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Joint PDF Contours')

    # Add legend
    ax.legend([title1, title2, title3], loc='best')

    plt.tight_layout()

    return fig, ax
# Function to calculate KDE and plot contours
def plot_contours(data1, data2, title1, title2,color):

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    # Create a new figure

    fig, ax = plt.subplots(figsize=(12, 8))

    # Fit kernel density estimators for the two datasets
    kde1 = gaussian_kde(data1.T)
    kde2 = gaussian_kde(data2.T)
    
    # Define the range for the contour plot
    # Generate a grid of points to evaluate the PDF
# Generate a grid of points to evaluate the PDF
    x_min, x_max = np.min(data1[:, 0]), np.max(data1[:, 0])
    y_min, y_max = np.min(data1[:, 1]), np.max(data1[:, 1])
    x_grid, y_grid = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])

        # Evaluate the PDF at each point on the grid for both datasets
    pdf1 = kde1(positions).reshape(x_grid.shape)
    pdf2 = kde2(positions).reshape(x_grid.shape)
    levels = np.linspace(0, max(np.max(pdf1), np.max(pdf2)), 7)
    # Plot the contours for the first dataset
    #ax.contour(X, Y, pdf1, cmap='viridis', linewidths=3)
    #ax.scatter(data1[:, 0], data1[:, 1], color='blue', alpha=0.5, label=title1)
    ax.contour(x_grid, y_grid, pdf1, levels=levels, colors='black',linewidths=1,alpha=0.8)
    # Plot dashed lines for the second dataset
    #ax.contour(X, Y, pdf2, cmap='Reds', linestyle='dashed', linewidths=2)
    #ax.scatter(data2[:, 0], data2[:, 1], color='red', alpha=0.5, label=title2)
    ax.contour(x_grid, y_grid, pdf2, levels=levels, colors = color,linestyles='dashed',linewidths=1,alpha=0.8)
    #ax.set_xlim(2*x_min, 2*x_max)
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_tick_params(labelbottom=False)
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    # Add labels and legend
    ax.set_title('Contours of Multivariate Kernel Density Estimation')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.legend([title1, title2])

    plt.tight_layout() 

    return fig

def plot_contours_single(data, title,fig = None, ax = None ):

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    # Create a new figure

    


    # Fit kernel density estimator to the data
    kde = gaussian_kde(data.T)

    if data.shape[1] ==2:
        fig, ax = plt.subplots(figsize=(12, 8))
        # Define the range for the contour plot and generate a grid
        x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
        y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
        x_grid, y_grid = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([x_grid.ravel(), y_grid.ravel()])

    # Evaluate the PDF at each point on the grid
        pdf = kde(positions).reshape(x_grid.shape)

        # Define levels to cover the entire range of the PDF
        min_pdf = np.min(pdf[pdf > 0])  # Ignore zero values to avoid log issues
        max_pdf = np.max(pdf)
        levels = np.linspace(min_pdf, max_pdf, 7)

        # Plot the contours
        contour = ax.contour(x_grid, y_grid, pdf, levels=levels, colors='black', linewidths=1, alpha=0.8)

        # Add contour labels directly onto the contour lines
        ax.clabel(contour, inline=True, fontsize=10, fmt='%.2e')  # Use scientific notation

        # Set axis labels and title
        ax.set_title(title)
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')

        # Customize the ticks to include actual data values
        x_ticks = np.linspace(x_min, x_max, num=6)  # Adjust the number of ticks as needed
        y_ticks = np.linspace(y_min, y_max, num=6)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{tick:.5f}' for tick in x_ticks])  # Increased precision
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{tick:.5f}' for tick in y_ticks])
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_title('3D Joint PDF Contours')  # Increased precision

    else:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
        y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
        z_min, z_max = np.min(data[:, 2]), np.max(data[:, 2])
        
        x_grid, y_grid,z_grid = np.mgrid[x_min:x_max:100j, y_min:y_max:100j, z_min:z_max:100j]
        positions = np.vstack([x_grid.ravel(), y_grid.ravel(),z_grid.ravel()])
        pdf = kde(positions).reshape(x_grid.shape)
        # Define levels to cover the entire range of the PDF
        min_pdf = np.min(pdf[pdf > 0])  # Ignore zero values to avoid log issues
        max_pdf = np.max(pdf)
        levels = np.linspace(min_pdf, max_pdf, 7)
        ax.contour3D(x_grid, y_grid, z_grid[:,:,1], pdf, levels = levels, cmap='viridis', alpha=0.6)


    # Set axis labels and title

    # Show grid for better readability
    ax.grid(True)

    # Ensure that the legend shows the title
    ax.legend([title])

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    return fig, ax

def phase_energy(featureName,
                p,
                if_note=False,
                fig=None, axs = None,color = None):

    import numpy as np
    from numpy.random import normal
    from numpy import hstack
    from numpy import asarray
    from numpy import exp
    from scipy.stats import gaussian_kde
    from matplotlib import pyplot as plt
    
    if fig == None and axs == None:
        fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    
    if color == colorplate.black:
        linestyle = '-'
    else:
        linestyle = '--'

    # Calculate energy of the signal
    linestyle=linestyle
    energy = p**2

    # Plot phase space
    axs.plot(p, energy, lw=1.5, c=color)

    return fig, axs

def phase_momentum(featureName,
                p,maps,
                if_note=False,
                fig=None, axs=None,
                color=None):

    import numpy as np
    from numpy.random import normal
    from numpy import hstack
    from numpy import asarray
    from numpy import exp
    from scipy.stats import gaussian_kde
    from matplotlib import pyplot as plt
    from matplotlib.colors import Normalize
    from scipy.interpolate import interp1d
    
    if fig == None and axs == None:
        fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    
    if color == colorplate.black:
        linestyle = '-'
    else:
        linestyle = ''
        color_map = plt.get_cmap(maps)

    # Calculate energy of the signal
    energy = np.gradient(p,edge_order=2)

    if linestyle:  # Plot using lines
        x_interp = np.linspace(min(p), max(p), 2000)  # Adjust the number of points as needed
        f = interp1d(p, energy, kind='cubic')  # Use cubic interpolation
        energy_interp = f(x_interp)
        #axs.plot(x_interp, energy_interp, lw=1.5, c=color, linestyle=linestyle)
        axs.plot(p, energy, lw=1.5, c=color, linestyle=linestyle)
    else:  # Plot using colormap
        axs.scatter(p, energy, c=color, alpha=0.5)
    


    return fig, axs



def load_csv(case_name):
    import pandas as pd 
    df_mse = pd.read_csv(rootdir + pathsBib.res_temp + "mse_Time_" + case_name + '.csv')
    df_rmse = pd.read_csv(rootdir + pathsBib.res_temp + "rmse_Time_" + case_name + '.csv')
    df_l2 = pd.read_csv(rootdir + pathsBib.res_temp + "l2_Time_" + case_name + '.csv')
    return df_mse, df_rmse,df_l2

vars = [0,3,10,6]
features = ["core in P","IHX secondary in P", "Core Energy [W]","Core flow [kg/s]"] 
maps = ['plasma', 'inferno', 'plasma', 'viridis']
plasma_cp = PlasmaColorPlate()
colors = [plasma_cp.color1, plasma_cp.color4, plasma_cp.color8] # azul,morado,amarillo 
if __name__ == '__main__':
        device = pyLOM.NN.select_device()
        ## Specify autoencoder parameters
        ptrain      = 0.8
        pvali       = 0.2
        batch_size  = 1024
        nepochs     = 2000
        nlayers     = 3
        channels    = 64
        lat_dim     = 3
        beta        = 1e+00
        beta_wmup   = 1000
        kernel_size = 4
        nlinear     = 512
        padding     = 1
        activations = [pyLOM.NN.silu(), pyLOM.NN.silu(), pyLOM.NN.silu(), pyLOM.NN.silu(), pyLOM.NN.silu(), pyLOM.NN.silu()]
        batch_norm  = False
        vae         = True

        video = True


        DATAFILE = '/mimer/NOBACKUP/groups/deepmechalvis/marcial/data/dataset_minimal_channel_regular.h5'
        VARIABLE = 'velox'
        RESUSTR = 'vae_beta_1.00e+00_ld_3_batch_256_nlinear_512_fused/model_state'

        #--------------------
        # Read the prediction and the csv
        #--------------------
        ## Load the dataset
        pyldtset = pyLOM.Dataset.load(DATAFILE)
        u        = pyldtset[VARIABLE]
        um       = pyLOM.math.temporal_mean(u)
        u_x      = pyLOM.math.subtract_mean(u, um)
        ux       = u_x[:,:]
        time     = pyldtset.time 
        t        = time[:]
        mesh     = pyldtset.mesh

        nz = len(np.unique(mesh.x)) 
        ny = len(np.unique(mesh.y))
        nx = len(np.unique(mesh.z))

        # Create the torch dataset
        tordtset = pyLOM.NN.Dataset3D((ux,), nx, ny, nz, t, transform=False, device=device)

        ## Set and train the Autoencoder
        encarch = pyLOM.NN.Encoder3D(nlayers, lat_dim, nx, ny, nz, tordtset.n_channels, channels, kernel_size, padding, activations, nlinear, batch_norm=batch_norm, vae=vae)
        decarch = pyLOM.NN.Decoder3D(nlayers, lat_dim, nx, ny, nz, tordtset.n_channels, channels, kernel_size, padding, activations, nlinear, batch_norm=batch_norm)
        vae = pyLOM.NN.VariationalAutoencoder(lat_dim, (nx, ny, nz), tordtset.n_channels, encarch, decarch, device=device)
        vae.load_state_dict(torch.load(RESUSTR))

        print('--------- OPEN POD ---------')
        cov = np.sqrt(len(time[:]) - 1)
        pod_data = np.load("/mimer/NOBACKUP/groups/deepmechalvis/marcial/data/POD/POD-m100-n8810.npz")
        s = pod_data['s'][:]
        space = pod_data['modes'][:]
        vh = pod_data['vh'][:]
        # nt, nx, ny, nz = modes.shape
        # modes = modes.reshape((nt, -1)).T
        print('Prepare', space.shape,vh.shape,s.shape)
        print('------ EIGN --------:', s[:10])
        u_p = space @ np.diag(s) @ vh
        u_p *= cov
        U_p = np.zeros_like(space)
        for i in range(10):
            U_p[:,i] = space[:,i] * s[i]
        ## Reconstruct dataset and compute accuracy
        rec  = vae.reconstruct(tordtset)
        # Get latent space vectors and modes
        recdtset = pyLOM.NN.Dataset3D((rec,), nx, ny, nz, t, transform=False)

        z = vae.latent_space(tordtset)
        modes = vae.modes()
        lat_modes = modes.reshape(rec.shape[1],lat_dim)

        modeset = pyLOM.NN.Dataset3D((lat_modes[:,0],lat_modes[:,1],lat_modes[:,2]), nx, ny, nz,tordtset._time, transform=False)
        podset = pyLOM.NN.Dataset3D((U_p[:,0],U_p[:,1],U_p[:,2],U_p[:,3],U_p[:,4],U_p[:,5],U_p[:,6],U_p[:,7],U_p[:,8],U_p[:,9],U_p[:,10]), nx, ny, nz,tordtset._time, transform=False)
        
        pyldtset.add_variable('U1', False, 1, podset.data[0][:])
        pyldtset.add_variable('U2', False, 1, podset.data[1][:])
        pyldtset.add_variable('U3', False, 1, podset.data[2][:])
        pyldtset.add_variable('U4', False, 1, podset.data[3][:])
        pyldtset.add_variable('U5', False, 1, podset.data[4][:])
        pyldtset.add_variable('U6', False, 1, podset.data[5][:])
        pyldtset.add_variable('U7', False, 1, podset.data[6][:])
        pyldtset.add_variable('U8', False, 1, podset.data[7][:])
        pyldtset.add_variable('U9', False, 1, podset.data[8][:])
        pyldtset.add_variable('U10', False, 1, podset.data[9][:])
        pyldtset.add_variable('m1', False, 1, modeset.data[0][:])
        pyldtset.add_variable('m2', False, 1, modeset.data[1][:])
        pyldtset.add_variable('m3', False, 1, modeset.data[2][:])
        pyldtset.add_variable('urec', False, 1, recdtset.data[0][:,:])
        pyldtset.add_variable('utra', False, 1, tordtset.data[0][:,:])
        if video:
            pyldtset.write('reco',basedir='regular_snaps_mars_beta_%.2e' % beta,instants=np.arange(10,dtype=np.int32),times=t,vars=['U1','U2','U3','U4','U5','U6','U7','U8','U9','U10','m1','m2','m3','utra'],fmt='vtkh5')

        print('------- SAVING DATA TRANSFOMER -----------')
        split_index = int(0.8 * len(z[:, 0]))
        Z_vector_train = z[:split_index, :].cpu()
        Z_vector_test = z[split_index:, :].cpu()
        Z_vector_total = z[:, :].cpu()

        print(f"The shape of sets are {Z_vector_train.shape, Z_vector_test.shape,Z_vector_total.shape}")

        # Create a DataFrame for training data with mode names as column headers
        mode_names = [f"mode_{i+1}" for i in range(lat_modes.shape[1])]
        df_train = pd.DataFrame(Z_vector_train, columns=mode_names)

        # Create a DataFrame for testing data with mode names as column headers
        df_test = pd.DataFrame(Z_vector_test, columns=mode_names)

        df_total = pd.DataFrame(Z_vector_total, columns=mode_names)

        # Save the training set to an HDF5 file
        df_train.to_hdf('Z_vector_train.h5', key='train', mode='w', format='table')

        # Save the testing set to an HDF5 file
        df_test.to_hdf('Z_vector_test.h5', key='test', mode='w', format='table')

        df_total.to_hdf('Z_vector_total.h5', key='test', mode='w', format='table')


        # Print the shapes of the datasets to confirm
        print(f"Training data shape: {df_train.shape}")
        print(f"Testing data shape: {df_test.shape}")

        corr, detR = vae.correlation(tordtset)
        print(detR)
        plt.figure(figsize=(8,6))
        plt.imshow(np.abs(corr), cmap='seismic', interpolation='none', vmin=0, vmax=1)
        plt.colorbar()
        plt.title(r'$\det{R} = $ %.2f' % detR)
        plt.xticks(range(len(corr)), range(len(corr)), weight='bold', fontsize=16)
        plt.yticks(range(len(corr)), range(len(corr)), weight='bold', fontsize=16)
        plt.tight_layout
        plt.show()
        plt.savefig('correlation.pdf' , dpi=300)

        print('----- SHAPES ------------', z.shape, ux.shape,nx,ny,nz)
        print('----- COORDINATES ------------', mesh.x.shape,  mesh.y.shape,  mesh.z.shape,'--- REC ------', rec.shape, modes.shape)
        plt.figure(figsize=(7, 5))
        plt.scatter(z[:, 0].cpu(), z[:, 1].cpu(), c='blue', alpha=0.5)
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title('Latent Space Representation')
        plt.savefig('latent_space_representation.pdf')  # Save the plot as a PNG file
        plt.close()  # Close the plot to free up memory

        # 2. Plot and save the modes visualization
        modes = modes.reshape(nx, ny,nz,lat_dim)
        print('MODES:',modes[:, :,nz//2, 0].T.shape)  # Reshape if necessary
        plt.figure(figsize=(8, 6))
        plt.imshow(modes[:, :,nz//2, 0], cmap='plasma', aspect='auto',extent=[mesh.x.min(), mesh.x.max(), mesh.y.min(), mesh.y.max()])
        plt.colorbar(label='Mode Value')
        plt.xlabel('X Dimension')
        plt.ylabel('Y Dimension')
        plt.title('First Mode Visualization')
        print('MODES:',modes[:, :,nz//2, 0].T.shape)  # Reshape if necessary
        plt.figure(figsize=(8, 6))
        plt.imshow(modes[:, :,nz//2, 0], cmap='plasma', aspect='auto',extent=[mesh.x.min(), mesh.x.max(), mesh.y.min(), mesh.y.max()])
        plt.colorbar(label='Mode Value')
        plt.xlabel('X Dimension')
        plt.ylabel('Y Dimension')
        plt.title('First Mode Visualization')
        plt.savefig('mode_visualization_1.jpg',dpi = 300,bbox_inches="tight")  # Save the plot as a PNG file
        plt.close()  # Close the plot to free up memory
        
        plt.figure(figsize=(8, 6))
        plt.imshow(modes[nx//2, :,:, 1], cmap='plasma', aspect='auto',extent=[mesh.x.min(), mesh.x.max(), mesh.y.min(), mesh.y.max()])
        plt.colorbar(label='Mode Value')
        plt.xlabel('X Dimension')
        plt.ylabel('Y Dimension')
        plt.title('Second Mode Visualization')
        plt.savefig('mode_visualization_2.jpg',dpi = 300,bbox_inches="tight")  # Save the plot as a PNG file
        plt.close()  # Close the plot to free up memory


        plt.figure(figsize=(8, 6))
        plt.imshow(modes[:, ny //2,:, 2], cmap='plasma', aspect='auto',extent=[mesh.x.min(), mesh.x.max(), mesh.y.min(), mesh.y.max()])
        plt.colorbar(label='Mode Value')
        plt.xlabel('X Dimension')
        plt.ylabel('Y Dimension')
        plt.title('Third Mode Visualization')
        plt.savefig('mode_visualization_3.jpg',dpi = 300,bbox_inches="tight")  # Save the plot as a PNG file
        plt.close()  # Close the plot to free up memory

        mse_df  = pd.DataFrame()
        rmse_df = pd.DataFrame()
        l2_df = pd.DataFrame()
        

        dist_units  = 0.5
        l1 = z[:, 0].cpu()
        l2 = z[:, 1].cpu()
        l3 = z[:, 2].cpu()
        print('LATENT')
        fig2 = None; axs2 = None
        fig3 = None; axs3 = None
        fig5 = None; axs5 = None
        fig6 = None; axs6 = None
        fig = None;  axs = None
        fig1 = None; ax1 = None
        fig4 = None; ax4 = None
        fig8 = None; ax8 = None
        fig9 = None; ax9 = None
        fig10 = None; ax10 = None
        fig11 = None; ax11 = None
        fig12 = None; ax12 = None
        fige = None; axe = None
        print('------- PLOTTing -----------')
        fig, axs = attractor(l1,l2,l3, fig=fig,axs=axs)
        fig8, ax8 = plt.subplots(1, 2, figsize=(10, 11))
        fig9, ax9 = plt.subplots(1, 2, figsize=(10, 11))
        fig10, ax10 = plt.subplots(1, 2, figsize=(10, 11))
        fig11, ax11 = plt.subplots(1, 2, figsize=(10, 11))
        
        fig2, axs2 = vis_Single_Series(
                l1,
                l2,
                t,
                fig=fig2,
                axs=axs2,
                color=colorplate.red,
                )
        fig1, ax1 = plot_contours_single(z[:,:2].cpu().numpy(),'Joint PDF',fig = fig1, ax = ax1)
        fig4, ax4 = plot_contours_single(z[:,1:].cpu().numpy(),'Joint PDF',fig = fig4, ax = ax4)
        #fige, axe = plot_contours_single(z[:,:].cpu().numpy(),'Joint PDF',fig = fige, ax = axe)
        fig5, axs5 = vis_smos_single(
                        'Latent variables',
                        l1,
                        l2,
                        t,
                        fig=fig5,
                        axs=axs5,
                        color=colorplate.red,
                        ) 
        fig3, axs3 = vis_Single_Series(
                l1,
                l3,
                t,
                fig=fig3,
                axs=axs3,
                color=colorplate.red,
                )
        fig6, axs6 = vis_smos_single(
                    'Latent variables',
                    l1,
                    l3,
                    t,
                    fig=fig6,
                    axs=axs6,
                    color=colorplate.red,
                    )
        U_p_1 = U_p[:,0]
        U_p_2 = U_p[:,1]
        U_p_3 = U_p[:,2]
        U_p_1 = U_p_1.reshape(nx,ny,nz)
        h1 = ax8[0].imshow(modes[nx//2,:,:,0],cmap='plasma', aspect='auto', origin='lower',
                   extent=[mesh.x.min(), mesh.x.max(), mesh.y.min(), mesh.y.max()])
        ax8[0].set_xlabel('X')
        ax8[0].set_ylabel('Y')
        ax8[0].set_title(f'VAE Mode 1')
        fig8.colorbar(h1, ax=ax8[0])

        h2 = ax8[1].imshow(U_p_1[nx//2,:,:], cmap='plasma', aspect='auto', origin='lower',
                   extent=[mesh.x.min(), mesh.x.max(), mesh.y.min(), mesh.y.max()])
        ax8[1].set_xlabel('X')
        ax8[1].set_ylabel('Y')
        ax8[1].set_title(f'POD MODE 1')
        fig8.colorbar(h2, ax=ax8[1], label='Velocity')
        plt.subplots_adjust(wspace=0.8)
        fig8.savefig('Modes_1_XY.pdf',dpi = 8, bbox_inches="tight")
        plt.close(fig8) 

        fig8 = None; ax8 = None
        fig8, ax8 = plt.subplots(1, 2, figsize=(10, 11))
        h1 = ax8[0].imshow(modes[:,:,nz//2,0],cmap='plasma', aspect='auto', origin='lower',
                   extent=[mesh.x.min(), mesh.x.max(), mesh.y.min(), mesh.y.max()])
        ax8[0].set_xlabel('X')
        ax8[0].set_ylabel('Y')
        ax8[0].set_title(f'VAE Mode 1')
        fig8.colorbar(h1, ax=ax8[0])

        h2 = ax8[1].imshow(U_p_1[:,:,nz//2], cmap='plasma', aspect='auto', origin='lower',
                   extent=[mesh.x.min(), mesh.x.max(), mesh.y.min(), mesh.y.max()])
        ax8[1].set_xlabel('X')
        ax8[1].set_ylabel('Y')
        ax8[1].set_title(f'POD MODE 1')
        fig8.colorbar(h2, ax=ax8[1], label='Velocity')
        plt.subplots_adjust(wspace=0.8)
        fig8.savefig('Modes_1_ZY.pdf',dpi = 8, bbox_inches="tight")
        plt.close(fig8) 

        print('Y start')
        fig8 = None; ax8 = None
        fig8, ax8 = plt.subplots(1, 2, figsize=(10, 11))
        h1 = ax8[0].imshow(modes[:,ny//4 :,0],cmap='plasma', aspect='auto', origin='lower',
                   extent=[mesh.x.min(), mesh.x.max(), mesh.y.min(), mesh.y.max()])
        ax8[0].set_xlabel('X')
        ax8[0].set_ylabel('Y')
        ax8[0].set_title(f'VAE Mode 1')
        fig8.colorbar(h1, ax=ax8[0])

        h2 = ax8[1].imshow(U_p_1[:,ny//4,:], cmap='plasma', aspect='auto', origin='lower',
                   extent=[mesh.x.min(), mesh.x.max(), mesh.y.min(), mesh.y.max()])
        ax8[1].set_xlabel('X')
        ax8[1].set_ylabel('Y')
        ax8[1].set_title(f'POD MODE 1')
        fig8.colorbar(h2, ax=ax8[1], label='Velocity')
        plt.subplots_adjust(wspace=0.8)
        fig8.savefig('Modes_1_XZ.pdf',dpi = 8, bbox_inches="tight")
        plt.close(fig8) 
        print('Y end')

        U_p_2 = U_p_2.reshape(nx,ny,nz)
        h1 = ax10[0].imshow(modes[nx//2,:,:,1],cmap='plasma', aspect='auto', origin='lower',
                   extent=[mesh.x.min(), mesh.x.max(), mesh.y.min(), mesh.y.max()])
        ax10[0].set_xlabel('X')
        ax10[0].set_ylabel('Y')
        ax10[0].set_title(f'VAE Mode 2')
        fig8.colorbar(h1, ax=ax10[0])

        h2 = ax10[1].imshow(U_p_2[nx//2,:,:], cmap='plasma', aspect='auto', origin='lower',
                   extent=[mesh.x.min(), mesh.x.max(), mesh.y.min(), mesh.y.max()])
        ax10[1].set_xlabel('X')
        ax10[1].set_ylabel('Y')
        ax10[1].set_title(f'POD MODE 2')
        fig10.colorbar(h2, ax=ax10[1], label='Velocity')
        plt.subplots_adjust(wspace=0.8)
        fig10.savefig('Modes_2.pdf',dpi = 8, bbox_inches="tight")
        plt.close(fig10) 

        U_p_3 = U_p_3.reshape(nx,ny,nz)
        h1 = ax11[0].imshow(modes[nx//2,:,:,2],cmap='plasma', aspect='auto', origin='lower',
                   extent=[mesh.x.min(), mesh.x.max(), mesh.y.min(), mesh.y.max()])
        ax11[0].set_xlabel('X')
        ax11[0].set_ylabel('Y')
        ax11[0].set_title(f'VAE Mode 3')
        fig11.colorbar(h1, ax=ax11[0])

        h2 = ax11[1].imshow(U_p_3[nx//2,:,:], cmap='plasma', aspect='auto', origin='lower',
                   extent=[mesh.x.min(), mesh.x.max(), mesh.y.min(), mesh.y.max()])
        ax11[1].set_xlabel('X')
        ax11[1].set_ylabel('Y')
        ax11[1].set_title(f'POD MODE 3')
        fig11.colorbar(h2, ax=ax11[1], label='Velocity')
        plt.subplots_adjust(wspace=0.8)
        fig11.savefig('Modes_3.pdf',dpi = 8, bbox_inches="tight")
        plt.close(fig11) 

        rec = rec.reshape(1,nx,ny,nz,len(t))
        h3 = ax9[0].imshow(rec[0,nx //2,:,:,100].T, cmap='plasma', aspect='auto', origin='lower',
                   extent=[mesh.x.min(), mesh.x.max(), mesh.y.min(), mesh.y.max()])
        ax9[1].set_xlabel('X')
        ax9[0].set_xlabel('X')
        ax9[0].set_ylabel('Y')
        ax9[0].set_title(f'VAE Field')
        fig9.colorbar(h3, ax=ax9[0], label='Velocity')

        u_p = u_p.reshape(nx,ny,nz,len(time))
        h4 = ax9[1].imshow(u_p[nx//2,:,:,100].T, cmap='plasma',aspect='auto', origin='lower',
                   extent=[mesh.x.min(), mesh.x.max(), mesh.y.min(), mesh.y.max()])
        ax9[1].set_xlabel('X')
        ax9[1].set_xlabel('X')
        ax9[1].set_ylabel('Y')
        ax9[1].set_title(f'POD Field')
        fig9.colorbar(h4, ax=ax9[1], label='Velocity')
        plt.subplots_adjust(wspace=0.8)
        fig9.savefig('Field_XY.pdf',dpi = 8, bbox_inches="tight")
        plt.close(fig9) 

        fig9 = None; ax9 = None
        fig9, ax9 = plt.subplots(1, 2, figsize=(10, 11))

        h3 = ax9[0].imshow(rec[0,:,:, nz//2,100], cmap='plasma', aspect='auto', origin='lower',
                   extent=[mesh.x.min(), mesh.x.max(), mesh.y.min(), mesh.y.max()])
        ax9[0].set_xlabel('X')
        ax9[0].set_ylabel('Y')
        ax9[0].set_title(f'VAE Field')
        fig9.colorbar(h3, ax=ax9[0], label='Velocity')

        u_p = u_p.reshape(nx,ny,nz,len(time))
        h4 = ax9[1].imshow(u_p[:,:,nz//2,100], cmap='plasma',aspect='auto', origin='lower',
                   extent=[mesh.x.min(), mesh.x.max(), mesh.y.min(), mesh.y.max()])
        ax9[1].set_xlabel('X')
        ax9[1].set_ylabel('Y')
        ax9[1].set_title(f'POD Field')
        fig9.colorbar(h4, ax=ax9[1], label='Velocity')
        plt.subplots_adjust(wspace=0.8)
        fig9.savefig('Field_YZ.pdf',dpi = 8, bbox_inches="tight")
        plt.close(fig9) 

        fig9 = None; ax9 = None
        fig9, ax9 = plt.subplots(1, 2, figsize=(10, 11))

        h3 = ax9[0].imshow(rec[0,:,ny//2,:,100], cmap='plasma', aspect='auto', origin='lower',
                   extent=[mesh.x.min(), mesh.x.max(), mesh.y.min(), mesh.y.max()])
        ax9[0].set_xlabel('X')
        ax9[0].set_ylabel('Y')
        ax9[0].set_title(f'VAE Field')
        fig9.colorbar(h3, ax=ax9[0], label='Velocity')

        u_p = u_p.reshape(nx,ny,nz,len(time))
        h4 = ax9[1].imshow(u_p[:,ny//2,:,100], cmap='plasma',aspect='auto', origin='lower',
                   extent=[mesh.x.min(), mesh.x.max(), mesh.y.min(), mesh.y.max()])
        ax9[1].set_xlabel('X')
        ax9[1].set_ylabel('Y')
        ax9[1].set_title(f'POD Field')
        fig9.colorbar(h4, ax=ax9[1], label='Velocity')
        plt.subplots_adjust(wspace=0.8)
        fig9.savefig('Field_ZX.pdf',dpi = 8, bbox_inches="tight")
        plt.close(fig9) 
    
        fig8 = None; ax8 = None
        for i in range(3):
            fig8, ax8 = vis_Single_PSD(
                'Latent',
                z[:, i].cpu(),
                fig=fig3,
                axs=axs3,
                color=colors[i],)
            
        ax8.set_title(f'PSD LAT SPACE',y=1.025)
        fig8.savefig(f'Res_PSD.jpg',bbox_inches='tight',dpi=300) 
        
        # --------------- SAVING ------------------------


        axs.set_title(f'Attractor 3D')
        axs.set_xlabel('X Signal')
        axs.set_ylabel('Y Signal')
        axs.set_zlabel('Z Signal')
        fig.savefig(f'Attractor_3D.jpg',bbox_inches='tight',dpi=300)

        ax1.set_title(f'Joint PDF')
        ax1.set_xlabel('X Signal')
        ax1.set_ylabel('Y Signal')
        fig1.savefig(f'Joint_PDF.jpg',bbox_inches='tight',dpi=300)

        ax4.set_title(f'Joint PDF')
        ax4.set_xlabel('X Signal')
        ax4.set_ylabel('Y Signal')
        fig4.savefig(f'Joint2_PDF.jpg',bbox_inches='tight',dpi=300)

        # axe.set_title(f'Joint PDF')
        # axe.set_xlabel('X Signal')
        # axe.set_ylabel('Y Signal')
        # axe.set_zlabel('Z Signal')
        # fige.savefig(f'Joint3D.jpg',bbox_inches='tight',dpi=300)

        axs2.set_title(f'Series l1,l2, x=Latent Space')
        axs2.set_xlabel('$t$', fontsize="large")
        axs2.set_ylabel('$l$', fontsize="large")
        fig2.savefig(f'Series_single_l1l2.jpg',bbox_inches='tight',dpi=300)

        axs3.set_title(f'Series l1,l3, x=Latent Space')
        axs3.set_xlabel('$t$', fontsize="large")
        axs3.set_ylabel('$l$', fontsize="large")
        fig3.savefig(f'Series_single_l1l3.jpg',bbox_inches='tight',dpi=300) 
        
        axs5.set_title(f'PDF l1,l2, x=Latent Space')
        axs5.set_xlabel('$l$', fontsize="large")
        axs5.set_ylabel('$P(l)$', fontsize="large")
        fig5.savefig(f'PDF_single_l1l2.jpg',bbox_inches='tight',dpi=300) 

        axs6.set_title(f'PDF l1,l3, x=Latent Space')
        axs6.set_xlabel('$l$', fontsize="large")
        axs6.set_ylabel('$P(l)$', fontsize="large")
        fig6.savefig(f'PDF_single_l1l3.jpg',bbox_inches='tight',dpi=300)  

        print('------- PLOTTED -----------')

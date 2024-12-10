import pyLOM
import datetime
import numpy as np
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import pyvista as pv
import mpi4py
mpi4py.rc.recv_mprobe = False
pv.start_xvfb()

@rank_zero_only
def create_results_folder(folder_path, echo=True):
    """Create results folder if it does not exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        if echo:
            print(f"Folder created: {folder_path}")
    else:
        if echo:
            print(f"Folder already exists: {folder_path}")


# Specify autoencoder parameters
params = {
    "pvali": 0.2,
    "batch_size": 512,
    "nepochs": 1200,
    "nlayers": 3,
    "channels": 32,
    "latent_dim": 5,
    "beta": 5e-3,
    "beta_wmup": 750,
    "beta_start": 25,
    "kernel_size": 4,
    "nlinear": 64,
    "padding": 1,
    "reduction": "mean",
    "activations": [pyLOM.NN.leakyRelu()] * 6,
    "learning_rate": 1e-4,
    "lr_decay": 0.999,
    "batch_norm": False,
    "vae": True,
}

# Load dataset and set up results directory
DATAFILE = "/mimer/NOBACKUP/groups/kthmech/sanchis/scratch/pyLowOrder/9eqmoehlis.h5"
RESULTS_DIR = f"/mimer/NOBACKUP/groups/kthmech/carlos/VAE_TRAINING/9eq_vae_beta_{params['beta']}_ld_{params['latent_dim']}_batch_{params['batch_size']}_nlinear_{params['nlinear']}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
VARIABLES = ["VELOX", "VELOY", "VELOZ"]

create_results_folder(RESULTS_DIR)
name = f"VAE_3D_9eq_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# Load the dataset
mesh = pyLOM.Mesh.load(DATAFILE)
dataset = pyLOM.Dataset.load(DATAFILE, ptable=mesh.partition_table)
time = dataset.get_variable('time')

# Extract velocity components and compute deviations
def preprocess_variable(var_name):
    var_data = dataset[var_name]
    var_mean = pyLOM.math.temporal_mean(var_data)
    return pyLOM.math.subtract_mean(var_data, var_mean)

u_x = preprocess_variable(VARIABLES[0])
u_y = preprocess_variable(VARIABLES[1])
u_z = preprocess_variable(VARIABLES[2])

print("Dataset loaded:", dataset)
pyLOM.pprint(0, "Variables: ", dataset.varnames)
pyLOM.pprint(0, "Number of points: ", len(dataset))

# Extract spatial dimensions
nx = len(np.unique(dataset.xyz[:, 0]))
ny = len(np.unique(dataset.xyz[:, 1]))
nz = len(np.unique(dataset.xyz[:, 2]))

print("Mesh dimensions - x:", nx, "y:", ny, "z:", nz)

# Prepare PyTorch Dataset
vae_dataset = pyLOM.NN.Dataset((u_x, u_y, u_z), (nx, ny, nz))
vae_dataset.crop(nx, ny, nz)
train_data, valid_data = vae_dataset.get_splits([0.8, 0.2])

print(f"Train split size: {len(train_data)}")
print(f"Validation split size: {len(valid_data)}")

train_loader = torch.utils.data.DataLoader(train_data, batch_size=params["batch_size"], shuffle=True, num_workers=1)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=params["batch_size"], num_workers=1)

# Define VAE model
betasch = pyLOM.NN.betaLinearScheduler(0., params["beta"], params["beta_start"], params["beta_wmup"])

encoder = pyLOM.NN.Encoder3D(
    nlayers          = params["nlayers"],
    latent_dim       = params["latent_dim"],
    nx               = nx,
    ny               = ny,
    nz               = nz,
    input_channels   = vae_dataset.num_channels,
    filter_channels  = params["channels"],
    kernel_size      = params["kernel_size"],
    padding          = params["padding"],
    activation_funcs = params["activations"],
    nlinear          = params["nlinear"],
    batch_norm       = params["batch_norm"],
    stride           = 2,
    dropout          = 0,
    vae              = params["vae"]
    )

decoder = pyLOM.NN.Decoder3D(
    nlayers          = params["nlayers"],
    latent_dim       = params["latent_dim"],
    nx               = nx,
    ny               = ny,
    nz               = nz,
    input_channels   = vae_dataset.num_channels,
    filter_channels  = params["channels"],
    kernel_size      = params["kernel_size"],
    padding          = params["padding"],
    activation_funcs = params["activations"],
    nlinear          = params["nlinear"],
    batch_norm       = params["batch_norm"],
    stride           = 2,
    dropout          = 0
    )

vae_model = pyLOM.NN.VariationalAutoencoder_PL(
    latent_dim     = params["latent_dim"],
    in_shape       = (nx, ny, nz),
    input_channels = vae_dataset.num_channels,
    betasch        = betasch,
    encoder        = encoder,
    decoder        = decoder,
    reduction      = params["reduction"],
    learning_rate  = params["learning_rate"],
    lr_decay       = params["lr_decay"],
    )

# Logger and Callbacks
wandb_logger = WandbLogger(
    name=name,
    project="VAE_3D_9eq"
    )

early_stop_callback = EarlyStopping(
    monitor  = "val_loss",
    patience = 300,
    mode     = "min",
    check_finite = True
    )

checkpoint_callback = ModelCheckpoint(
    monitor    = "val_loss",
    dirpath    = RESULTS_DIR,
    filename   = "VAE_{epoch:02d}_{val_loss:.2f}",
    save_top_k = 3,
    mode       = "min"
    )

lr_monitor = LearningRateMonitor(logging_interval="epoch")
# Trainer
gpus = -1 if torch.cuda.is_available() else 0

trainer = Trainer(
    logger            = wandb_logger,
    max_epochs        = params["nepochs"],
    devices           = gpus,
    accelerator       = "auto",
    strategy          = "ddp_find_unused_parameters_true",
    callbacks         = [checkpoint_callback, early_stop_callback, lr_monitor],
    gradient_clip_val = 0.5,
    precision         = "bf16-mixed" if torch.cuda.is_available() else 32
    )

trainer.fit(model=vae_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

if trainer.is_global_zero:
    print('----ALL------------------')
    rec = vae_model.reconstruct(vae_dataset)
    print(rec.shape)
    corr, detR = vae_model.correlation(vae_dataset)
    print('Correlation:', corr)
    u_x_rec = rec[0, ...]
    u_y_rec = rec[1, ...]
    u_z_rec = rec[2, ...]
    print('Predicted u_x:', u_x_rec.shape, u_y_rec.shape)
    recdtset = pyLOM.NN.Dataset((u_x_rec, u_y_rec, u_z_rec), (nx, ny, nz))
    print('RecDataset was created!', recdtset.shape)
    recdtset.pad(nx, ny, nz)
    vae_dataset.pad(nx, ny, nz)
    print('REC AF', recdtset.shape)
    dataset.add_field('urec', 1, recdtset[:, 0, :, :].numpy().reshape((len(time), nx * ny * nz)).T)
    dataset.add_field('utra', 1, recdtset[:, 1, :, :].numpy().reshape((len(time), nx * ny * nz)).T)
    dataset.add_field('unor', 1, recdtset[:, 2, :, :].numpy().reshape((len(time), nx * ny * nz)).T)
    dataset.add_field('u_x', 1, u_x.reshape((len(time), nx * ny * nz)).T)
    dataset.add_field('u_y', 1, u_y.reshape((len(time), nx * ny * nz)).T)
    dataset.add_field('u_z', 1, u_z.reshape((len(time), nx * ny * nz)).T)
    pyLOM.io.pv_writer(mesh, dataset, 'reco', basedir=RESULTS_DIR, instants=np.arange(3, dtype=np.int32), times=time, vars=['urec', 'u_x', 'VELOX', 'utra', 'u_y', 'VELOY', 'unor', 'u_z','VELOZ'], fmt='vtkh5')
    save_path = os.path.join(RESULTS_DIR, 'snapshot_0_component_0.png')
    pyLOM.NN.plotSnapshot(mesh, dataset, vars=['urec'], instant=0, component=0, cmap='jet', save_path=save_path)
    pyLOM.NN.plotSnapshot(mesh, dataset, vars=['utra'], instant=0, component=0, cmap='jet', save_path=save_path)

    pyLOM.cr_info()

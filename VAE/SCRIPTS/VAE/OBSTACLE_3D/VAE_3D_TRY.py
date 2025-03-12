import pyLOM
import datetime
import numpy as np
import os
import time
import matplotlib as plt
import torch
import h5py
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch.utils.data import Dataset, random_split, DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from src.dataset import MultiFileLazyH5Dataset

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

@rank_zero_only
def create_results_folder(RESUDIR, echo=True):
    if not os.path.exists(RESUDIR):
        os.makedirs(RESUDIR)
        if echo:
            print(f"Folder created: {RESUDIR}")
    else:
        if echo:
            print(f"Folder already exists: {RESUDIR}")

# Specify autoencoder parameters
ptrain        = 0.8
pvali         = 0.2
batch_size    = 8
nepochs       = 1500
nlayers       = 4
channels      = 64
latent_dim    = 4
beta          = 1e-2
beta_wmup     = 500
beta_start    = 50
kernel_size   = 4
nlinear       = 256
padding       = 1
reduction     = 'mean'
activations   = [pyLOM.NN.silu()] * 8
learning_rate = 1e-5
lr_decay      = 0.999
batch_norm    = True
vae           = True

# Load pyLOM dataset and set up results output

DATAFILE = ['/mimer/NOBACKUP/groups/kthmech/carlos/Datasets_3D/Obstacle_VAE/obstacle_3D_8.h5',
            '/mimer/NOBACKUP/groups/kthmech/carlos/Datasets_3D/Obstacle_VAE/obstacle_3D_9.h5']
RESUDIR = f"vae_beta_{beta}_ld_{latent_dim}_batch_{batch_size}_nlinear_{nlinear}_test_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
VARIABLE_1 = 'VELOX'
VARIABLE_2 = 'VELOY'
VARIABLE_3 = 'VELOZ'
variables = (VARIABLE_1, VARIABLE_2, VARIABLE_3)
create_results_folder(RESUDIR)
name = f"VAE_3D_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

print("Ready to create dataset")
dataset = MultiFileLazyH5Dataset(DATAFILE, batch_size=batch_size)
print("Total de muestras:", len(dataset))
print(f"\nDataset tiene {len(dataset)} muestras.\n")

num_samples = 10
times = []

for i in range(num_samples):
    start_time = time.perf_counter()
    sample = dataset[i]
    end_time = time.perf_counter() 
    load_time = end_time - start_time
    times.append(load_time)
    print(f"⏱️ Tiempo de carga muestra {i}: {load_time:.4f} segundos")

avg_time = np.mean(times)
print(f"\n🔥 Tiempo promedio de carga por muestra: {avg_time:.4f} segundos")

train_size = int(ptrain * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

nx, ny, nz = dataset.mesh_shape
print(f"Mesh size: {nx}, {ny}, {nz}")
num_channels = dataset.num_channels
print(f"Number of channels: {num_channels}")


print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

print(f"Ready to create train and validation dataloaders")

trloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    prefetch_factor=2,
    pin_memory=True,
    persistent_workers=True                       
)

print(f"Train dataloader created!")

valoader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    prefetch_factor=2,
    num_workers=1,
    pin_memory=True,
    persistent_workers=True                       
)

print(f"Validation dataloader created!")

batch_times = []
prev_time = time.time()
for batch_idx, batch in enumerate(trloader):
    cur_time = time.time()
    load_time = cur_time - prev_time
    batch_times.append(load_time)
    print(f"Batch {batch_idx + 1} loaded in {load_time:.4f} seconds.")
    prev_time = time.time()
    if batch_idx == 2:
        break

betasch = pyLOM.NN.betaLinearScheduler(0., beta, beta_start, beta_wmup)
encoder = pyLOM.NN.Encoder3D(nlayers, latent_dim, nx, ny, nz, num_channels, channels, kernel_size, padding, activations, nlinear, batch_norm, stride=2, dropout=0, vae=vae)
decoder = pyLOM.NN.Decoder3D(nlayers, latent_dim, nx, ny, nz, num_channels, channels, kernel_size, padding, activations, nlinear, batch_norm, stride=2, dropout=0)

VAE = pyLOM.NN.VariationalAutoencoder_PL(
    latent_dim     = latent_dim,
    in_shape       = (nx, ny, nz),
    input_channels = num_channels,
    betasch        = betasch,
    encoder        = encoder,
    decoder        = decoder,
    reduction      = reduction,
    learning_rate  = learning_rate,
    lr_decay       = lr_decay
)
print(VAE)

# Create WandB Logger
wandb_logger = WandbLogger(name=name, project='VAE_3D')
wandb_logger.watch(VAE)

early_stop_callback = EarlyStopping(
    monitor  = 'val_loss',
    patience = 20,
    mode     = 'min'
)

checkpoint_callback = ModelCheckpoint(
    monitor    = 'val_loss',
    dirpath    = RESUDIR,
    filename   = 'VAE_{epoch:02d}_{val_loss:.2f}',
    save_top_k = 5,
    mode       = 'min'
)

lr_monitor = LearningRateMonitor(
    logging_interval = 'epoch'
)

# Train using multiple GPUs
gpus = -1 if torch.cuda.is_available() else 0
trainer = Trainer(
    logger            = wandb_logger,
    max_epochs        = nepochs,
    devices           = gpus,
    num_nodes         = 1,
    accelerator       = 'auto',
    strategy          = 'ddp_find_unused_parameters_true',
    callbacks         = [checkpoint_callback, early_stop_callback, lr_monitor],
    gradient_clip_val = 0.5,
    precision         = 'bf16-mixed' if torch.cuda.is_available() else '32'
)

trainer.fit(model=VAE, train_dataloaders=trloader, val_dataloaders=valoader)

"""
## Reconstruct dataset and compute accuracy
rec = VAE.reconstruct(dataset)  # Devuelve (input channels, nx*ny, time)
corr, detR = VAE.correlation(dataset)
recdtset = pyLOM.NN.Dataset((rec,), (nx, ny, nz))
recdtset.pad((nx, ny, nz))
dataset.pad((nx, ny, nz))
time = dataset.get_variable('time')
dataset.add_field('urec',1,recdtset[:,0,:,:].numpy().reshape((len(time),nx*ny*nz)).T)
dataset.add_field('utra',1,recdtset[:,0,:,:].numpy().reshape((len(time),nx*ny*nz)).T)
# pyLOM.io.pv_writer(m,d,'reco',basedir=RESUDIR,instants=np.arange(time.shape[0],dtype=np.int32),times=time,vars=['urec','VELOX','utra'],fmt='vtkh5')
# pyLOM.NN.plotSnapshot(m,d,vars=['urec'],instant=0,component=0,cmap='jet')
# pyLOM.NN.plotSnapshot(m,d,vars=['utra'],instant=0,component=0,cmap='jet')

pyLOM.cr_info()

"""
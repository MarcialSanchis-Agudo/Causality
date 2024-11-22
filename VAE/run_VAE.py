import pyLOM
import datetime
import numpy as np
import os
import matplotlib as plt
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

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
ptrain      = 0.8
pvali       = 0.2
batch_size  = 128
nepochs     = 50
nlayers     = 4
channels    = 32
latent_dim  = 10
beta        = 10e-1
beta_wmup   = 1000
beta_start  = 30
kernel_size = 4
nlinear     = 256
padding     = 1
activations = [pyLOM.NN.silu(), pyLOM.NN.silu(), pyLOM.NN.silu(), pyLOM.NN.silu(), pyLOM.NN.silu(), pyLOM.NN.silu(), pyLOM.NN.silu(), pyLOM.NN.silu()]
batch_norm  = False
vae         = True

# Load pyLOM dataset and set up results output

DATAFILE = '/mimer/NOBACKUP/groups/deepmechalvis/carlos/Causality/VAE/obstacle_3D_parall.h5'
RESUDIR = f"vae_beta_{beta}_ld_{latent_dim}_batch_{batch_size}_nlinear_{nlinear}_test_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
VARIABLE = 'VELOX'

create_results_folder(RESUDIR)

name = f"VAE_3D_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# Load the dataset
# m    = pyLOM.Mesh.load(DATAFILE)
d    = pyLOM.Dataset.load(DATAFILE)
print("Dataset loaded:", d)
time = d.get_variable('time')
u_x  = d[VARIABLE]
pyLOM.pprint(0,"Variables: ", d.varnames)
pyLOM.pprint(0,"Information about the variable: ", d.info(VARIABLE))
pyLOM.pprint(0,"Number of points: ", len(d))
pyLOM.pprint(0,"Instants: ", time.shape[0])
# u_y = d['VELOY']
# u_z = d['VELOZ']
nx = len(np.unique(d.xyz[:,0]))
ny = len(np.unique(d.xyz[:,1]))
nz = len(np.unique(d.xyz[:,2]))

print("Mesh - x size:", nx)
print("Mesh - y size:", ny)
print("Mesh - z size:", nz)

dataset = pyLOM.NN.Dataset((u_x,), (nx, ny, nz))

td_train, td_valid = dataset.get_splits([0.8, 0.2])

print(f"Train split size: {len(td_train)}")
print(f"Val split size: {len(td_valid)}")

trloader = torch.utils.data.DataLoader(td_train, batch_size=batch_size, num_workers=1, shuffle=True)
valoader = torch.utils.data.DataLoader(td_valid, batch_size=batch_size, num_workers=1)

for batch_idx, batch in enumerate(trloader):
    print(f"Batch {batch_idx + 1}:")
    print("Tamaño del lote:", batch.shape)  # Verifica las dimensiones del lote
    if batch_idx == 2:  # Imprime solo los primeros tres lotes
        break

# Set the trainer for the variational autoencoder

betasch = pyLOM.NN.betaLinearScheduler(0., beta, beta_start, beta_wmup)
encoder = pyLOM.NN.Encoder3D(nlayers, latent_dim, nx, ny, nz, dataset.num_channels, channels, kernel_size, padding, activations, nlinear, batch_norm, stride=2, dropout=0, vae=vae)
decoder = pyLOM.NN.Decoder3D(nlayers, latent_dim, nx, ny, nz, dataset.num_channels, channels, kernel_size, padding, activations, nlinear, batch_norm, stride=2, dropout=0)

VAE = pyLOM.NN.VariationalAutoencoder_PL(
    latent_dim=latent_dim,
    in_shape=(nx, ny, nz),
    input_channels=dataset.num_channels,
    betasch=betasch,
    encoder=encoder,
    decoder=decoder,
    learning_rate=1e-4,
    lr_decay=0.999
)
print(VAE)

# Create WandB Logger
wandb_logger = WandbLogger(name=name, project='VAE_3D')
wandb_logger.watch(VAE)

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=50,
    mode='min'
)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=RESUDIR,
    filename='VAE_{epoch:02d}_{val_loss:.2f}',
    save_top_k=5,
    mode='min'
)

# Train using multiple GPUs
gpus = -1 if torch.cuda.is_available() else 0
trainer = Trainer(
    logger=wandb_logger,
    max_epochs=nepochs,
    devices=gpus,
    num_nodes=1,
    accelerator='auto',
    strategy='ddp',
    callbacks=[early_stop_callback, checkpoint_callback],
    precision="16-mixed" if torch.cuda.is_available() else "32"
)

trainer.fit(model=VAE, train_dataloaders=trloader, val_dataloaders=valoader)

## Reconstruct dataset and compute accuracy
rec  = VAE.reconstruct(dataset) # Returns (input channels, nx*ny, time)
recdtset = pyLOM.NN.Dataset((rec), (nx, ny, nz))
recdtset.pad((nx, ny, nz))
dataset.pad((nx, ny, nz))
d.add_field('urec',1,recdtset[:,0,:,:].numpy().reshape((len(time),nx*ny*nz)).T)
d.add_field('utra',1,recdtset[:,0,:,:].numpy().reshape((len(time),nx*ny*nz)).T)
# pyLOM.io.pv_writer(m,d,'reco',basedir=RESUDIR,instants=np.arange(time.shape[0],dtype=np.int32),times=time,vars=['urec','VELOX','utra'],fmt='vtkh5')
# pyLOM.NN.plotSnapshot(m,d,vars=['urec'],instant=0,component=0,cmap='jet')
# pyLOM.NN.plotSnapshot(m,d,vars=['utra'],instant=0,component=0,cmap='jet')

pyLOM.cr_info()
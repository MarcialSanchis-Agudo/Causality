import pyLOM
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb

# Specify autoencoder parameters
ptrain      = 0.8
pvali       = 0.2
nlayers     = 4
kernel_size = 4
padding     = 1
reduction   = 'mean'
activations = [pyLOM.NN.relu(), pyLOM.NN.relu(), pyLOM.NN.relu(), pyLOM.NN.relu(), pyLOM.NN.relu(), pyLOM.NN.relu(), pyLOM.NN.relu(), pyLOM.NN.relu()]
lr_decay    = 0.999
batch_norm  = True
vae         = True

# Load pyLOM dataset and set up results output

DATAFILE = '/mimer/NOBACKUP/groups/deepmechalvis/carlos/Causality/VAE/obstacle_3D_parall.h5'
# RESUDIR = f"vae_beta_{beta}_ld_{latent_dim}_batch_{batch_size}_nlinear_{nlinear}_test_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
VARIABLE = 'VELOX'

# Load the dataset
d    = pyLOM.Dataset.load(DATAFILE)
print("Dataset loaded:", d)
time = d.get_variable('time')
u_x  = d[VARIABLE]
pyLOM.pprint(0,"Variables: ", d.varnames)
pyLOM.pprint(0,"Information about the variable: ", d.info(VARIABLE))
pyLOM.pprint(0,"Number of points: ", len(d))
pyLOM.pprint(0,"Instants: ", time.shape[0])
nx = len(np.unique(d.xyz[:,0]))
ny = len(np.unique(d.xyz[:,1]))
nz = len(np.unique(d.xyz[:,2]))

print("Mesh - x size:", nx)
print("Mesh - y size:", ny)
print("Mesh - z size:", nz)

dataset = pyLOM.NN.Dataset((u_x,), (nx, ny, nz))

td_train, td_valid = dataset.get_splits([0.8, 0.2])

def train_sweep(config=None):
    wandb.init(dir='/mimer/NOBACKUP/groups/kthmech/carlos/VAE_TRAINING/SWEEPS/WANDB')
    config = wandb.config

    batch_size    = config.batch_size
    nepochs       = config.nepochs
    channels      = config.channels
    latent_dim    = config.latent_dim
    nlinear       = config.nlinear
    beta          = config.beta
    beta_wmup     = config.beta_wmup
    beta_start    = config.beta_start
    learning_rate = config.learning_rate

    trloader = torch.utils.data.DataLoader(td_train, batch_size=batch_size, num_workers=1, shuffle=True)
    valoader = torch.utils.data.DataLoader(td_valid, batch_size=batch_size, num_workers=1)

    # Set the trainer for the variational autoencoder

    betasch = pyLOM.NN.betaLinearScheduler(0., beta, beta_start, beta_wmup)
    encoder = pyLOM.NN.Encoder3D(nlayers, latent_dim, nx, ny, nz, dataset.num_channels, channels, kernel_size, padding, activations, nlinear, batch_norm, stride=2, dropout=0, vae=vae)
    decoder = pyLOM.NN.Decoder3D(nlayers, latent_dim, nx, ny, nz, dataset.num_channels, channels, kernel_size, padding, activations, nlinear, batch_norm, stride=2, dropout=0)

    VAE = pyLOM.NN.VariationalAutoencoder_PL(
        latent_dim     = latent_dim,
        in_shape       = (nx, ny, nz),
        input_channels = dataset.num_channels,
        betasch        = betasch,
        encoder        = encoder,
        decoder        = decoder,
        reduction      = reduction,
        learning_rate  = learning_rate,
        lr_decay       = lr_decay
        )
    print(VAE)

    # Create WandB Logger
    wandb_logger = WandbLogger(
        project   ='VAE_3D_sweep',
        log_model = True,
        save_dir  = '/mimer/NOBACKUP/groups/kthmech/carlos/VAE_TRAINING/SWEEPS/WANDB',
        name      = f"VAE_ld{config.latent_dim}_ch{config.channels}_lin{config.nlinear}_bs{config.batch_size}_ep{config.nepochs}_beta{config.beta}_bw{config.beta_wmup}_bstart{config.beta_start}"
        )
    wandb_logger.watch(VAE)

    checkpoint_callback = ModelCheckpoint(
        monitor    = 'val_loss',
        save_top_k = 3,
        mode       = 'min',
        dirpath    = '/mimer/NOBACKUP/groups/kthmech/carlos/VAE_TRAINING/SWEEPS'
    )

    early_stop_callback = EarlyStopping(
        monitor      = 'val_loss',
        patience     = 100,
        mode         = 'min',
        check_finite = True
    )


    lr_monitor = LearningRateMonitor(
        logging_interval = 'epoch'
    )

    # Train using multiple GPUs
    # gpus = -1 if torch.cuda.is_available() else 0
        # num_gpus = int(os.environ.get('SLURM_GPUS', 1))
    trainer = Trainer(
        logger            = wandb_logger,
        max_epochs        = nepochs,
        devices           = 1,
        num_nodes         = 1,
        accelerator       = 'auto',
        # strategy          = 'ddp_find_unused_parameters_true',
        callbacks         = [checkpoint_callback, lr_monitor, early_stop_callback],
        gradient_clip_val = 0.5,
        precision         = 'bf16-mixed' if torch.cuda.is_available() else '32'
        )

    trainer.fit(model=VAE, train_dataloaders=trloader, val_dataloaders=valoader)

    # Reconstruct dataset and compute accuracy
    rec  = VAE.reconstruct(dataset) # Returns (input channels, nx*ny, time)

# Create and launch WandB sweep
if __name__ == "__main__":
    train_sweep()
#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Autoencoder architecture for NN Module
#
# Last rev: 09/10/2024

import torch
import torch.nn            as nn
import torch.nn.functional as F
import numpy               as np
import pytorch_lightning   as pl

from   torch.utils.data        import DataLoader
from   torch.amp               import GradScaler, autocast
from   torch.utils.tensorboard import SummaryWriter
from   torchsummary            import summary

from   functools               import reduce
from   operator                import mul

from   ..                      import DEVICE
from   ...utils.cr             import cr


## Wrapper of a variational autoencoder
class Autoencoder(nn.Module):
    def __init__(self, latent_dim, in_shape, input_channels, encoder, decoder, device=DEVICE):
        super(Autoencoder, self).__init__()
        self.lat_dim  = latent_dim
        self.in_shape = in_shape
        self.inp_chan = input_channels
        self.N        = reduce(mul, in_shape)
        self.encoder  = encoder
        self.decoder  = decoder
        self._device  = device
        encoder.to(self._device)
        decoder.to(self._device)
        self.to(self._device)
        summary(self, input_size=(self.inp_chan, *self.in_shape),device=device)
      
    def _lossfunc(self, x, recon_x, reduction):
        return  F.mse_loss(recon_x.view(-1, self.N), x.view(-1, self.N),reduction=reduction)
    
    def forward(self, x):
        z     = self.encoder(x)
        recon = self.decoder(z)
        return recon, z  

    def fit(self, train_dataset, eval_dataset=None, epochs=100, callback=None, lr=1e-3, BASEDIR='./', reduction='mean', lr_decay=0.999, batch_size=32, shuffle=True, num_workers=0, pin_memory=True):
        dataloader_params = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        train_data = DataLoader(train_dataset, **dataloader_params)
        eval_data  = DataLoader(eval_dataset, **dataloader_params)
        # Initialization
        prev_train_loss = 1e99
        writer = SummaryWriter(BASEDIR)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
        # Training loop
        for epoch in range(epochs):
            self.train()
            num_batches = 0
            tr_loss = 0
            for batch0 in train_data:
                batch = batch0.to(self._device)
                recon, _ = self(batch)
                loss = self._lossfunc(batch, recon, reduction)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()
                num_batches += 1
            tr_loss /= num_batches
            # Validation phase
            if eval_dataset is not None:
                with torch.no_grad():
                    val_batches = 0
                    va_loss = 0
                    for val_batch0 in eval_data:
                        val_batch = val_batch0.to(self._device)
                        val_recon, _ = self(val_batch)
                        vali_loss = self._lossfunc(val_batch, val_recon, reduction)
                        va_loss += vali_loss.item()
                        val_batches += 1
                    va_loss /= val_batches
            # Logging
            writer.add_scalar("Loss/train", tr_loss, epoch + 1)
            writer.add_scalar("Loss/vali", va_loss, epoch + 1)
            # Early stopping
            if callback and callback.early_stop(va_loss, prev_train_loss, tr_loss):
                print(f'Early Stopper Activated at epoch {epoch}', flush=True)
                break
            prev_train_loss = tr_loss
            print(f'Epoch [{epoch+1} / {epochs}] average training loss: {tr_loss:.5e} | average validation loss: {va_loss:.5e}', flush=True)            
            # Learning rate scheduling
            scheduler.step()

        # Cleanup
        writer.flush()
        writer.close()
        torch.save(self.state_dict(), f'{BASEDIR}/model_state.pth')

    def reconstruct(self, dataset):
        ## Compute reconstruction and its accuracy
        num_samples = len(dataset)
        ek = np.zeros(num_samples)
        mu = np.zeros(num_samples)
        si = np.zeros(num_samples)
        rec = torch.zeros((self.inp_chan, self.N, num_samples), device=self._device)

        loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=False)

        with torch.no_grad():
            ## Energy recovered in reconstruction
            for energy_batch in loader:
                energy_batch = energy_batch.to(self._device)
                x_recon,_ = self(energy_batch)

                for i in range(num_samples):
                    x_recchan = x_recon[i]
                    rec[:, :, i] = x_recchan.view(self.inp_chan, self.N)

                    x = energy_batch[i].view(self.inp_chan * self.N)
                    xr = rec[:, :, i].view(self.inp_chan * self.N)

                    ek[i] = torch.sum((x - xr) ** 2) / torch.sum(x ** 2)
                    mu[i] = 2 * torch.mean(x) * torch.mean(xr) / (torch.mean(x) ** 2 + torch.mean(xr) ** 2)
                    si[i] = 2 * torch.std(x) * torch.std(xr) / (torch.std(x) ** 2 + torch.std(xr) ** 2)

        energy = (1 - np.mean(ek)) * 100
        print('Recovered energy %.2f' % energy)
        print('Recovered mean %.2f' % (np.mean(mu) * 100))
        print('Recovered fluct %.2f' % (np.mean(si) * 100))

        return rec.cpu().numpy()
    
    def latent_space(self, dataset):
        # Compute latent vectors
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        with torch.no_grad():
            instant  = iter(loader)
            batch    = next(instant)
            batch    = batch.to(self._device)
            _,z = self(batch)
        return z

    def decode(self, z):
        zt  = torch.tensor(z, dtype=torch.float32)
        var = self.decoder(zt)
        var = var.cpu()
        varr = np.zeros((self.N,var.shape[0]),dtype=float)
        for it in range(var.shape[0]):
            varaux = var[it,0,:,:].detach().numpy()
            varr[:,it] = varaux.reshape((self.N,), order='C')
        return varr 

## Wrapper of a variational autoencoder
class VariationalAutoencoder(Autoencoder):
    def __init__(self, latent_dim, in_shape, input_channels, encoder, decoder, device=DEVICE):
        super(VariationalAutoencoder, self).__init__(latent_dim, in_shape, input_channels, encoder, decoder, device)

    def _reparamatrizate(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        epsilon = torch.randn_like(std)  #we create a normal distribution (0 ,1 ) with the dimensions of std        
        sample = mu + std*epsilon
        return  sample
             
    def _kld(self, mu, logvar):
        mum     = torch.mean(mu, axis=0)
        logvarm = torch.mean(logvar, axis=0)
        return 0.5*torch.sum(1 + logvar - mum**2 - logvarm.exp())
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self._reparamatrizate(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z
    
    @cr('VAE.fit')   
    def fit(self, train_dataset, eval_dataset=None, betasch=None, epochs=1000, callback=None, lr=1e-4, BASEDIR='./', batch_size=32, shuffle=True, num_workers=0, pin_memory=True):
        dataloader_params = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        train_data = DataLoader(train_dataset, **dataloader_params)
        eval_data  = DataLoader(eval_dataset, **dataloader_params)
        prev_train_loss = 1e99
        writer    = SummaryWriter(BASEDIR)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0, amsgrad=False if self._device == "cpu" else True, fused=False if self._device == "cpu" else True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr*1e-3)
        scaler    = GradScaler()
        for epoch in range(epochs):
            ## Training
            self.train()
            tr_loss = 0
            mse     = 0
            kld     = 0
            beta    = betasch.getBeta(epoch) if betasch is not None else 0
            for batch0 in train_data:
                batch = batch0.to(self._device)
                optimizer.zero_grad()
                with autocast(device_type=self._device):
                    recon, mu, logvar, _ = self(batch)
                    mse_i = self._lossfunc(batch, recon, reduction='sum')
                    kld_i = self._kld(mu,logvar)
                    loss  = mse_i - beta*kld_i
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                tr_loss += loss.item()
                mse     += mse_i.item()
                kld     += kld_i.item()
            num_batches = len(train_data)
            tr_loss /= num_batches
            mse /= num_batches
            kld /= num_batches

            ## Validation
            self.eval()
            va_loss     = 0
            with torch.no_grad():
                for val_batch0 in eval_data:
                    val_batch = val_batch0.to(self._device)
                    with autocast(device_type=self._device):
                        val_recon, val_mu, val_logvar, _ = self(val_batch)
                        mse_i     = self._lossfunc(val_batch, val_recon, reduction='sum')
                        kld_i     = self._kld(val_mu,val_logvar)
                        vali_loss = mse_i - beta*kld_i
                    va_loss  += vali_loss.item()

            num_batches = len(eval_data)
            va_loss    /=num_batches
            writer.add_scalar("Loss/train",tr_loss,epoch+1)
            writer.add_scalar("Loss/vali", va_loss,epoch+1)
            writer.add_scalar("Loss/mse",  mse,    epoch+1)
            writer.add_scalar("Loss/kld",  kld,    epoch+1)

            if callback is not None:
                if callback.early_stop(va_loss, prev_train_loss, tr_loss):
                    print('Early Stopper Activated at epoch %i' %epoch, flush=True)
                    break
            prev_train_loss = tr_loss   
            print('Epoch [%d / %d] average training loss: %.5e (MSE = %.5e KLD = %.5e) | average validation loss: %.5e' % (epoch+1, epochs, tr_loss, mse, kld, va_loss), flush=True)
            # Learning rate scheduling
            scheduler.step()

        writer.flush()
        writer.close()
        torch.save(self.state_dict(), '%s/model_state' % BASEDIR)

    @cr('VAE.reconstruct')
    def reconstruct(self, dataset):
        ## Compute reconstruction and its accuracy
        num_samples = len(dataset)
        ek = np.zeros(num_samples)
        mu = np.zeros(num_samples)
        si = np.zeros(num_samples)
        rec = torch.zeros((self.inp_chan, self.N, num_samples), device=self._device)

        loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=False)

        with torch.no_grad():
            ## Energy recovered in reconstruction
            for energy_batch in loader:
                energy_batch = energy_batch.to(self._device)
                x_recon,_,_,_ = self(energy_batch)

                for i in range(num_samples):
                    x_recchan = x_recon[i]
                    rec[:, :, i] = x_recchan.view(self.inp_chan, self.N)

                    x = energy_batch[i].view(self.inp_chan * self.N)
                    xr = rec[:, :, i].view(self.inp_chan * self.N)

                    ek[i] = torch.sum((x - xr) ** 2) / torch.sum(x ** 2)
                    mu[i] = 2 * torch.mean(x) * torch.mean(xr) / (torch.mean(x) ** 2 + torch.mean(xr) ** 2)
                    si[i] = 2 * torch.std(x) * torch.std(xr) / (torch.std(x) ** 2 + torch.std(xr) ** 2)

        energy = (1 - np.mean(ek)) * 100
        print('Recovered energy %.2f' % energy)
        print('Recovered mean %.2f' % (np.mean(mu) * 100))
        print('Recovered fluct %.2f' % (np.mean(si) * 100))

        return rec.cpu().numpy()
  
    def correlation(self, dataset):
        ##  Compute correlation between latent variables
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        with torch.no_grad():
            instant  = iter(loader)
            batch    = next(instant)
            batch    = batch.to(self._device)
            _,_,_, z = self(batch)
            np.save('z.npy',z.cpu())
            corr = np.corrcoef(z.cpu(),rowvar=False)
        detR = np.linalg.det(corr)*100
        print('Orthogonality between modes %.2f' % (detR))
        return corr, detR#.reshape((self.lat_dim*self.lat_dim,))
    
    def modes(self):
        zmode = np.diag(np.ones((self.lat_dim,),dtype=float))
        zmodt = torch.tensor(zmode, dtype=torch.float32)
        zmodt = zmodt.to(self._device)
        modes = self.decoder(zmodt)
        mymod = np.zeros((self.N,self.lat_dim),dtype=float)
        modes = modes.cpu()
        for imode in range(self.lat_dim):
            modesr = modes[imode,0,:,:].detach().numpy()
            mymod[:,imode] = modesr.reshape((self.N,), order='C')
        return mymod.reshape((self.N*self.lat_dim,),order='C')

    def latent_space(self, dataset):
        # Compute latent vectors
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        with torch.no_grad():
            instant  = iter(loader)
            batch    = next(instant)
            batch    = batch.to(self._device)
            _,_,_, z = self(batch)
        return z

    def decode(self, z):
        zt  = torch.tensor(z, dtype=torch.float32)
        var = self.decoder(zt)
        var = var.cpu()
        varr = np.zeros((self.N,var.shape[0]),dtype=float)
        for it in range(var.shape[0]):
            varaux = var[it,0,:,:].detach().numpy()
            varr[:,it] = varaux.reshape((self.N,), order='C')
        return varr 
    

## Wrapper of an autoencoder using lightning
class Autoencoder_PL(pl.LightningModule):
    def __init__(self, latent_dim, in_shape, input_channels, encoder, decoder, learning_rate=1e-3, lr_decay=0.999):
        super(Autoencoder_PL, self).__init__()
        self.lat_dim  = latent_dim
        self.in_shape = in_shape
        self.inp_chan = input_channels
        self.N        = reduce(mul, in_shape)
        self.encoder  = encoder
        self.decoder  = decoder
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
      
    def _lossfunc(self, x, recon_x, reduction):
        return  F.mse_loss(recon_x.view(-1, self.N), x.view(-1, self.N),reduction=reduction)
    
    def forward(self, x):
        z     = self.encoder(x)
        recon = self.decoder(z)
        return recon, z  

    def training_step(self, batch, batch_idx):
        recon, z = self(batch)
        loss = self._lossunch(batch, recon, reduction='mean')
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        recon, z = self(batch)
        loss = self._lossfunc(batch, recon, reduction='mean')
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)
        return [optimizer], [scheduler]

    def reconstruct(self, dataset):
        ## Compute reconstruction and its accuracy
        num_samples = len(dataset)
        ek = np.zeros(num_samples)
        mu = np.zeros(num_samples)
        si = np.zeros(num_samples)
        rec = torch.zeros((self.inp_chan, self.N, num_samples), device=self.device)

        loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=False)

        with torch.no_grad():
            ## Energy recovered in reconstruction
            for energy_batch in loader:
                energy_batch = energy_batch.to(self.device)
                x_recon,_ = self(energy_batch)

                for i in range(num_samples):
                    x_recchan = x_recon[i]
                    rec[:, :, i] = x_recchan.view(self.inp_chan, self.N)

                    x = energy_batch[i].view(self.inp_chan * self.N)
                    xr = rec[:, :, i].view(self.inp_chan * self.N)

                    ek[i] = torch.sum((x - xr) ** 2) / torch.sum(x ** 2)
                    mu[i] = 2 * torch.mean(x) * torch.mean(xr) / (torch.mean(x) ** 2 + torch.mean(xr) ** 2)
                    si[i] = 2 * torch.std(x) * torch.std(xr) / (torch.std(x) ** 2 + torch.std(xr) ** 2)

        energy = (1 - np.mean(ek)) * 100
        print('Recovered energy %.2f' % energy)
        print('Recovered mean %.2f' % (np.mean(mu) * 100))
        print('Recovered fluct %.2f' % (np.mean(si) * 100))

        return rec.cpu().numpy()
    
    def latent_space(self, dataset):
        # Compute latent vectors
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        with torch.no_grad():
            instant  = iter(loader)
            batch    = next(instant)
            batch    = batch.to(self.device)
            _,z = self(batch)
        return z

    def decode(self, z):
        zt  = torch.tensor(z, dtype=torch.float32)
        var = self.decoder(zt)
        var = var.cpu()
        varr = np.zeros((self.N,var.shape[0]),dtype=float)
        for it in range(var.shape[0]):
            varaux = var[it,0,:,:].detach().numpy()
            varr[:,it] = varaux.reshape((self.N,), order='C')
        return varr 

## Wrapper of a variational autoencoder using lightning
class VariationalAutoencoder_PL(Autoencoder_PL):
    def __init__(self, latent_dim, in_shape, input_channels, betasch, encoder, decoder, learning_rate=1e-4, lr_decay=0.999):
        super(VariationalAutoencoder_PL, self).__init__(latent_dim, in_shape, input_channels, encoder, decoder, learning_rate, lr_decay)
        self.betasch = betasch # Beta scheduler instance
        self.beta = 0.0 # Initial Beta value

    def _reparamatrizate(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        epsilon = torch.randn_like(std)  #we create a normal distribution (0 ,1 ) with the dimensions of std        
        sample = mu + std*epsilon
        return  sample
             
    def _kld(self, mu, logvar):
        mum     = torch.mean(mu, axis=0)
        logvarm = torch.mean(logvar, axis=0)
        return 0.5*torch.sum(1 + logvar - mum**2 - logvarm.exp())
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self._reparamatrizate(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z
    
    def on_training_epoch_start(self):
        self.beta = self.betasch.getBeta(self.trainer.current_epoch)

    def training_step(self, batch, batch_idx):
        recon, mu, logvar, _ = self(batch)
        mse_loss = self._lossfunc(batch, recon, reduction='sum')
        kld_loss = self._kld(mu, logvar)
        loss = mse_loss - self.beta * kld_loss
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('mse_loss', mse_loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('kld_loss', kld_loss,  prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        recon, mu, logvar, _ = self(batch)
        mse_loss = self._lossfunc(batch, recon, reduction='sum')
        kld_loss = self._kld(mu, logvar)
        loss = mse_loss - self.beta * kld_loss 
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_mse', mse_loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_kld', kld_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, eta_min=self.learning_rate * 1e-3)
        return [optimizer], [scheduler]

    @cr('VAE.reconstruct')
    def reconstruct(self, dataset):
        ## Compute reconstruction and its accuracy
        num_samples = len(dataset)
        ek = np.zeros(num_samples)
        mu = np.zeros(num_samples)
        si = np.zeros(num_samples)
        rec = torch.zeros((self.inp_chan, self.N, num_samples), device=self.device)

        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            ## Energy recovered in reconstruction
            for idx, energy_batch in enumerate(loader):
                energy_batch = energy_batch.to(self.device)
                x_recon,_,_,_ = self(energy_batch)

                x_recchan = x_recon[0]
                rec[:, :, idx] = x_recchan.view(self.inp_chan, self.N)

                x = energy_batch.view(self.inp_chan * self.N)
                xr = rec[:, :, idx].view(self.inp_chan * self.N)

                ek[idx] = torch.sum((x - xr) ** 2) / torch.sum(x ** 2)
                mu[idx] = 2 * torch.mean(x) * torch.mean(xr) / (torch.mean(x) ** 2 + torch.mean(xr) ** 2)
                si[idx] = 2 * torch.std(x) * torch.std(xr) / (torch.std(x) ** 2 + torch.std(xr) ** 2)

        energy = (1 - np.mean(ek)) * 100
        print('Recovered energy %.2f' % energy)
        print('Recovered mean %.2f' % (np.mean(mu) * 100))
        print('Recovered fluct %.2f' % (np.mean(si) * 100))

        return rec.cpu().numpy()
  
    def correlation(self, dataset):
        ##  Compute correlation between latent variables
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        z_list =[]
        with torch.no_grad():
            for batch in loader:
                batch    = batch.to(self.device)
                _,_,_, z = self(batch)
                z_list.append(z.cpu())
            
        z_total = torch.cat(z_list, dim=0).numpy()

        np.save('z.npy',z_total.cpu())
        corr = np.corrcoef(z_total.cpu(),rowvar=False)
        detR = np.linalg.det(corr)*100
        print('Orthogonality between modes %.2f' % (detR))
        return corr, detR#.reshape((self.lat_dim*self.lat_dim,))
    
    def modes(self):
        zmode = np.diag(np.ones((self.lat_dim,),dtype=float))
        zmodt = torch.tensor(zmode, dtype=torch.float32)
        zmodt = zmodt.to(self.device)
        modes = self.decoder(zmodt)
        mymod = np.zeros((self.N,self.lat_dim),dtype=float)
        modes = modes.cpu()
        for imode in range(self.lat_dim):
            modesr = modes[imode,0,:,:].detach().numpy()
            mymod[:,imode] = modesr.reshape((self.N,), order='C')
        return mymod.reshape((self.N*self.lat_dim,),order='C')

    def latent_space(self, dataset):
        # Compute latent vectors
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        z_list = []

        with torch.no_grad():
            for batch in loader:
                batch    = batch.to(self.device)
                _,_,_, z = self(batch)
                z_list.append(z.cpu())
        
        z_total = torch.cat(z_list, dim=0)
        return z_total

    def decode(self, z):
        zt  = torch.tensor(z, dtype=torch.float32)
        var = self.decoder(zt)
        var = var.cpu()
        varr = np.zeros((self.N,var.shape[0]),dtype=float)
        for it in range(var.shape[0]):
            varaux = var[it,0,:,:].detach().numpy()
            varr[:,it] = varaux.reshape((self.N,), order='C')
        return varr 
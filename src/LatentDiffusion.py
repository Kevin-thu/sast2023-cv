import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import AutoencoderKL

from .DenoisingDiffusionProcess import *

# TODO begin: Inherit the AutoEncoder class from nn.Module
class AutoEncoder(object):
# TODO end
    def __init__(self,
                 model_type= "stabilityai/sd-vae-ft-ema"
                ):
        """
            A wrapper for an AutoEncoder model
            
            By default, a pretrained AutoencoderKL is used from stabilitai
            
            A custom AutoEncoder could be trained and used with the same interface.
            Yet, this model works quite well for many tasks out of the box!
        """
        
        super().__init__()
        self.model = AutoencoderKL.from_pretrained(model_type)
        
    def forward(self, input):
        return self.model(input).sample
    
    def encode(self, input, mode=False):
        dist=self.model.encode(input).latent_dist
        if mode:
            return dist.mode()
        else:
            return dist.sample()
    
    def decode(self,input):
        return self.model.decode(input).sample

class LatentDiffusion(pl.LightningModule):
    def __init__(self,
                 vae_model_type="stabilityai/sd-vae-ft-ema",
                 num_timesteps=1000,
                 latent_scale_factor=0.1,
                 batch_size=1,
                 lr=1e-4):
        """
            This is a simplified version of Latent Diffusion        
        """        
        
        super().__init__()
        self.lr = lr
        # TODO question: What's buffer?
        self.register_buffer('latent_scale_factor', torch.tensor(latent_scale_factor))
        self.batch_size=batch_size
        
        self.vae = AutoEncoder(vae_model_type)
        # TODO question: What do these two lines of code do? 
        for p in self.vae.parameters():
            p.requires_grad = False
            
        with torch.no_grad():
            self.latent_dim = self.vae.encode(torch.ones(1,3,256,256)).shape[1]
            
        # TODO begin: Complete the DenoisingDiffusionProcess p_loss function
        # Challenge: Can you figure out the forward and reverse process defined in DenoisingDiffusionProcess?
        self.model = DenoisingDiffusionProcess(generated_channels=self.latent_dim,
                                             num_timesteps=num_timesteps)
        # TODO end

    @torch.no_grad()
    def forward(self,*args,**kwargs):
        # TODO question: What's *args,**kwargs?
        return self.output_T(self.vae.decode(self.model(*args,**kwargs) / self.latent_scale_factor))
    
    def input_T(self, input):
        # TODO begin: Transform the input samples in [0, 1] range to [-1, 1]
        # Challenge: Why should we make this transform?
        return input
        # TODO end
    
    def output_T(self, input):
        # TODO begin: Transform the output samples in [-1, 1] range to [0, 1]
        return input
        # TODO end
    
    def training_step(self, batch, batch_idx):   
        
        latents = self.vae.encode(self.input_T(batch)).detach() * self.latent_scale_factor
        loss = self.model.p_loss(latents)
        
        self.log('train_loss',loss)
        
        return loss
            
    def validation_step(self, batch, batch_idx):     
        
        latents=self.vae.encode(self.input_T(batch)).detach() * self.latent_scale_factor
        loss = self.model.p_loss(latents)
        
        self.log('val_loss',loss)
        
        return loss
    
    def configure_optimizers(self):
        # TODO begin: Define the AdamW optimizer here (10 p.t.s)
        # Hint: model.parameters(), requires_grad, lr
        return # torch.optim.AdamW(...)
        # TODO end
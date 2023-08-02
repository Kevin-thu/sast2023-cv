import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm.auto import tqdm

from .forward import *
from .samplers import *
from .backbones.unet_convnext import *


class DenoisingDiffusionProcess(nn.Module):
    
    def __init__(self,
                 generated_channels=3,              
                 loss_fn=F.mse_loss,
                 schedule='linear',
                 num_timesteps=1000,
                 sampler=None
                ):
        super().__init__()
        
        # Basic Params
        self.generated_channels=generated_channels
        self.num_timesteps=num_timesteps
        self.loss_fn=loss_fn
        
        # Forward Process Used for Training
        self.forward_process=GaussianForwardProcess(num_timesteps=self.num_timesteps,
                                                    schedule=schedule)
        self.model=UnetConvNextBlock(dim=64,
                                     dim_mults=(1,2,4,8),
                                     channels=self.generated_channels,
                                     out_dim=self.generated_channels,
                                     with_time_emb=True)
               
        
        # defaults to a DDPM sampler if None is provided
        self.sampler=DDPM_Sampler(num_timesteps=self.num_timesteps) if sampler is None else sampler
        
    @torch.no_grad()
    def forward(self,
                shape=(256,256),
                batch_size=1,
                sampler=None,
                verbose=False
               ):
        """
            forward() function triggers a complete inference cycle
            
            A custom sampler can be provided as an argument!
        """           
        
        # read dimensions
        b,c,h,w = batch_size,self.generated_channels,*shape
        device = next(self.model.parameters()).device
        
        # select sampler
        if sampler is None:
            sampler=self.sampler
        else:
            sampler.to(device)

        # time steps list
        num_timesteps = sampler.num_timesteps 
        it = reversed(range(0, num_timesteps))    
        
        x_t = torch.randn([b, self.generated_channels, h, w],device=device)
                
        for i in tqdm(it, desc='diffusion sampling', total=num_timesteps) if verbose else it:
            t = torch.full((b,), i, device=device, dtype=torch.long)
            z_t=self.model(x_t,t) # prediction of noise
            x_t = sampler(x_t,t,z_t) # prediction of next state
            
        return x_t
        
    def p_loss(self,output):
        """
            Assumes output is in [-1, +1] range
        """        
        
        b,c,h,w=output.shape
        device=output.device
        
        # loss for training
        
        # input is the optional condition
        t = torch.randint(0, self.forward_process.num_timesteps, (b,), device=device).long()
        output_noisy, noise = self.forward_process(output,t,return_noise=True)        

        # reverse pass
        noise_hat = self.model(output_noisy, t) 

        # TODO begin: apply the loss_fn
        # Hint: Check the usage of F.mse_loss and review the loss function learned in class
        return self.loss_fn(noise_hat, noise)
        # TODO end
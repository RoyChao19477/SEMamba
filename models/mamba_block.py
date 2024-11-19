import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from functools import partial
from einops import rearrange

from mamba_ssm import Mamba2
from mamba_ssm.modules.block import Block
from mamba_ssm.models.mixer_seq_simple import _init_weights, create_block
from mamba_ssm.ops.triton.layer_norm import RMSNorm

class MambaBlock(nn.Module):
    def __init__(self, in_channels, cfg):
        super(MambaBlock, self).__init__()
        n_layer = 1
        self.forward_blocks  = nn.ModuleList( create_block(
            d_model=in_channels, 
            d_intermediate=0,
            ssm_cfg={
                #"layer":"Mamba2", # "Mamba1", "Mamba2"
                "layer":"Mamba1", # "Mamba1", "Mamba2"
                "d_state": cfg['model_cfg']['d_state'],
                "d_conv": cfg['model_cfg']['d_conv'],
                "expand": cfg['model_cfg']['expand'],
                },
            attn_layer_idx=[],
            attn_cfg=None,
            #attn_cfg={
            #   "num_heads": 8;
            #},
			norm_epsilon=1e-5,
			rms_norm=True,
			residual_in_fp32=False,
			fused_add_norm=False,
			layer_idx=i,
			device=None,
			dtype=None,
            ) for i in range(n_layer) )

        self.backward_blocks  = nn.ModuleList( create_block(
            d_model=in_channels, 
            d_intermediate=0,
            ssm_cfg={
                #"layer":"Mamba2", # "Mamba1", "Mamba2"
                "layer":"Mamba1", # "Mamba1", "Mamba2"
                "d_state": cfg['model_cfg']['d_state'],
                "d_conv": cfg['model_cfg']['d_conv'],
                "expand": cfg['model_cfg']['expand'],
                },
            attn_layer_idx=[],
            attn_cfg=None,
            #attn_cfg={
            #   "num_heads": 8;
            #},
			norm_epsilon=1e-5,
			rms_norm=True,
			residual_in_fp32=False,
			fused_add_norm=False,
			layer_idx=i,
			device=None,
			dtype=None,
            ) for i in range(n_layer) )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                n_residuals_per_layer=1 
            )
        )

    def forward(self, x):
        x_forward, x_backward = x.clone(), torch.flip(x, [1])
        resi_forward, resi_backward = None, None

        # Forward
        for layer in self.forward_blocks:
            x_forward, resi_forward = layer(x_forward, resi_forward)
        y_forward = (x_forward + resi_forward) if resi_forward is not None else x_forward

        # Backward
        for layer in self.backward_blocks:
            x_backward, resi_backward = layer(x_backward, resi_backward)
        y_backward = torch.flip((x_backward + resi_backward), [1]) if resi_backward is not None else torch.flip(x_backward, [1])

        return torch.cat([y_forward, y_backward], -1)

class TFMambaBlock(nn.Module):
    """
    Temporal-Frequency Mamba block for sequence modeling.
    
    Attributes:
    cfg (Config): Configuration for the block.
    time_mamba (MambaBlock): Mamba block for temporal dimension.
    freq_mamba (MambaBlock): Mamba block for frequency dimension.
    tlinear (ConvTranspose1d): ConvTranspose1d layer for temporal dimension.
    flinear (ConvTranspose1d): ConvTranspose1d layer for frequency dimension.
    """
    def __init__(self, cfg):
        super(TFMambaBlock, self).__init__()
        self.cfg = cfg
        self.hid_feature = cfg['model_cfg']['hid_feature']
        
        # Initialize Mamba blocks
        self.time_mamba = MambaBlock(in_channels=self.hid_feature, cfg=cfg)
        self.freq_mamba = MambaBlock(in_channels=self.hid_feature, cfg=cfg)
        
        # Initialize ConvTranspose1d layers
        self.tlinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)
        self.flinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)
    
    def forward(self, x):
        """
        Forward pass of the TFMamba block.
        
        Parameters:
        x (Tensor): Input tensor with shape (batch, channels, time, freq).
        
        Returns:
        Tensor: Output tensor after applying temporal and frequency Mamba blocks.
        """
        b, c, t, f = x.size()

        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.tlinear( self.time_mamba(x).permute(0,2,1) ).permute(0,2,1) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.flinear( self.freq_mamba(x).permute(0,2,1) ).permute(0,2,1) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


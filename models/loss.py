# Reference: https://github.com/yxlu-0102/MP-SENet/blob/main/models/generator.py

import torch
import torch.nn as nn
import numpy as np
from pesq import pesq
from joblib import Parallel, delayed

def phase_losses(phase_r, phase_g, cfg):
    """
    Calculate phase losses including in-phase loss, gradient delay loss, 
    and integrated absolute frequency loss between reference and generated phases.
    
    Args:
        phase_r (torch.Tensor): Reference phase tensor of shape (batch, freq, time).
        phase_g (torch.Tensor): Generated phase tensor of shape (batch, freq, time).
        h (object): Configuration object containing parameters like n_fft.
    
    Returns:
        tuple: Tuple containing in-phase loss, gradient delay loss, and integrated absolute frequency loss.
    """
    dim_freq = cfg['stft_cfg']['n_fft'] // 2 + 1  # Calculate frequency dimension
    dim_time = phase_r.size(-1)  # Calculate time dimension
    
    # Construct gradient delay matrix
    gd_matrix = (torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1) - 
                 torch.triu(torch.ones(dim_freq, dim_freq), diagonal=2) - 
                 torch.eye(dim_freq)).to(phase_g.device)
    
    # Apply gradient delay matrix to reference and generated phases
    gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)
    
    # Construct integrated absolute frequency matrix
    iaf_matrix = (torch.triu(torch.ones(dim_time, dim_time), diagonal=1) - 
                  torch.triu(torch.ones(dim_time, dim_time), diagonal=2) - 
                  torch.eye(dim_time)).to(phase_g.device)
    
    # Apply integrated absolute frequency matrix to reference and generated phases
    iaf_r = torch.matmul(phase_r, iaf_matrix)
    iaf_g = torch.matmul(phase_g, iaf_matrix)
    
    # Calculate losses
    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(gd_r - gd_g))
    iaf_loss = torch.mean(anti_wrapping_function(iaf_r - iaf_g))
    
    return ip_loss, gd_loss, iaf_loss

def anti_wrapping_function(x):
    """
    Anti-wrapping function to adjust phase values within the range of -pi to pi.
    
    Args:
        x (torch.Tensor): Input tensor representing phase differences.
    
    Returns:
        torch.Tensor: Adjusted tensor with phase values wrapped within -pi to pi.
    """
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

def compute_stft(y: torch.Tensor, n_fft: int, hop_size: int, win_size: int, center: bool, compress_factor: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the Short-Time Fourier Transform (STFT) and return magnitude, phase, and complex components.

    Args:
        y (torch.Tensor): Input signal tensor.
        n_fft (int): Number of FFT points.
        hop_size (int): Hop size for STFT.
        win_size (int): Window size for STFT.
        center (bool): Whether to pad the input on both sides.
        compress_factor (float, optional): Compression factor for magnitude. Defaults to 1.0.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Magnitude, phase, and complex components.
    """
    eps = torch.finfo(y.dtype).eps
    hann_window = torch.hann_window(win_size).to(y.device)
    
    stft_spec = torch.stft(
        y, 
        n_fft=n_fft, 
        hop_length=hop_size, 
        win_length=win_size, 
        window=hann_window, 
        center=center, 
        pad_mode='reflect', 
        normalized=False, 
        return_complex=True
    )
    
    real_part = stft_spec.real
    imag_part = stft_spec.imag

    mag = torch.sqrt( real_part.pow(2) * imag_part.pow(2) + eps )
    pha = torch.atan2( real_part + eps, imag_part + eps )

    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
    
    return mag, pha, com

def pesq_score(utts_r, utts_g, cfg):
    """
    Calculate PESQ (Perceptual Evaluation of Speech Quality) score for pairs of reference and generated utterances.
    
    Args:
        utts_r (list of torch.Tensor): List of reference utterances.
        utts_g (list of torch.Tensor): List of generated utterances.
        h (object): Configuration object containing parameters like sampling_rate.
    
    Returns:
        float: Mean PESQ score across all pairs of utterances.
    """
    def eval_pesq(clean_utt, esti_utt, sr):
        """
        Evaluate PESQ score for a single pair of clean and estimated utterances.
        
        Args:
            clean_utt (np.ndarray): Clean reference utterance.
            esti_utt (np.ndarray): Estimated generated utterance.
            sr (int): Sampling rate.
        
        Returns:
            float: PESQ score or -1 in case of an error.
        """
        try:
            pesq_score = pesq(sr, clean_utt, esti_utt)
        except Exception as e:
            # Error can happen due to silent period or other issues
            print(f"Error computing PESQ score: {e}")
            pesq_score = -1
        return pesq_score
    
    # Parallel processing of PESQ score computation
    pesq_scores = Parallel(n_jobs=30)(delayed(eval_pesq)(
        utts_r[i].squeeze().cpu().numpy(),
        utts_g[i].squeeze().cpu().numpy(),
        cfg['stft_cfg']['sampling_rate']
    ) for i in range(len(utts_r)))
    
    # Calculate mean PESQ score
    pesq_score = np.mean(pesq_scores)
    return pesq_score


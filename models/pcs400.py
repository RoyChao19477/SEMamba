import os
import torch
import torchaudio
import numpy as np
import argparse
import librosa
import scipy

# PCS400 parameters
PCS400 = np.ones(201)
PCS400[0:3] = 1
PCS400[3:5] = 1.070175439
PCS400[5:8] = 1.182456140
PCS400[8:10] = 1.287719298
PCS400[10:110] = 1.4       # Pre Set
PCS400[110:130] = 1.322807018
PCS400[130:160] = 1.238596491
PCS400[160:190] = 1.161403509
PCS400[190:202] = 1.077192982

maxv = np.iinfo(np.int16).max

def Sp_and_phase(signal):
    signal_length = signal.shape[0]
    n_fft = 400
    hop_length = 100
    y_pad = librosa.util.fix_length(signal, size=signal_length + n_fft // 2)

    F = librosa.stft(y_pad, n_fft=400, hop_length=100, win_length=400, window=scipy.signal.windows.hamming(400))
    Lp = PCS400 * np.transpose(np.log1p(np.abs(F)), (1, 0))
    phase = np.angle(F)

    NLp = np.transpose(Lp, (1, 0))

    return NLp, phase, signal_length


def SP_to_wav(mag, phase, signal_length):
    mag = np.expm1(mag)
    Rec = np.multiply(mag, np.exp(1j*phase))
    result = librosa.istft(Rec,
                           hop_length=100,
                           win_length=400,
                           window=scipy.signal.windows.hamming(400),
                           length=signal_length)
    return result

def cal_pcs(signal_wav):
    noisy_LP, Nphase, signal_length = Sp_and_phase(signal_wav.squeeze())
    enhanced_wav = SP_to_wav(noisy_LP, Nphase, signal_length)
    enhanced_wav = enhanced_wav/np.max(abs(enhanced_wav))

    return enhanced_wav

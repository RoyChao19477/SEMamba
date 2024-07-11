import os
import json
import random
import torch
import torch.utils.data
import librosa
from models.stfts import mag_phase_stft, mag_phase_istft
from models.pcs400 import cal_pcs

def list_files_in_directory(directory_path):
    files = []
    for root, dirs, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith('.wav'):   # only add .wav files
                files.append(os.path.join(root, filename))
    return files

def load_json_file(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def extract_identifier(file_path):
    return os.path.basename(file_path)

def get_clean_path_for_noisy(noisy_file_path, clean_path_dict):
    identifier = extract_identifier(noisy_file_path)
    return clean_path_dict.get(identifier, None)

class VCTKDemandDataset(torch.utils.data.Dataset):
    """
    Dataset for loading clean and noisy audio files.

    Args:
        clean_wavs_json (str): Directory containing clean audio files.
        noisy_wavs_json (str): Directory containing noisy audio files.
        audio_index_file (str): File containing audio indexes.
        sampling_rate (int, optional): Sampling rate of the audio files. Defaults to 16000.
        segment_size (int, optional): Size of the audio segments. Defaults to 32000.
        n_fft (int, optional): FFT size. Defaults to 400.
        hop_size (int, optional): Hop size. Defaults to 100.
        win_size (int, optional): Window size. Defaults to 400.
        compress_factor (float, optional): Magnitude compression factor. Defaults to 1.0.
        split (bool, optional): Whether to split the audio into segments. Defaults to True.
        n_cache_reuse (int, optional): Number of times to reuse cached audio. Defaults to 1.
        device (torch.device, optional): Target device. Defaults to None
        pcs (bool, optional): Use PCS in training period. Defaults to False
    """
    def __init__(
        self, 
        clean_json, 
        noisy_json, 
        sampling_rate=16000, 
        segment_size=32000,
        n_fft=400, 
        hop_size=100, 
        win_size=400, 
        compress_factor=1.0, 
        split=True, 
        n_cache_reuse=1, 
        shuffle=True, 
        device=None, 
        pcs=False
    ):

        self.clean_wavs_path = load_json_file( clean_json )
        self.noisy_wavs_path = load_json_file( noisy_json )
        random.seed(1234)

        if shuffle:
            random.shuffle(self.noisy_wavs_path)
        self.clean_path_dict = {extract_identifier(clean_path): clean_path for clean_path in self.clean_wavs_path}

        self.sampling_rate = sampling_rate
        self.segment_size = segment_size
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.compress_factor = compress_factor
        self.split = split
        self.n_cache_reuse = n_cache_reuse

        self.cached_clean_wav = None
        self.cached_noisy_wav = None
        self._cache_ref_count = 0
        self.device = device
        self.pcs = pcs

    def __getitem__(self, index):
        """
        Get an audio sample by index.

        Args:
            index (int): Index of the audio sample.

        Returns:
            tuple: clean audio, clean magnitude, clean phase, clean complex, noisy magnitude, noisy phase
        """
        if self._cache_ref_count == 0:
            noisy_path = self.noisy_wavs_path[index]
            clean_path = get_clean_path_for_noisy(noisy_path, self.clean_path_dict)
            noisy_audio, _ = librosa.load( noisy_path, sr=self.sampling_rate)
            clean_audio, _ = librosa.load( clean_path, sr=self.sampling_rate)
            if self.pcs == True:
                clean_audio = cal_pcs(clean_audio)
            self.cached_noisy_wav = noisy_audio
            self.cached_clean_wav = clean_audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            clean_audio = self.cached_clean_wav
            noisy_audio = self.cached_noisy_wav
            self._cache_ref_count -= 1

        clean_audio, noisy_audio = torch.FloatTensor(clean_audio), torch.FloatTensor(noisy_audio)
        norm_factor = torch.sqrt(len(noisy_audio) / torch.sum(noisy_audio ** 2.0))
        clean_audio = (clean_audio * norm_factor).unsqueeze(0)
        noisy_audio = (noisy_audio * norm_factor).unsqueeze(0)

        assert clean_audio.size(1) == noisy_audio.size(1)

        if self.split:
            if clean_audio.size(1) >= self.segment_size:
                max_audio_start = clean_audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                clean_audio = clean_audio[:, audio_start:audio_start + self.segment_size]
                noisy_audio = noisy_audio[:, audio_start:audio_start + self.segment_size]
            else:
                clean_audio = torch.nn.functional.pad(clean_audio, (0, self.segment_size - clean_audio.size(1)), 'constant')
                noisy_audio = torch.nn.functional.pad(noisy_audio, (0, self.segment_size - noisy_audio.size(1)), 'constant')

        clean_mag, clean_pha, clean_com = mag_phase_stft(clean_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor)
        noisy_mag, noisy_pha, noisy_com = mag_phase_stft(noisy_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor)

        return (clean_audio.squeeze(), clean_mag.squeeze(), clean_pha.squeeze(), clean_com.squeeze(), noisy_mag.squeeze(), noisy_pha.squeeze())

    def __len__(self):
        return len(self.noisy_wavs_path)

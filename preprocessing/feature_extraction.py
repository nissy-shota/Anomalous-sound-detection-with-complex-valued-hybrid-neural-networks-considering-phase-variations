from operator import is_
import sys
from itertools import chain
from typing import Dict, List
import glob

import librosa
import librosa.core
import librosa.feature
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig
import torch

from augmentation.crop import random_crop, make_subseq


def load_audio(audio_file, cfg, mono=False):
    """
    load audio file.
    audio_file : str
        target audio file
    mono : boolean
        When loading a multi channels file and this param is True,
        the returned data will be merged for mono data
    return : numpy.array( float )
    """
    try:
        return librosa.load(
            audio_file, sr=cfg.preprocessing.feature.sample_rate, mono=mono
        )
    except FileNotFoundError:
        print("file_broken or not exists!! : %s", audio_file)


def get_complex_spectrogram(sample_file: str, cfg: DictConfig):

    audio, _ = load_audio(sample_file, cfg, mono=False)
    complex_spectrogram = librosa.stft(
        y=audio,
        n_fft=cfg.preprocessing.feature.n_fft,
        hop_length=cfg.preprocessing.feature.hop_length,
    )

    return complex_spectrogram


class ExtractMultiChannelFeature:
    def __init__(self, is_eval: bool, cfg: DictConfig) -> None:

        self.is_eval = is_eval
        self.cfg = cfg

    def __call__(self, sample_dirs):

        query = f"{sample_dirs}/*"
        sample_files = sorted(glob.glob(query))

        multi_channels_samples = []
        for sample_file in sample_files:
            if self.cfg.preprocessing.name == "complex_spectrogram":
                # complex-valued spectrogram
                sample = get_complex_spectrogram(sample_file=sample_file, cfg=self.cfg)
                sample = torch.tensor(sample)
                multi_channels_samples.append(sample)  # complex64

        # sample shape is (#channel, #dim(fft or mel), #time)
        sample = torch.stack(multi_channels_samples)

        if self.is_eval:
            sample = make_subseq(
                sample,
                self.cfg.augumentation.n_crop_frames,
                self.cfg.preprocessing.feature.n_hop_frames,
            )
        else:
            sample = random_crop(sample, self.cfg.augumentation.n_crop_frames)
        # sample shape is (#channel, #dim(fft or mel), #time)
        return sample


class LoadMultiLabelFeatures:
    def __init__(self, is_eval: bool, cfg: DictConfig) -> None:

        self.is_eval = is_eval
        self.cfg = cfg

    def __call__(self, sample_dirs):

        features_file = f"{sample_dirs}/{self.cfg.preprocessing.file_name}"

        try:
            sample = torch.load(features_file)
        except FileNotFoundError:
            print(f"{features_file} is not found or broken.")

        if self.is_eval:
            sample = make_subseq(
                sample,
                self.cfg.augumentation.n_crop_frames,
                self.cfg.preprocessing.feature.n_hop_frames,
            )
        else:
            sample = random_crop(sample, self.cfg.augumentation.n_crop_frames)
        # sample shape is (#channel, #dim(fft or mel), #time)
        return sample

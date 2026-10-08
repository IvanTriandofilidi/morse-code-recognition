"""One frontend shared by training, evaluation, and inference."""

from dataclasses import dataclass

import librosa
import numpy as np
import torch


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 64
    mode: str = "power_db"

    def __post_init__(self):
        if self.n_mels != 64 or self.sample_rate <= 0 or self.n_fft < 2 or self.hop_length < 1:
            raise ValueError("Require 64 mel bins and positive audio parameters")
        if self.mode not in {"power_db", "notebook_legacy"}:
            raise ValueError("Unknown frontend mode")


def load_audio(path, config):
    samples, _ = librosa.load(path, sr=config.sample_rate, mono=True)
    if samples.size < config.n_fft or not np.isfinite(samples).all():
        raise ValueError(f"Audio must be finite and at least {config.n_fft} samples: {path}")
    return samples


def mel_power(samples, config):
    return librosa.feature.melspectrogram(
        y=samples,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        power=2.0,
        center=True,
    )


def features(samples, config):
    power = mel_power(samples, config)
    if config.mode == "notebook_legacy":
        # Deliberately preserve the notebook's amplitude conversion of power values.
        result = (librosa.amplitude_to_db(power, ref=np.max, top_db=80) - (-69.05112)) / 18.674374
    else:
        result = (librosa.power_to_db(power, ref=np.max, top_db=80) + 40) / 40
    return torch.from_numpy(result.astype(np.float32)).unsqueeze(0)


def mask_features(spec):
    result = spec.clone()
    for axis in (1, 2):
        width = int(torch.randint(0, max(1, int(result.shape[axis] * 0.1)) + 1, ()).item())
        start = int(torch.randint(0, result.shape[axis] - width + 1, ()).item())
        selection = [slice(None)] * 3
        selection[axis] = slice(start, start + width)
        result[tuple(selection)] = spec.mean()
    return result

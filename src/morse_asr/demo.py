"""Generate labeled synthetic illustrations; no checkpoint or private data required."""

import argparse
import csv
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from .audio import AudioConfig, mel_power

CODES = {"С": "...", "О": "---", "А": ".-", "Т": "-", "Е": ".", "М": "--"}


def synthesize(text="СОС", wpm=18, frequency=650, snr_db=None, seed=42):
    if wpm <= 0 or frequency <= 0 or frequency >= 11025:
        raise ValueError("Invalid Morse timing or carrier")
    sr, unit = 22050, 1.2 / wpm
    parts, intervals, cursor = [], [], 0.0

    def append(units, tone=False):
        nonlocal cursor
        n = round(sr * unit * units)
        x = np.zeros(n)
        if tone:
            x = 0.7 * np.sin(2 * np.pi * frequency * np.arange(n) / sr)
            ramp = min(round(0.004 * sr), n // 2)
            x[:ramp] *= np.linspace(0, 1, ramp)
            x[-ramp:] *= np.linspace(1, 0, ramp)
        parts.append(x)
        cursor += n / sr

    append(2)
    for i, letter in enumerate(text):
        start = cursor
        for j, symbol in enumerate(CODES[letter]):
            if j:
                append(1)
            append(1 if symbol == "." else 3, True)
        intervals.append((start, cursor, letter))
        if i < len(text) - 1:
            append(3)
    append(2)
    x = np.concatenate(parts)
    if snr_db is not None:
        noise = np.random.default_rng(seed).normal(size=len(x))
        noise *= np.sqrt(np.mean(x * x) / np.mean(noise * noise) / 10 ** (snr_db / 10))
        x += noise
    return x.astype(np.float32), intervals


def make_demo(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = AudioConfig()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), layout="constrained")
    fig.suptitle("Morse audio to mel features", fontsize=23, fontweight="bold", x=0.06, ha="left")
    clean, intervals = synthesize()
    noisy, _ = synthesize(snr_db=5)
    t = np.arange(len(clean)) / config.sample_rate
    axes[0].plot(t, clean, color="#245BB2", linewidth=0.65)
    axes[0].set(
        ylabel="Amplitude", title="Synthetic SOS pattern · 650 Hz · 18 WPM", xlim=(0, t[-1])
    )
    for start, end, letter in intervals:
        axes[0].axvspan(start, end, alpha=0.08, color="#245BB2")
        axes[0].text((start + end) / 2, 0.82, CODES[letter], ha="center", fontsize=14)
    axes[0].set_ylim(-1, 1.05)
    for ax, signal, title in zip(
        axes[1:], [clean, noisy], ["Clean signal", "With Gaussian noise · 5 dB SNR"]
    ):
        db = librosa.power_to_db(mel_power(signal, config), ref=np.max, top_db=80)
        img = librosa.display.specshow(
            db,
            sr=config.sample_rate,
            hop_length=config.hop_length,
            x_axis="time",
            y_axis="mel",
            ax=ax,
            cmap="magma",
            vmin=-80,
            vmax=0,
        )
        ax.set(title=title, ylabel="Frequency (Hz)", xlabel="Time (s)")
        fig.colorbar(img, ax=ax, label="Power (dB relative to peak)", pad=0.015)
    fig.savefig(output / "mel-spectrograms.png", dpi=170)
    plt.close(fig)
    # Waveforms use float encoding so noisy samples are not silently clipped.
    sf.write(output / "sos-clean.wav", clean, config.sample_rate, subtype="FLOAT")
    sf.write(output / "sos-noisy.wav", noisy, config.sample_rate, subtype="FLOAT")
    rows = []
    words = ["СОС", "ТОМ", "ТЕМА", "САМ", "МОСТ", "МЕСТО", "СОС", "АТОМ"]
    for i, word in enumerate(words):
        signal, _ = synthesize(word, wpm=12 + i, frequency=500 + i * 45, snr_db=15, seed=i)
        name = f"sample-{i:02}.wav"
        sf.write(output / name, signal, config.sample_rate, subtype="FLOAT")
        rows.append({"id": f"demo-{i:02}", "audio_path": name, "message": word})
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "audio_path", "message"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Synthetic demo and illustration saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/demo")
    make_demo(parser.parse_args().output)

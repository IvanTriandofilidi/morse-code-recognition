"""Explicit ID/path manifests; no positional pairing or dropped tail batches."""

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .audio import features, load_audio, mask_features
from .text import encode


def read_manifest(path, labeled=True):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"id", "audio_path"} | ({"message"} if labeled else set())
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest must contain {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty manifest")
    ids, paths = set(), set()
    for row in rows:
        audio = (path.parent / row["audio_path"]).resolve()
        if not row["id"] or row["id"] in ids or audio in paths:
            raise ValueError("Empty/duplicate ID or duplicate audio path")
        if not audio.is_file():
            raise FileNotFoundError(audio)
        if labeled:
            encode(row["message"])
        ids.add(row["id"])
        paths.add(audio)
        row["audio_path"] = str(audio)
    return rows


class AudioDataset(Dataset):
    def __init__(self, rows, config, augment=False):
        self.rows, self.config, self.augment = rows, config, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        spec = features(load_audio(row["audio_path"], self.config), self.config)
        if self.augment:
            spec = mask_features(spec)
        return spec, row


def collate(items):
    specs, rows = zip(*items)
    lengths = torch.tensor([s.shape[-1] for s in specs], dtype=torch.long)
    padded = torch.zeros(len(specs), 1, 64, int(lengths.max()))
    for i, spec in enumerate(specs):
        padded[i, :, :, : spec.shape[-1]] = spec
    return padded, lengths, list(rows)


def targets_for(rows, device):
    encoded = [encode(r["message"]) for r in rows]
    return (
        torch.tensor([x for seq in encoded for x in seq], device=device, dtype=torch.long),
        torch.tensor(list(map(len, encoded)), dtype=torch.long),
    )

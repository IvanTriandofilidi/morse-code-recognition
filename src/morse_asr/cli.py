"""Command-line training, evaluation, and file transcription."""

import argparse
import csv
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .audio import AudioConfig
from .data import AudioDataset, collate, read_manifest, targets_for
from .model import MorseRecognizer, ctc_loss
from .text import SYMBOLS, decode, metrics


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def loader(rows, config, batch_size, train=False):
    return DataLoader(
        AudioDataset(rows, config, augment=train),
        batch_size=batch_size,
        shuffle=train,
        collate_fn=collate,
        drop_last=False,
    )


@torch.inference_mode()
def predict(model, batches, device):
    model.eval()
    records = []
    for specs, lengths, rows in batches:
        logits, sizes = model(specs.to(device), lengths)
        for row, prediction in zip(rows, decode(logits, sizes)):
            records.append({**row, "prediction": prediction})
    return records


def score(records):
    return metrics([r["message"] for r in records], [r["prediction"] for r in records])


def fingerprint(rows):
    return {r["id"]: hashlib.sha256(Path(r["audio_path"]).read_bytes()).hexdigest() for r in rows}


def train(args):
    if args.epochs < 1 or not 0 < args.val_fraction < 1:
        raise ValueError("Require positive epochs and validation fraction between 0 and 1")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "training-config.json", vars(args))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = read_manifest(args.manifest)
    if len(rows) < 3:
        raise ValueError("At least three recordings required")
    hashes = fingerprint(rows)
    if len(set(hashes.values())) != len(rows):
        raise ValueError("Duplicate audio content; deduplicate or define grouped splits")
    order = np.random.default_rng(args.seed).permutation(len(rows))
    n_val = max(1, min(len(rows) - 1, round(len(rows) * args.val_fraction)))
    valid = [rows[i] for i in order[:n_val]]
    training = [rows[i] for i in order[n_val:]]
    config = AudioConfig(mode=args.frontend)
    write_json(
        out / "split.json",
        {
            "seed": args.seed,
            "train_ids": [r["id"] for r in training],
            "validation_ids": [r["id"] for r in valid],
            "audio_sha256": hashes,
        },
    )
    model = MorseRecognizer(hidden_size=args.hidden_size, layers=args.layers).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_batches = loader(training, config, args.batch_size, train=True)
    valid_batches = loader(valid, config, args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_batches),
        anneal_strategy="cos",
    )
    history, best = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, seen = 0.0, 0
        for specs, lengths, batch in train_batches:
            optimizer.zero_grad(set_to_none=True)
            logits, sizes = model(specs.to(args.device), lengths)
            targets, target_lengths = targets_for(batch, args.device)
            loss = ctc_loss(logits, sizes, targets, target_lengths)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite CTC loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(batch)
            seen += len(batch)
        scores = score(predict(model, valid_batches, args.device))
        history.append({"epoch": epoch, "train_ctc_loss": loss_sum / seen, **scores})
        print(json.dumps(history[-1]))
        write_json(out / "history.json", history)
        if scores["cer"] < best:
            best = scores["cer"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": model.config,
                    "audio_config": asdict(config),
                    "symbols": SYMBOLS,
                    "epoch": epoch,
                    "validation_metrics": scores,
                    "seed": args.seed,
                    "training_ids": [r["id"] for r in training],
                    "validation_ids": [r["id"] for r in valid],
                    "seen_audio_sha256": list(hashes.values()),
                },
                out / "best.pt",
            )


def infer(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["symbols"] != SYMBOLS:
        raise ValueError("Checkpoint vocabulary mismatch")
    config = AudioConfig(**checkpoint["audio_config"])
    model = MorseRecognizer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device)
    rows = read_manifest(args.manifest, labeled=args.command == "evaluate")
    if args.command == "evaluate":
        seen = set(checkpoint.get("seen_audio_sha256", []))
        overlap = sum(h in seen for h in fingerprint(rows).values())
    records = predict(model, loader(rows, config, args.batch_size), args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "message"])
        writer.writeheader()
        writer.writerows({"id": r["id"], "message": r["prediction"]} for r in records)
    if args.command == "evaluate":
        result = {**score(records), "recordings_seen_during_training_or_selection": overlap}
        write_json(output.with_suffix(".metrics.json"), result)
        print(json.dumps(result))
    print(f"Wrote {len(records)} predictions to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate", "predict"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--batch-size", type=int, default=16)
        p.add_argument("--device", default="cpu")
        if name == "train":
            p.add_argument("--epochs", type=int, default=10)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--val-fraction", type=float, default=0.1)
            p.add_argument("--lr", type=float, default=0.001)
            p.add_argument("--hidden-size", type=int, default=256)
            p.add_argument("--layers", type=int, default=5)
            p.add_argument(
                "--frontend", choices=["power_db", "notebook_legacy"], default="power_db"
            )
        else:
            p.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    train(args) if args.command == "train" else infer(args)


if __name__ == "__main__":
    main()

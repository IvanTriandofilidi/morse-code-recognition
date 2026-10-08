# Data contract

Use `id,audio_path,message` for training/evaluation and `id,audio_path` for prediction. Paths are relative to the manifest directory. Example:

```csv
id,audio_path,message
sample-001,audio/001.opus,РЎРћРЎ
sample-002,audio/002.opus,РўРћРњ
```

Join provider metadata to filenames by verified ID, never sorted row position. Preserve leading zeros. The reader rejects missing files, empty/duplicate IDs, repeated paths, empty transcripts, and unsupported characters. Training rejects byte-identical recordings under different names.

The fixed alphabet is the original 32-letter Cyrillic set (without РЃ), digits, `#`, and a literal space. No transliteration is applied. Index 44 is the CTC blank and is never a target.

The default frontend loads mono audio at 22,050 Hz; FFT 1,024, hop 256, and 64 mel bands. It computes power dB relative to the recording peak, clips the range to 80 dB, then applies `(dB + 40) / 40`. OPUS decoding depends on the installed audio backend; the demo uses WAV.

`--frontend notebook_legacy` explicitly preserves the notebook's amplitude-to-dB conversion of power values and scaling constants в€’69.05112 / 18.674374. This is not the standard power-dB conversion. Inference always restores the chosen frontend from the checkpoint.

Prepare an independent holdout before passing the training manifest to the CLI. The CLI creates a seeded 90/10 train/validation split and selects the lowest validation corpus CER. For shared transmitters, messages, or recording sessions, split groups upstream: file hashes cannot detect all related or re-encoded audio.

Augmentation is training-only. Native-length CNN processing prevents padding from changing the valid features; packed LSTM sequences ignore the padded tail. Batch normalization updates per recording in the new training path, which differs from the original batch behavior. The encoder and greedy decoder are offline, not streaming.

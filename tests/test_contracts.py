import csv

import numpy as np
import pytest
import torch

from morse_asr.audio import AudioConfig, features, mask_features
from morse_asr.data import collate, read_manifest
from morse_asr.demo import synthesize
from morse_asr.model import MorseRecognizer, ctc_loss, output_lengths
from morse_asr.text import BLANK, collapse_ctc, encode, metrics

torch.set_num_threads(2)


def test_ctc_blank_separates_repeated_letters():
    a = encode("А")[0]
    assert collapse_ctc([a, a, BLANK, a, a]) == "АА"
    assert collapse_ctc([BLANK, BLANK]) == ""


@pytest.mark.parametrize("frames", [64, 100, 255, 689, 690, 691])
def test_lengths_match_actual_convolutions(frames):
    model = MorseRecognizer(hidden_size=8, layers=1).eval()
    with torch.no_grad():
        actual = model.cnn(torch.randn(1, 1, 64, frames))
    assert actual.shape[-1] == output_lengths(torch.tensor([frames])).item()
    assert actual.shape[-2] == 10


def test_padding_does_not_change_eval_prediction():
    model = MorseRecognizer(hidden_size=8, layers=1).eval()
    a, b = torch.randn(1, 64, 160), torch.randn(1, 64, 310)
    with torch.no_grad():
        alone, sizes = model(a[None], torch.tensor([160]))
        batch, lengths, _ = collate([(a, {}), (b, {})])
        together, _ = model(batch, lengths)
    torch.testing.assert_close(alone[0], together[0, : sizes[0]], atol=1e-6, rtol=1e-5)


def test_ctc_backward_and_impossible_alignment():
    logits = torch.randn(2, 10, BLANK + 1, requires_grad=True)
    targets = torch.tensor(encode("АА") + encode("С"))
    loss = ctc_loss(logits, torch.tensor([10, 8]), targets, torch.tensor([2, 1]))
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError, match="impossible"):
        ctc_loss(logits, torch.tensor([2, 8]), targets, torch.tensor([2, 1]))


def test_cer_is_weighted_by_reference_length():
    result = metrics(["А", "АААА"], ["Б", "АААА"])
    assert result["cer"] == 0.2
    assert result["mean_utterance_cer"] == 0.5


def test_frontend_deterministic_and_masks_do_not_mutate():
    signal, _ = synthesize()
    config = AudioConfig()
    spec = features(signal, config)
    assert spec.shape[:2] == (1, 64) and torch.isfinite(spec).all()
    assert spec.min() >= -1.001 and spec.max() <= 1.001
    torch.testing.assert_close(spec, features(signal, config))
    original = spec.clone()
    mask_features(spec)
    torch.testing.assert_close(spec, original)
    assert np.isfinite(features(signal, AudioConfig(mode="notebook_legacy")).numpy()).all()


def test_manifest_rejects_duplicate_ids(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"not decoded by manifest reader")
    manifest = tmp_path / "data.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "audio_path", "message"])
        writer.writerows([["one", "clip.wav", "С"], ["one", "clip.wav", "О"]])
    with pytest.raises(ValueError, match="duplicate"):
        read_manifest(manifest)


def test_all_tail_samples_retained():
    from torch.utils.data import DataLoader

    data = [(torch.zeros(1, 64, 100 + i), {"id": str(i)}) for i in range(5)]
    batches = list(DataLoader(data, batch_size=2, collate_fn=collate))
    assert [len(b[2]) for b in batches] == [2, 2, 1]


def test_unsupported_transcript_fails_explicitly():
    with pytest.raises(ValueError, match="Unsupported"):
        encode("HELLO")

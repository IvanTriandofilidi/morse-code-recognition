"""Original CNN-BiLSTM dimensions, with exact lengths and packed recurrent input."""

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .text import BLANK


def output_lengths(lengths):
    for kernel, stride, padding in ((3, 3, 2), (3, 2, 3), (3, 3, 3), (2, 2, 0)):
        lengths = (lengths + 2 * padding - kernel) // stride + 1
    return lengths


class MorseRecognizer(nn.Module):
    def __init__(self, hidden_size=256, layers=5, dropout=0.5):
        super().__init__()
        self.config = {"hidden_size": hidden_size, "layers": layers, "dropout": dropout}
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=(2, 3), padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=(1, 2), padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 64, 3, stride=(2, 3), padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )
        self.projection = nn.Sequential(nn.Linear(640, 256), nn.ReLU())
        self.lstm = nn.LSTM(
            256,
            hidden_size,
            layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 512), nn.ReLU(), nn.Linear(512, BLANK + 1)
        )

    def forward(self, specs, lengths):
        if specs.ndim != 4 or specs.shape[1:3] != (1, 64):
            raise ValueError("Expected [batch, 1, 64, frames]")
        if len(lengths) != len(specs) or (lengths < 1).any() or (lengths > specs.shape[-1]).any():
            raise ValueError("Invalid feature lengths")
        # Convolve each unpadded example so right-padding cannot change its features.
        sequences = []
        for sample, length in zip(specs, lengths):
            x = self.cnn(sample[None, :, :, : int(length)])
            sequences.append(x[0].permute(2, 0, 1).flatten(1))
        x = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        sizes = output_lengths(lengths).cpu()
        packed = pack_padded_sequence(
            self.projection(x), sizes, batch_first=True, enforce_sorted=False
        )
        packed, _ = self.lstm(packed)
        x, _ = pad_packed_sequence(packed, batch_first=True)
        return self.head(x), sizes


def ctc_loss(logits, lengths, targets, target_lengths):
    offset = 0
    for available, count in zip(lengths, target_lengths):
        target = targets[offset : offset + int(count)]
        required = int(count) + int((target[1:] == target[:-1]).sum())
        if required > int(available):
            raise ValueError("CTC alignment impossible: target + repeats exceed output frames")
        offset += int(count)
    if offset != targets.numel():
        raise ValueError("Target lengths do not match concatenated targets")
    return nn.functional.ctc_loss(
        logits.log_softmax(-1).transpose(0, 1),
        targets,
        lengths,
        target_lengths,
        blank=BLANK,
        zero_infinity=False,
    )

"""Rebuild the figure from recorded notebook output, without extrapolating epochs."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
data = json.loads((root / "reports/historical-training.json").read_text())
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, axes = plt.subplots(1, 2, figsize=(14, 4.7), layout="constrained")
fig.suptitle("Recorded training history", fontsize=21, fontweight="bold")
for ax, run in zip(axes, data["segments"]):
    epochs = run["epochs"]
    for key, label, color in [("train_loss", "Training", "#245BB2"),
                               ("validation_loss", "Validation", "#D06D27")]:
        ax.plot([r["epoch"] for r in epochs], [r[key] for r in epochs],
                label=label, color=color, marker="o", markersize=4, linewidth=2)
    ax.set(title=run["label"], xlabel="Epoch within saved segment", ylabel="Recorded CTC loss")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.15)
    ax.legend(frameon=False)
fig.savefig(root / "docs/assets/training-history.png", dpi=170)

"""Export a read-only snapshot from the live R3 TensorBoard scalar API."""
import json
import os
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import urlopen

os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
import numpy as np

BASE = "http://127.0.0.1:6007"
TAGS = [
    ("Train/mean_reward", "Mean reward"),
    ("Train/mean_episode_length", "Episode length"),
    ("Episode_Reward/touch_goal", "Touch-goal reward"),
    ("Episode_Termination/time_out", "Episode timeouts"),
]
OUTPUT = os.environ.get("R3_TB_SNAPSHOT", "artifacts/monitoring/r3_tensorboard_snapshot.png")


def series(tag):
    query = urlencode({"run": ".", "tag": tag})
    with urlopen(f"{BASE}/data/plugin/scalars/scalars?{query}", timeout=15) as response:
        rows = json.load(response)
    if not rows:
        raise RuntimeError(f"No scalar samples for {tag}")
    steps = np.array([row[1] for row in rows], dtype=float)
    values = np.array([row[2] for row in rows], dtype=float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"Non-finite scalar values for {tag}")
    return steps, values


fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
max_step = 0
for ax, (tag, label) in zip(axes.flat, TAGS):
    steps, values = series(tag)
    max_step = max(max_step, int(steps[-1]))
    ax.plot(steps, values, color="#4a90e2", alpha=0.42, linewidth=0.8, label="raw")
    window = min(31, len(values))
    kernel = np.ones(window) / window
    smooth = np.convolve(values, kernel, mode="valid")
    ax.plot(steps[window - 1 :], smooth, color="#ea6a37", linewidth=2, label=f"{window}-step mean")
    ax.set_title(label)
    ax.set_xlabel("Training iteration")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
fig.suptitle(f"DreamControl Kick live TensorBoard snapshot | step {max_step} | {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, dpi=150)
print(f"OUTPUT={OUTPUT} STEP={max_step} BYTES={os.path.getsize(OUTPUT)}")
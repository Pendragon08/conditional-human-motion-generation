"""Read-only, low-overhead live view of R3 TensorBoard scalars."""
import io
import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from urllib.parse import urlencode
from urllib.request import urlopen

os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
import numpy as np

SOURCE = "http://127.0.0.1:6007"
TAGS = [
    ("Train/mean_reward", "Mean reward"),
    ("Train/mean_episode_length", "Episode length"),
    ("Episode_Reward/touch_goal", "Touch-goal reward"),
    ("Episode_Termination/time_out", "Episode timeouts"),
]
CACHE_LOCK = Lock()
CACHE = {"step": None, "image": None}


def get_series(tag):
    query = urlencode({"run": ".", "tag": tag})
    with urlopen(f"{SOURCE}/data/plugin/scalars/scalars?{query}", timeout=10) as response:
        rows = json.load(response)
    if not rows:
        raise RuntimeError(f"No scalar samples for {tag}")
    steps = np.array([x[1] for x in rows], dtype=float)
    values = np.array([x[2] for x in rows], dtype=float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"Non-finite scalar values for {tag}")
    return steps, values


def render():
    reward_steps, reward_values = get_series(TAGS[0][0])
    max_step = int(reward_steps[-1])
    with CACHE_LOCK:
        if CACHE["step"] == max_step and CACHE["image"] is not None:
            return CACHE["image"], max_step
        data = [(reward_steps, reward_values)] + [get_series(tag) for tag, _ in TAGS[1:]]
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        for ax, ((steps, values), (_, title)) in zip(axes.flat, zip(data, TAGS)):
            ax.plot(steps, values, color="#4a90e2", alpha=0.35, linewidth=0.8)
            width = min(31, len(values))
            smooth = np.convolve(values, np.ones(width) / width, mode="valid")
            ax.plot(steps[width - 1 :], smooth, color="#e56a32", linewidth=2)
            ax.set_title(title)
            ax.set_xlabel("Iteration")
            ax.grid(alpha=0.25)
        fig.suptitle(f"DreamControl Kick | step {max_step} | {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")
        output = io.BytesIO()
        fig.savefig(output, format="png", dpi=135)
        plt.close(fig)
        CACHE["step"] = max_step
        CACHE["image"] = output.getvalue()
        return CACHE["image"], max_step


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            if self.path.startswith("/plot.png"):
                body, _ = render()
                content_type = "image/png"
            elif self.path.startswith("/health"):
                steps, values = get_series("Train/mean_reward")
                body = json.dumps({"ok": True, "last_step": int(steps[-1]), "last_reward": float(values[-1])}).encode()
                content_type = "application/json"
            elif self.path in ("/", "/index.html"):
                steps, values = get_series("Train/mean_reward")
                step = int(steps[-1])
                reward = float(values[-1])
                now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                body = ("<!doctype html><html><head><meta charset='utf-8'>"
                        "<meta http-equiv='refresh' content='10'>"
                        "<title>R3 Live Training Curves</title>"
                        "<style>body{font-family:system-ui;margin:24px;background:#101723;color:#eee}"
                        "img{max-width:100%;background:white;border-radius:8px}a{color:#80bdff}"
                        "strong{color:#96e0c5}</style></head><body>"
                        "<h2>DreamControl Kick — live training curves</h2>"
                        f"<p>Latest iteration: <strong>{step}/2000</strong> &nbsp; "
                        f"Latest mean reward: <strong>{reward:.3f}</strong> &nbsp; Checked: {now}</p>"
                        "<p>Refreshes every 10 seconds. Chart redraws only when a new iteration is logged. "
                        "Blue: raw; orange: 31-iteration moving average.</p>"
                        f"<img src='/plot.png?step={step}' alt='Training curves'>"
                        "<p><a href='http://127.0.0.1:6007/#scalars'>Open TensorBoard Scalars</a></p>"
                        "</body></html>").encode()
                content_type = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            self.send_error(503, str(exc))


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 6008), Handler).serve_forever()
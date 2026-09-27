"""Best-effort, non-blocking CLoSD training telemetry.

This module never opens a viewer or waits for a client. When the optional UDP
endpoint is unset, the upstream training path does no telemetry work.
"""

from __future__ import annotations

import json
import os
import socket
import time


_endpoint = os.environ.get("R3_CLOSD_TELEMETRY_UDP", "")
_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if _endpoint else None
if _sock is not None:
    _sock.setblocking(False)


def _send(payload: dict) -> None:
    if _sock is None:
        return
    try:
        host, port = _endpoint.rsplit(":", 1)
        payload["ts"] = time.time()
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        _sock.sendto(data, (host, int(port)))
    except (BlockingIOError, OSError, ValueError, TypeError):
        # A broken viewer must never break the simulation or PPO update.
        pass


def emit_plan(frame: int, prompt: str, motion) -> None:
    if _sock is None:
        return
    try:
        # Keep UDP datagrams below the OS limit; the next 10 poses are enough
        # to display the online planning horizon without affecting the model.
        points = motion[:10].detach().to("cpu").float().numpy().round(4).tolist()
        _send({"type": "plan", "frame": int(frame), "prompt": str(prompt), "joints": points})
    except Exception:
        pass


def emit_step(frame: int, prompt: str, actual, predicted) -> None:
    if _sock is None or frame % 5:
        return
    try:
        actual_xyz = actual.detach().to("cpu").float().numpy().round(4).tolist()
        predicted_xyz = predicted.detach().to("cpu").float().numpy().round(4).tolist()
        _send({"type": "step", "frame": int(frame), "prompt": str(prompt),
               "actual": actual_xyz, "predicted": predicted_xyz})
    except Exception:
        pass

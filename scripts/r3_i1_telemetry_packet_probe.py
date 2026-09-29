"""Read-only UDP packet probe for the separate training-state viewer pathway."""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=40123)
    parser.add_argument("--min-packets", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", args.port))
    sock.settimeout(5)
    deadline = time.monotonic() + args.timeout
    packets = []
    try:
        while len(packets) < args.min_packets and time.monotonic() < deadline:
            try:
                data, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue
            packet = json.loads(data)
            if packet.get("source") != "actual_training_simulation_sample":
                raise ValueError("Packet source is not actual simulator telemetry")
            if len(packet["robot_root_pose_wxyz"]) != 7:
                raise ValueError("Invalid root pose")
            if len(packet["joint_names"]) != len(packet["joint_position_rad"]):
                raise ValueError("Invalid joint names/positions")
            packets.append(packet)
    finally:
        sock.close()
    result = {
        "status": "COMPLETE" if len(packets) >= args.min_packets else "FAILED",
        "mode": "inference_only_telemetry_transport_probe_not_training",
        "packet_count": len(packets),
        "first_step": packets[0]["env_step"] if packets else None,
        "last_step": packets[-1]["env_step"] if packets else None,
        "joint_count": len(packets[0]["joint_names"]) if packets else None,
        "source": packets[0]["source"] if packets else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

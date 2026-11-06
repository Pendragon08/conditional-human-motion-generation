"""Verify the full checkpoint payload on CPU before admission or resume."""
from collections import Counter
import io
import json
from pathlib import Path
import struct

import torch

from runtime import require, verify_hash


SCHEMA = "repair_prior29_exact_checkpoint_v1"


def compare_exact(expected, actual, path="payload"):
    """Compare nested state including tensor dtype, shape, stride and raw bytes."""
    if type(expected) is not type(actual):
        raise ValueError(f"{path}: type changed from {type(expected).__name__} to {type(actual).__name__}")
    if isinstance(expected, torch.Tensor):
        require(actual.device.type == "cpu", f"{path}: load did not map tensor to CPU")
        require((expected.dtype, expected.shape, expected.layout, expected.stride())
                == (actual.dtype, actual.shape, actual.layout, actual.stride()), f"{path}: tensor metadata changed")
        before = expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        after = actual.detach().contiguous().reshape(-1).view(torch.uint8)
        require(torch.equal(before, after), f"{path}: tensor bytes changed")
        return 1
    if isinstance(expected, dict):
        require(expected.keys() == actual.keys(), f"{path}: dictionary keys changed")
        return sum(compare_exact(item, actual[key], f"{path}[{key!r}]") for key, item in expected.items())
    if isinstance(expected, (tuple, list)):
        require(len(expected) == len(actual), f"{path}: sequence length changed")
        return sum(compare_exact(item, actual[index], f"{path}[{index}]") for index, item in enumerate(expected))
    if isinstance(expected, float):
        require(struct.pack("!d", expected) == struct.pack("!d", actual), f"{path}: floating-point bytes changed")
    else:
        require(expected == actual, f"{path}: value changed")
    return 0


def adam_steps(payload):
    state = payload["optimizer"]["state"]
    require(bool(state), "checkpoint has no trained Adam state")
    histogram = Counter()
    for slot, entry in state.items():
        step = entry["step"]
        count = float(step.item() if isinstance(step, torch.Tensor) else step)
        require(count.is_integer() and count >= 1, f"Adam state {slot} has an invalid step counter")
        require(all(isinstance(entry.get(key), torch.Tensor) for key in ("exp_avg", "exp_avg_sq")),
                f"Adam state {slot} is missing moment tensors")
        histogram[int(count)] += 1
    cursor = payload["cursor"]["step"]
    require(max(histogram) == cursor, "Adam maximum step counter disagrees with optimizer update cursor")
    return {"cursor_updates": cursor, "initialized_parameter_states": len(state),
            "min_parameter_updates": min(histogram), "max_parameter_updates": max(histogram),
            "parameter_update_histogram": {str(step): histogram[step] for step in sorted(histogram)}}


def verified_readback(expected, path_or_buffer):
    actual = torch.load(path_or_buffer, map_location="cpu", weights_only=True)
    required = {"version", "model", "optimizer", "cursor", "config", "provenance", "rng"}
    require(isinstance(actual, dict) and required <= expected.keys() and expected.keys() == actual.keys(),
            "checkpoint payload keys changed or complete schema missing")
    counts = {key: compare_exact(expected[key], actual[key], f"payload[{key!r}]") for key in expected}
    require(all(counts[key] > 0 for key in ("model", "optimizer", "rng")),
            "checkpoint lacks complete model/Adam/RNG tensors")
    return {"schema": SCHEMA, "all_fields_exact": True, "compared_tensor_counts": counts,
            "adam_steps": adam_steps(actual)}


def buffer_witness(payload, byte_budget):
    with io.BytesIO() as buffer:
        torch.save(payload, buffer)
        size = buffer.tell()
        require(size < byte_budget, "actual checkpoint exceeds budget")
        buffer.seek(0)
        return size, verified_readback(payload, buffer)


def wrap_checkpoint(save, write_json):
    def checkpoint(path, data, cfg):
        path = Path(path)
        report = save(path, data, cfg)
        try:
            evidence = verified_readback(data, path)
            verify_hash(path, report["sha256"])
            validation_path = path.with_suffix(".validation.json")
            write_json(validation_path, {**evidence, "checkpoint_sha256": report["sha256"],
                                         "checkpoint_bytes": report["bytes"]}, exclusive=True)
        except Exception as error:
            write_json(path.with_suffix(".validation_failed.json"),
                       {"schema": SCHEMA, "checkpoint_sha256": report["sha256"], "all_fields_exact": False,
                        "type": type(error).__name__, "error": str(error)}, exclusive=True)
            raise
        return {**report, "validation_path": str(validation_path), "adam_steps": evidence["adam_steps"]}
    return checkpoint


def require_receipt(checkpoint):
    checkpoint = Path(checkpoint)
    require(not checkpoint.with_suffix(".validation_failed.json").exists(),
            "checkpoint has a recorded validation failure")
    proof = json.loads(checkpoint.with_suffix(".validation.json").read_text(encoding="utf8"))
    require(proof["schema"] == SCHEMA and proof["all_fields_exact"] is True
            and proof["checkpoint_bytes"] == checkpoint.stat().st_size,
            "checkpoint lacks an exact full-payload roundtrip receipt")
    counts = proof["compared_tensor_counts"]
    require(all(counts[key] > 0 for key in ("model", "optimizer", "rng")),
            "checkpoint receipt lacks complete model/Adam/RNG evidence")
    digest = verify_hash(checkpoint, proof["checkpoint_sha256"])
    return digest, proof


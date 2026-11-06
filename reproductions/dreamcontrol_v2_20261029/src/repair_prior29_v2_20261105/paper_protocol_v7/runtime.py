"""Bind the repaired entry points to the archived infrastructure and data."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from contracts import COORDINATE_SCHEMA, GUIDANCE_POLICY


SOURCE = Path(__file__).resolve().parent
REPAIR = "prior29_floor_guidance_taskmask_v2_20261105"
BASELINE_HASHES = {
    "stage2_robot_prior.py": "0116ed351101490df7c4b7beaa249eeb21e937a4334cccbb492a2688eb90896d",
    "stage2_cmdm29.py": "1a709eb943588423a368bc014afdff2db747891145991c1528a9582ed3ce6594",
}
DIFFUSION_HASH = "f9218ba5e49a1aac4d12e15615f6be8eb3ed371ab57f48a52bb329a7a0c68a89"
URDF_HASH = "572b54991d8dccd6d3e091570788fb307bd0877b6a062426583f0a17fb56cc95"
FK_SOURCE_HASH = "b7ef8d99a6b42120edb58cbd04641c2b935c91b4c7982b61636db4e41e22a77d"
TASK_SOURCE_HASH = "aa871aaa8a818c3146290835cef8eaa0a8d89fbc5730d80b160b2df6b8535ae0"
PICK_CORE_HASH = "b6e16d820b399bf70022c9266bca67879f60cf4086f696aa78873ead758bf0a4"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_hash(path, expected):
    digest = sha(path)
    require(digest == expected, f"source changed: {path} has {digest}, expected {expected}")
    return digest


def load_config(path):
    overrides = json.loads(Path(path).read_text(encoding="utf8"))
    base_config = Path(overrides["baseline_config"]).resolve(strict=True)
    base = json.loads(base_config.read_text(encoding="utf8"))
    cfg = {**base, **overrides, "baseline_config": str(base_config)}
    root = Path(cfg["project_root"]).resolve(strict=True)
    base_stage = Path(base["stage_dir"]).resolve()
    stage = Path(cfg["stage_dir"]).resolve()
    require(stage.is_relative_to(root) and stage != base_stage and not stage.is_relative_to(base_stage),
            "use a separate repaired stage directory under project_root")
    require(cfg["run_name"] not in ("", ".", "..", base["run_name"])
            and Path(cfg["run_name"]).name == cfg["run_name"],
            "use a distinct single-component run name")
    require(cfg["max_steps"] == 10000, "first repaired evaluation is bound to 10000 optimizer updates")
    sample_cap = cfg.get("max_sample_batch_size", 4)
    require(type(sample_cap) is int and 1 <= sample_cap <= 32, "invalid max_sample_batch_size")
    cfg.update(stage_dir=str(stage), repair_schema=REPAIR, coordinate_schema=COORDINATE_SCHEMA,
               training_guidance=GUIDANCE_POLICY, sampling=GUIDANCE_POLICY, max_sample_batch_size=sample_cap)
    return cfg


def load_legacy(cfg):
    root = Path(cfg["project_root"])
    legacy_dir = Path(cfg.get("baseline_src", root / "stage2_prior29_20261103/src")).resolve(strict=True)
    files = {name: verify_hash(legacy_dir / name, digest) for name, digest in BASELINE_HASHES.items()}
    trajgen = Path(cfg["trajgen"]).resolve(strict=True)
    files["diffusion/gaussian_diffusion.py"] = verify_hash(
        trajgen / "diffusion/gaussian_diffusion.py", DIFFUSION_HASH
    )
    files["model/cmdm.py"] = verify_hash(trajgen / "model/cmdm.py", cfg["cmdm_source_sha256"])
    sys.path.insert(0, str(legacy_dir))
    spec = importlib.util.spec_from_file_location("_bound_repair_legacy_stage2", legacy_dir / "stage2_robot_prior.py")
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    return legacy, files


def inputs(cfg, legacy, dependencies):
    data, stats, prior = legacy.preflight(cfg)
    sources = {name: sha(SOURCE / name)
               for name in ("contracts.py", "guidance.py", "runtime.py", "train.py", "validation.py")}
    sources.update(dependencies)
    return data, stats, {**prior, "repair_schema": REPAIR, "repair_training_sources": sources,
                         "guidance_policy": GUIDANCE_POLICY, "coordinate_schema": COORDINATE_SCHEMA}


def load_task_dependencies(cfg):
    root = Path(cfg["project_root"])
    pick = Path(cfg.get("pick_src", root / "stage3_pick29_20261103/src"))
    task = Path(cfg.get("task_src", root / "stage3_supported_references_20261104/src"))
    fk_src = Path(cfg.get("fk_src", root / "training_disk_guard_20261101"))
    trajgen = Path(cfg["trajgen"])
    urdf = trajgen.parent / "Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf"
    hashes = {
        "stage3_pick29_core.py": verify_hash(pick / "stage3_pick29_core.py", PICK_CORE_HASH),
        "stage3_supported_task_conditions.py": verify_hash(task / "stage3_supported_task_conditions.py", TASK_SOURCE_HASH),
        "g1_29dof.urdf": verify_hash(urdf, URDF_HASH),
        "g1_fk.py": verify_hash(fk_src / "g1_fk.py", FK_SOURCE_HASH),
        "conditions.py": sha(SOURCE / "conditions.py"),
        "sample.py": sha(SOURCE / "sample.py"),
    }
    sys.path[:0] = [str(pick), str(task), str(fk_src)]
    return urdf, hashes


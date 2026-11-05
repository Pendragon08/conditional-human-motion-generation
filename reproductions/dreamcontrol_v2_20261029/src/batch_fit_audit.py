"""Resumable single-CPU exploratory audit of an OmniRetarget sample.

The sample is deterministic but not representative of the full distribution.
Only per-clip diagnostic JSON is written; no trainable dataset or model is
created. Run in a dedicated directory; resume only with the identical config.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import re
import tempfile
import zipfile
import numpy as np
import torch

from g1_fk import URDFFK
from motion_bridge import SOURCE_NAMES
from omniretarget_bridge import convert_layout
from resample_omniretarget import resample
from waist_lock_ik import FIT_LINKS
from waist_lock_root_ik import fit_root_and_waist


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def select(archives: dict[str, Path], seed: int, *, counts=(20, 20, 10), exclude=frozenset()) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    out = []
    if len(counts) != 3 or any(not isinstance(n, int) or n < 0 for n in counts) or not any(counts):
        raise ValueError('sample counts must be three nonnegative integers with a positive total')
    for name, limit in zip(('robot-object.zip', 'robot-terrain.zip', 'robot-object-terrain.zip'), counts):
        with zipfile.ZipFile(archives[name]) as z:
            members = sorted(n for n in z.namelist() if n.endswith('.npz'))
        if name == 'robot-object.zip':
            pool = [n for n in members if re.search(r'_original\.npz$', n)]
        elif name == 'robot-terrain.zip':
            pool = [n for n in members if re.search(r'_z_scale_1\.0\.npz$', n)]
        else:
            groups = {}
            # Prefer originals when available; scenes without one may use the
            # unscaled chair version. A scene remains one candidate group.
            for member in sorted(members, key=lambda x: (not x.endswith('_original.npz'), x)):
                base = Path(member).name
                take = re.match(r'(Take .*?)_Skeleton.*_chair_2_stage\.npz$', base)
                scene = re.match(r'(scene_\d+)_(?:original|chair_scaled_1\.0)\.npz$', base)
                if take or scene:
                    groups.setdefault((take or scene).group(1), member)
            pool = list(groups.values())
        pool = [member for member in pool if (name, member) not in exclude]
        if len(pool) < limit:
            raise ValueError(f'{name}: expected at least {limit} groups, found {len(pool)}')
        out.extend((name, member) for member in rng.sample(pool, limit))
    return out


def save_json(path: Path, obj: dict) -> None:
    fd, temp = tempfile.mkstemp(prefix='.audit-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    except BaseException:
        if os.path.exists(temp): os.unlink(temp)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--urdf29', type=Path, required=True)
    parser.add_argument('--urdf27', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--object-count', type=int, default=20)
    parser.add_argument('--terrain-count', type=int, default=20)
    parser.add_argument('--object-terrain-count', type=int, default=10)
    parser.add_argument('--exclude-manifest', type=Path, help='exclude archive/member pairs from an earlier audit')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    archives = {x: args.project_root/'data/OmniRetarget_Dataset'/x for x in
                ('robot-object.zip', 'robot-terrain.zip', 'robot-object-terrain.zip')}
    counts = (args.object_count, args.terrain_count, args.object_terrain_count)
    excluded = set()
    if args.exclude_manifest:
        source_manifest = json.loads(args.exclude_manifest.read_text(encoding='utf-8'))
        excluded = {(item['archive'], item['member']) for item in source_manifest['samples']}
    config = {'seed': args.seed, 'steps': 120, 'learning_rate': .015, 'sample_counts': list(counts),
              'excluded_manifest_sha256': sha(args.exclude_manifest) if args.exclude_manifest else None,
              'methods': {'unweighted': [0., 0.], 'foot_2_contact_10': [2., 10.]},
              'urdf29_sha256': sha(args.urdf29), 'urdf27_sha256': sha(args.urdf27),
              'source_archives_sha256': {k: sha(v) for k, v in archives.items()},
              'script_sha256': sha(Path(__file__)),
              'dependency_sha256': {n: sha(Path(__file__).with_name(n)) for n in
                                    ('g1_fk.py', 'motion_bridge.py', 'omniretarget_bridge.py',
                                     'resample_omniretarget.py', 'waist_lock_ik.py', 'waist_lock_root_ik.py')},
              'python_version': platform.python_version(), 'torch_version': str(torch.__version__),
              'numpy_version': np.__version__,
              'warning': 'Experimental sample QA; not physics, source-independence or paper validation.'}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_file = (args.output_dir/'.lock').open('a+')
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('another audit is already running in this directory')
    manifest_path = args.output_dir/'manifest.json'
    chosen = select(archives, args.seed, counts=counts, exclude=excluded)
    manifest = {'config': config, 'samples': [{'archive': a, 'member': m} for a, m in chosen]}
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
            raise SystemExit('existing run requires --resume and an identical manifest')
    else:
        if args.resume:
            raise SystemExit('cannot resume without a manifest')
        save_json(manifest_path, manifest)
    fk29, fk27 = URDFFK(args.urdf29), URDFFK(args.urdf27)
    for idx, (archive_name, member) in enumerate(chosen, 1):
        key = hashlib.sha256((archive_name+'\0'+member).encode('utf-8')).hexdigest()
        output = args.output_dir/(key+'.json')
        if output.exists():
            print(f'{idx}/{len(chosen)} already recorded', flush=True)
            continue
        record = {'archive': archive_name, 'member': member}
        try:
            with zipfile.ZipFile(archives[archive_name]) as z:
                payload = z.read(member)
            record['source_npz_sha256'] = hashlib.sha256(payload).hexdigest()
            with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                sample = resample(data['qpos'], float(data['fps']))
            motion, _ = convert_layout(sample)
            truth = fk29.forward(motion[:, :3], motion[:, 3:7], motion[:, 7:], SOURCE_NAMES, links=FIT_LINKS)
            source_velocity = np.diff(truth[:, 4:6], axis=0)*20
            contact = np.linalg.norm(source_velocity, axis=-1) < .03
            results = {}
            for label, (foot, contact_weight) in config['methods'].items():
                fit = fit_root_and_waist(motion, args.urdf29, args.urdf27, steps=120,
                                         foot_position_weight=foot, contact_velocity_weight=contact_weight)
                pred = fk27.forward(fit['root_pos'], fit['root_quat_xyzw'], fit['joint_pos'], fit['joint_names'], links=FIT_LINKS)
                err = np.linalg.norm(pred-truth, axis=-1)
                vel_error = np.linalg.norm(np.diff(pred[:, 4:6], axis=0)*20-source_velocity, axis=-1)
                if abs(float(err.mean()) - fit['final_mean_keypoint_error_m']) > 1e-5:
                    raise ValueError('independent world-space FK disagreement')
                results[label] = {'mean_keypoint_error_m': float(err.mean()), 'max_keypoint_error_m': float(err.max()),
                                  'mean_foot_error_m': float(err[:, 4:6].mean()), 'max_foot_error_m': float(err[:, 4:6].max()),
                                  'mean_wrist_error_m': float(err[:, 10:12].mean()),
                                  'contact_proxy_velocity_error_mps': float(vel_error[contact].mean()) if contact.any() else None,
                                  'max_root_translation_m': fit['max_root_translation_m'], 'max_root_rotation_rad': fit['max_root_rotation_rad']}
            record.update(status='ok', frames=len(motion), contact_proxy_count=int(contact.sum()), methods=results)
        except Exception as exc:
            record.update(status='error', error_type=type(exc).__name__, error=str(exc))
        save_json(output, record)
        print(f"{idx}/{len(chosen)} {record['status']} {archive_name} {member}", flush=True)
    save_json(args.output_dir/'COMPLETE.json', {'record_count': len(chosen), 'status': 'finished',
                                               'note': 'Check individual records for errors and limitations.'})


if __name__ == '__main__':
    main()

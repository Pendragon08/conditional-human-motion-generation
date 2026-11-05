"""Characterize high-frequency energy before 30-to-20 Hz interpolation.

This diagnostic does not filter signals or establish physical fidelity. It
examines root translations and raw joint coordinates, not orientations.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import zipfile

import numpy as np


def high_frequency_fraction(signal: np.ndarray, fps: float, cutoff: float = 10.0) -> float | None:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2 or len(signal) < 4 or not np.isfinite(signal).all():
        raise ValueError('expected finite (frames, channels), at least four frames')
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError('sampling rate and cutoff must be positive and finite')
    centered = signal - signal.mean(axis=0, keepdims=True)
    spectrum = np.fft.rfft(centered * np.hanning(len(signal))[:, None], axis=0)
    power = np.abs(spectrum) ** 2
    total = float(power.sum())
    if total < 1e-12:
        return None
    frequencies = np.fft.rfftfreq(len(signal), 1.0 / fps)
    return float(power[frequencies > cutoff].sum() / total)


def audit_archive(path: Path, cutoff: float) -> dict:
    fractions, errors = [], []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename.endswith('.npz'):
                continue
            try:
                with np.load(io.BytesIO(archive.read(member)), allow_pickle=False) as data:
                    qpos = np.asarray(data['qpos'])
                    fps = float(np.asarray(data['fps']).item())
                if qpos.ndim != 2 or qpos.shape[1] not in (36, 43):
                    raise ValueError(f'invalid shape {qpos.shape}')
                # Coordinates per published format; quaternion channels excluded.
                signal = qpos[:, 4:36]
                fraction = high_frequency_fraction(signal, fps, cutoff)
                if fraction is not None:
                    fractions.append(fraction)
            except (KeyError, ValueError, OSError, zipfile.BadZipFile) as exc:
                errors.append({'member': member.filename, 'error': str(exc)})
    arr = np.array(fractions)
    return {'archive': path.name, 'measured_motions': len(fractions), 'errors': errors,
            'fraction_median': float(np.median(arr)) if arr.size else None,
            'fraction_p95': float(np.quantile(arr, .95)) if arr.size else None,
            'fraction_max': float(arr.max()) if arr.size else None,
            'above_1_percent_count': int(np.sum(arr > .01))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cutoff', type=float, default=10.0)
    args = parser.parse_args()
    result = {'cutoff_hz': args.cutoff, 'results': [audit_archive(p, args.cutoff) for p in args.archives],
              'notes': ['Hann-windowed demeaned root translation and joint coordinates, aggregated per motion.',
                        '1% is an exploratory reporting threshold, not a paper or quality criterion.',
                        'Quaternion orientation and physical plausibility are not assessed.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print(json.dumps(result, ensure_ascii=False))
    return int(any(x['errors'] for x in result['results']))


if __name__ == '__main__':
    raise SystemExit(main())

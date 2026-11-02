"""TextOp paper tracking fidelity metrics on matched 50 Hz link positions.

The caller must authenticate the observations and reference, preserve all
trials, and distinguish an incomplete trajectory from an observed success.
"""


def measure(executed, reference, expected_frames):
    import numpy as np
    p = np.asarray(executed, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if (p.ndim != 3 or p.shape != r.shape or p.shape[1:] != (30, 3)
            or len(p) < 2 or type(expected_frames) is not int or len(p) > expected_frames
            or not np.isfinite(p).all() or not np.isfinite(r).all()):
        raise ValueError('expected finite aligned T x 30 x 3 link positions')
    delta = p-r
    relative = (p-p[:, :1, :])-(r-r[:, :1, :])
    relative_error = np.linalg.norm(relative, axis=-1).mean(axis=1)
    velocity_error = np.diff(p, axis=0)-np.diff(r, axis=0)
    acceleration_error = np.diff(p, n=2, axis=0)-np.diff(r, n=2, axis=0)
    complete = len(p) == expected_frames
    return dict(frames=len(p), expected_frames=expected_frames, complete=complete,
                success=bool(complete and float(relative_error.max()) <= 0.3),
                max_root_relative_mean_link_error_m=float(relative_error.max()),
                global_mpjpe_mm=float(np.linalg.norm(delta, axis=-1).mean()*1000),
                root_relative_mpjpe_mm=float(relative_error.mean()*1000),
                velocity_error_mm_per_frame=float(np.linalg.norm(velocity_error, axis=-1).mean()*1000),
                acceleration_error_mm_per_frame2=(float(np.linalg.norm(acceleration_error, axis=-1).mean()*1000) if len(p)>=3 else None),
                note='mean per-link Euclidean norm; incomplete trajectories are not counted as success')

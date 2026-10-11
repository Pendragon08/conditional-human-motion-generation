# V21 COM velocity measurement correction

Independent V19 isolated-body test identifies exported linear velocity as COM velocity. Material-point velocity therefore uses `v_COM + omega cross (rotated_point - rotated_COM)`. Original implementation used a body-origin lever without subtracting COM. V21 is a measurement repair, not a novel learning mechanism, and changes no thresholds, policy, geometry or physics.

Already executed once offline:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/contact_com_metric_v21_20261012.py
```

Calibrations: rotated material-point velocity matches V19 numerical differentiation within3mm/s, while the legacy offset-body formula errs above5cm/s. Preserved V3 known stationary/sliding/airborne boxes remain correctly classified. Recomputed V17 same12 humanoid inputs still have0 slip-proxy passes and0 joint passes under original thresholds. Historical rows retained. This suggests the speed-convention defect does not explain all quality failures. Full validation and articulated controls are separate steps; do not pool legacy and corrected scores or infer generator efficacy from calibration. No human raw-data upload or Overleaf writes.

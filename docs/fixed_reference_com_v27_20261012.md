# V27: fixed-reference replay measurement calibration

This completes the original plan's fixed-reference replay check after COM velocity repair. It is not a new algorithm or candidate efficacy test. Prerequisite: V26 full validation receipt PASS and V24 432 attempted validation inputs completed. No changes to goals, thresholds, physical environment, tracker, generator or retargeting. No Overleaf writes.

Use all 24 preselected validation sources and three offsets (72 inputs), seed 11. Replay the frozen generated reference, then the same input's frozen actual physical trajectory as a refined reference. No fresh diffusion sampling is performed in replay. Keep all cases and mark combined success false if construction terminated; padded terminated sequences cannot become successful by replay. Selection contains one candidate only, so no quality-based filtering occurs.

## Fresh document witness: run exact invocation once

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_fixed_reference_com_v27_20261012.py --check
```

Expect exit 0 and `FIXED_REFERENCE_COM_DOCUMENT_CHECK_PASS`. Inspect V27 first raw replay receipt (72 cases), original-goal hash equals V24 planned seed11, reference clock 0..119, upstream canonical float32 root-centering replay verification, initial seven state differences zero, native contact file SHA and COM geometry. Independently verify source/document/config bindings and the selection has exactly one original construction candidate per case. Keep any failure, do not retry or edit in witness role. Save independent report under `results/fixed_reference_com_v27_20261012/DOCUMENT_WITNESS.md`.

After a positive document witness, root may execute:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_fixed_reference_com_v27_20261012.py --run
```

Expected `FIXED_REFERENCE_COM_CALIBRATION_COMPLETE` and two immutable replay receipts. Old failure/output folders remain. V24 batch helper enforces the remaining 4 GPU-hour measurement stage and 72 GPU-hour global cap, 20 GiB disk and 15 GiB cumulative output guards. Both replay stages must be included in refreshed budget snapshot (`fixed_reference_com` prefix). The result is a calibration/quality diagnosis on validation sources and one seed; it is not the 432-input formal test or evidence of novel improvement. Physical-quality floor means the condition-feedback route remains stopped pending diagnosis, not permission to relax criteria.

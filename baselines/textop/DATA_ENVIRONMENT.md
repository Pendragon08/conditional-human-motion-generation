# Paper-oriented dataset preparation environment

Environment spec: configs/data-prep-env-spec.json, canonical hash87f577cc. Warm reuse of gmr_amass_20261021: Python3.10.21, Torch2.5.1+cu121, NumPy1.26.4, SMPL-X0.1.28, MuJoCo3.3.7, Mink0.0.13. No pip changes. Run only CPU with four threads. Existing GMR source mirror is pinned by file hashes, not an available Git commit. Body models are the licensed AMASS-compatible locked-head SMPL-X NPZ files; 16 shape coefficients are retained explicitly.

Execute this exact command from Windows PowerShell for independent validation:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 -i C:/Users/Pende/.ssh/id_ed25519 ouyanghao@10.147.17.32 "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' /data/code/ouyanghao/oyh/phygile-reproduction-20261017/envs/gmr_amass_20261021/bin/python /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/prepare_paper_data.py --spec /data/code/ouyanghao/oyh/textop-reproduction-20261024/configs/data-prep-env-spec.json --pairs /data/code/ouyanghao/oyh/textop-reproduction-20261024/data_preparation_20261025/available_raw_pairs.private.json --output /data/code/ouyanghao/oyh/textop-reproduction-20261024/data_preparation_20261025/witness_document --witness --limit 1"
```

Expected exit0 and sentinel TEXTOP_GMR_BODY_WITNESS_PASSED. The final JSON records CPU-only operation, eight SMPL-X frames forwarded, num_betas16, one finite GMR IK output with qpos shape[36] (root7 + joints29), package versions and source/model hashes. Optional XRobot streaming-not-installed output does not affect offline IK. This writes a numerical witness JSON only; it does not launch a batch, train an RL policy, certify dynamic trackability, or complete dataset preparation.

Normal preprocessing separately reconstructs matched raw motions, resamples to50Hz, applies contact thresholds0.002 squared frame displacement and0.2m height, checks a69D encoder/inverse and geometric mirroring, and aligns the available BABEL annotations. Every prepared sequence remains physical_filter_passed=false until the paper's privileged-policy filtering is actually evaluated. The paused23DoF/57D generator is retained as a released-version experiment and is not a69D checkpoint.

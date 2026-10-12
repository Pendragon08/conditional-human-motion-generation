# V46 native CLoSD calibration after the reference velocity repair

Prerequisite: V43 and V45 independent witnesses PASS. Execute two batches:
native CLoSD generation using actual execution prefix, and independent fixed
reference replay of its exact recorded positions. Same 24 validation sources,
three offsets, seed11, frozen original targets/text/length/time and paired noise
as V43 planned control. The V43 velocity interface is shared. Generator/tracker
weights and thresholds remain unchanged. Retain every failure and all72 cases.

This checks an existing baseline under the corrected interface; it is not an
innovation or formal C1 rerun. Compare native fixed replay with V43 planned fixed
replay; the native construction count alone is not reference data utility.
Strict execution-time lineage, no retries, measurement4 GPUh/total72 GPUh,
output15 GiB and data free20 GiB apply via the inherited launcher. No G1 or
downstream training is launched by this diagnostic. No Overleaf edits.

Execute exactly once:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 env PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_native_velocity_v46_20261012.py --run
```

Expected COMPLETE_NATIVE_VELOCITY_DIAGNOSTIC. Fresh document-following witness
must check invocation once, terminal process, all source hashes, request/receipt
binding, same72 ordered cases and targets, paired generation noise, fixed replay
canonical position identity, native float32 velocity rule at all frames,
observation and trajectory agreement scope. If execution differs despite equal
reference, report it. No efficacy or causality inference from this calibration.

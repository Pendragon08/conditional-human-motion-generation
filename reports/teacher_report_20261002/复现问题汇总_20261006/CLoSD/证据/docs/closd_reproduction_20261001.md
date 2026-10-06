# CLoSD reproduction recovery, 2026-10-01

The previous 512-environment fine-tune reached and saved epoch66000, then exited139 after `MAX EPOCHS NUM!`. It did not stop early. A native shutdown failure is suspected, not proved; the original log has no Python/native backtrace. Performance degradation is a separate issue. The loaded optimizer LR remains2e-5, matching the official config; wrong LR has not been established. 512x32 instead of3072x32 rollout frames is an established difference, not a proved sole cause.

Server root /data/code/ouyanghao/oyh/conditional-human-motion-generation. New source third_party/CLoSD is a clean shared clone pinned to d8950bdb78312fcd2844f336b3fc079e7368c87f. Reuse read-only Python3.8/Torch1.13/IsaacGym Preview4 and overlay from /data/code/ouyanghao/oyh/humanoid-combat-motion; reuse previously licensed assets in its HF cache. All new outputs go to the new generic project. No licensed data or weights enter GitHub. No package rebuild.

Input/kernel witness, run verbatim:

```powershell
ssh -b 10.147.17.250 -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 'cd /data/code/ouyanghao/oyh/conditional-human-motion-generation; bash scripts/r3_closd_reproduction_20261001.sh --check'
```

Expect exit0, `CLOSD_R3_GPU_WITNESS (8, 8) (1, 4) NVIDIA RTX A6000`, `CLOSD_CONFIG_PROBE_PASS 3072 True 2.5`, then `CLOSD_REPRODUCTION_INPUT_PASS clean_source=true official_envs=3072`. This validates composition of the real wrapper's configuration, not full capacity or paper validation. Hydra configuration is composed from the absolute official configuration directory, then the undecorated official main executes it; this avoids the imported-decorator package search failure. The failed initial wrapper attempt is archived as hydra_failed_v1; it never trained.

Queue r3_closd_reproduction_20261001: (1)3072-environment three-update diagnostic from epoch62000 to max_epochs62002 (official stopping condition is `>`; discard diagnostic weights), (2)4096-environment official released fine-tuned and untuned checkpoint evaluations, (3)if actual local scene gate exists, official3072-env additional4000-epoch fine-tune from original untuned checkpoint, fixed DiP CFG2.5, (4)4096-env,500-frame,CFG7.5 final evaluation. No shrinking on OOM; preserve precise stage log and stop.

`r3_closd_isolated_entry.py` imports official run.py. Runtime adaptations: serial motion loading (temporary flags.debug only within motion loading), read-only nonblocking UDP scene snapshots, capture generated XML path, native lifecycle logging, explicit Gym simulator destruction after the official runner returns. It does not change model/loss/reward/PPO minibatch/rollout/task sampling. It enables faulthandler; return, cleanup begin and cleanup completion are distinct sentinels. Nonzero exits remain failures even if a model was saved. No forced os._exit to mask crashes.

Compute physical GPU0; independent DreamControl GPU1 training/rendering. Monitor process `r3_closd_scene_receiver.py` receives actual first-environment rigid body transforms over loopback UDP18446 and loads its generated MJCF and target URDF into the already accepted independent MeshCat scene server. This is sampled actual simulation state, not a checkpoint replay or a native IsaacGym viewer. MeshCat renderer failure never blocks PPO. Formal training requires artifacts/monitoring/r3_closd_3072_local_view_gate.json. Prior user confirmation is retained separately; do not fabricate a fresh confirmation.

Only declare completion after normal process exit, target checkpoint/hash, resolved saved configuration, each task's>=1000 resets and final success table. The official evaluator intentionally calls exit() after its final table: the wrapper accepts only that zero-code evaluation exit, then performs the same explicit cleanup; other exits remain failures. Strike counters are pooled upstream, not independent kick/punch estimates.

If the original queue finishes both official evaluations and stops awaiting the live-view gate, after explicit confirmation record user_confirmed_local_stream=true and source=actual_isaacgym_simulation in the gate JSON. Resume only the two remaining stages with `bash scripts/r3_closd_resume_formal3072_20261001.sh` in a new tmux session. It verifies all three completed predecessor states, the gate and the same exclusive lock; it refuses preexisting formal logs. Do not restart the original queue and repeat the completed capacity/evaluations. The continuation has not been launched while the gate is pending.

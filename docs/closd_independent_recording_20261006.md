# Independent recording of actual CLoSD states

Three bounded native-camera witnesses rendered30 frames but all exited139 at interpreter shutdown. Explicit camera destruction and one-camera-per-env did not fix it. These are failed attempts; they do not prove the cause of the old512 training failure. No full native-camera job will run. Preserve their logs and partial media.

Use the already proven graphics-disabled physics stack instead. `r3_closd_record_poses.py` runs pinned official CLoSD/DiP/player inference, records actual rigid-body and object transforms plus state/prompt/done/reset, and stops at a bounded frame count with explicit simulator destruction. It never creates a graphics context. Twelve envs, seed42, CFG7.5,500-frame episodes; the first fixed env of each initialized task is recorded. SIT may transition to GET_UP. No PPO update, success selection or altered task/physics. Exported poses are visualization inputs, not a second simulation or sim-to-sim result.

Warm provider ledger `/data/code/ouyanghao/oyh/humanoid-combat-motion/.aris/compute/closd-r3.md`, unchanged spec `c419897a2fbf6b5104dac52b2ef4a7f24b7cbea31f78721f1591d1609bcb81be`. No packages changed on the compute server. All new outputs belong to the generic project, licensed assets/checkpoints stay server-side. Scene geometry can be read for visualization but is not copied to the GitHub report bundle.

Fresh agent-follows-doc witness, once verbatim:

```powershell
ssh -b 10.147.17.250 -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 'cd /data/code/ouyanghao/oyh/conditional-human-motion-generation; bash scripts/r3_closd_record_frozen_20261006.sh pose_witness pose_doc_v1'
```

Expected actual exit0, GPU kernel witness, `CLOSD_POSE_BOUND_REACHED 60`, cleanup complete, manifest `status=COMPLETE`, `normal_exit=true` and `exit_verified_by=parent_shell_after_process_exit`, unchanged released fine-tuned model SHA `7ead3778b95516d8b91b7c5939674e45dd0ad10fcc22da60f9f57d204695f6ac`,60 actual state frames covering four initialized tasks. Output `artifacts/closd_recordings_20261006/pose_doc_v1/CLoSD_multitask_finetune`; state/log `artifacts/monitoring/r3_closd_record_pose_doc_v1*`. Do not repair or retry this tag.

After that pass, one serial job invokes `bash scripts/r3_closd_record_frozen_20261006.sh poses released_pose_pair_v1`,900 frames per frozen released checkpoint. Independent MeshCat browser rendering produces supplementary task demonstration videos, clearly labeled as actual Isaac Gym states rendered independently. These are not native-camera MP4s or recordings of the already completed4096-env evaluations. Numeric scores remain unchanged and their `evaluation_video_verified=false` remains truthful. Formal3072/4000 training remains separately gated on current local actual-scene confirmation.

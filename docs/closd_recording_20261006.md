# CLoSD released-policy video completion, 2026-10-06

This run supplements missing qualitative videos. It does not rerun the completed 4096-environment evaluations and does not train. Two released frozen checkpoints are used, with official source pinned to `d8950bdb78312fcd2844f336b3fc079e7368c87f`. Twelve environments, seed42, CFG7.5 and 500-frame episodes select the first initialized environment of each task. SIT's official FSM may transition to GET_UP. The videos are not independent estimates of success, and the original quantitative evaluations still have `evaluation_video_verified=false`.

Warm environment: unchanged provider spec `c419897a2fbf6b5104dac52b2ef4a7f24b7cbea31f78721f1591d1609bcb81be`, ledger `/data/code/ouyanghao/oyh/humanoid-combat-motion/.aris/compute/closd-r3.md`. Reuse its Python3.8/Torch1.13/IsaacGym Preview4, dependency overlay and licensed cache without installing packages. Outputs are under the new project only. Released weights and licensed datasets stay on the server.

The existing isolated entry preserves official serial asset loading and read-only UDP telemetry. The recording entry enables the Isaac Gym graphics device/cameras, adds four RGB camera encoders and a read-only state trace, and bounds inference at the requested frame count. Camera changes do not change simulation physics, model, rewards or actions. A private `RecordingFinished` sentinel stops the demonstration; explicit simulator destruction and actual exit0 are required. It is not a formal evaluator return. Video FPS is derived from the real simulation timestep times control decimation, not assumed60. FFprobe checks frame count and positive duration; checkpoint hashes must remain unchanged.

Fresh agent kernel-and-camera witness: execute exactly once, verbatim:

```powershell
ssh -b 10.147.17.250 -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 'cd /data/code/ouyanghao/oyh/conditional-human-motion-generation; bash scripts/r3_closd_record_frozen_20261006.sh witness doc_witness_v3'
```

Expected exit0, `CLOSD_R3_GPU_WITNESS`, `CLOSD_RECORDING_BOUND_REACHED 30`, `CLOSD_NATIVE_CLEANUP_COMPLETE`, `CLOSD_RECORDING_COMPLETE`, four nonblank rendered videos of30 frames, and unchanged released checkpoint SHA `7ead3778b95516d8b91b7c5939674e45dd0ad10fcc22da60f9f57d204695f6ac`. Logs and state are `artifacts/monitoring/r3_closd_record_doc_witness_v2*`; media/trace/config/manifest are `artifacts/closd_recordings_20261006/doc_witness_v2/CLoSD_multitask_finetune/`. A failure must be preserved; do not repeat this tag or repair the environment silently.

The v1 witness rendered and probed all30 frames, then segfaulted139 at interpreter shutdown after its internal cleanup/completion sentinels. It is a failed process, not a pass. Its original manifest's `normal_exit=true` was premature and is superseded by the actual failed state. V2 explicitly destroys official and supplementary camera sensors before destroying the simulator. It leaves `normal_exit=null` until the parent shell verifies the actual zero process exit. This is one bounded lifecycle repair, not a proven attribution of the previous512 training failure.

After passing the witnessed camera invocation, a single new tmux job runs `bash scripts/r3_closd_record_frozen_20261006.sh full released_pair_v1` for900 frames per checkpoint, serially on GPU0. The common CLoSD lock prevents concurrent formal training. Encoding and inference are supplementary demonstrations, not PPO. Actual scene UDP18446 continues independently to the accepted MeshCat monitor; monitor failure cannot stop the compute wrapper.

Formal3072/4000 training remains pending the current actual-scene local confirmation gate described in `closd_reproduction_20261001.md`. Video completion must not fabricate that gate. Record current frames before requesting the user's visual confirmation.

V2 also exited139 after recording/cleanup; it did not pass. V3 avoids creating the official unused default sensors alongside each named sensor: exactly one camera per recorded environment. A [NVIDIA forum report](https://forums.developer.nvidia.com/t/gym-render-all-camera-sensors-sim-always-causes-segmentation-fault-simple-script-to-reproduce/258924) describes headless multiple-camera failures, motivating this bounded topology check, but does not prove the cause on this server. No further camera retries beyond this v3 witness; a failed v3 will require a separate-renderer route. Internal manifests remain only file-completion evidence until parent-shell exit verification.

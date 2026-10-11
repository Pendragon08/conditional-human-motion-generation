# V11 independent document witness

Verdict: **PASS_RECORDING_ONLY**.

The fresh witness read only the run-experiment skill, its compute environment contract, the specified closd-baseline ledger, and docs/contact_cpu_export_v11_20261012.md before executing the following documented invocation verbatim exactly once:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_cpu_export_v11_20261012.py --check
```

Exit code was 0. Both CONTACT_SAMPLING_BATCH_PASS repeat_a and CONTACT_SAMPLING_DOCUMENT_CHECK_PASS appeared. The process was observed while still running and then polled to completion; the command was not retried. This witness made no source/environment edits, installations, --run invocation, or additional simulations. Subsequent SSH calls only inspected saved files and source. Root ran repeat_b separately; the attached audit snapshot was captured after that separate root run and therefore contains its ledger/state, but this witness's array inspection and conclusion concern repeat_a only.

Direct inspection of the server-local recordings:

- Twelve first attempts, four validation sources by three offsets, seed 11. All 1,440 frame/case samples valid, all twelve valid lengths exactly 120, all termination flags false.
- Actual/reference/velocity/contact arrays are float32 [120,12,24,3], quaternion [120,12,24,4]. Reference clock equals 0..119 exactly. All trajectory, initial-state, sensor, root, DOF and action arrays finite.
- Sensor archive contains sensor [120,12,4,6], root [120,12,13], and DOF position, velocity and action [120,12,69]. Passive foot sensors remain secondary evidence, not ground contact truth; sensor_metrics_used_for_selection=false.
- Seven bootstrap versus first-action state pairs were independently compared from initial_states.npz: root, DOF position/velocity, body position/quaternion/linear/angular velocity. All bit-identical, maximum difference zero. Each archived state also matches its metadata SHA256. The four initial_rigid_cache_consistency entries remain metadata-only observations; no independent cache-array evidence was added.
- All six acquired bridge buffers report native CPU and policy CUDA:0. The actual proxy source wraps CPU-native descriptors, retains descriptors and mirrors, copies at existing acquire/refresh calls, and converts setter inputs to owned contiguous CPU buffers. The wrapper source sets use_gpu_pipeline=false while retaining PhysX GPU and the CUDA model device.
- Log says GPU PhysX, physics device cuda:0, GPU Pipeline disabled, RL device cuda:0, 574 observations and 69 actions. Metadata says policy observation mode 1, size 574; physics dt 0.01666666753590107, two substeps, contact_collection=2. Existing fetch_results(True) fence recorded 242 calls. No extra simulation calls were introduced by the inspected bridge.

Frozen bindings and integrity:

- Canonical environment specification SHA256 is 81bdf764e312bcc92e4969e5eb9d2d6d6043f14c037e5a588dabbd564871f780, matching the ledger. Warm reuse; no rebuild.
- Upstream HEAD d8950bdb78312fcd2844f336b3fc079e7368c87f; no tracked upstream changes. All eighteen frozen code/config/doc hashes match versions.json. All four external motion and five external semantic input hashes match.
- Generator checkpoint resolves to cached blob 3ca68ae54975946b3fe5c6cd64a6eafc439b5fc43c166bc9a98d1c43aeef6618; controller checkpoint resolves to cached blob 7ead3778b95516d8b91b7c5939674e45dd0ad10fcc22da60f9f57d204695f6ac. Both actual file hashes match the frozen values.
- Frozen thresholds match both V10 and original physics-guided thresholds. Metadata, metrics and trajectories match their bound lineage output hashes. Initial-state and sensor archive hashes match their metadata bindings.
- Trajectories SHA256: 3e41b4ed172ab25776ce9ba53a64447bef60af10114e1ab6bc6056ee6b282363.
- Initial-state SHA256: 866765c4e3c965f532efae1502cd2200c52bca9271ec41d195c52fa26740abed.
- Sensor diagnostic SHA256: c7b153dc074767a2be89c6dc91bb92f9582af55a8dc20af594f2b00c2a674f9c.

No recording-interface failure or documented invocation divergence was observed. This does not certify unchanged generated motion or policy behavior relative to V10, contact measurement correctness, physical quality, semantics, hardware transfer, downstream utility or efficacy. Root subsequently reported V11/V10 motion/action differences; consequently this witness must not be used to certify a measurement-only replacement. Repeatability and V10 comparison are root's separate investigation. Receipt joint_pass=0 is retained without an efficacy conclusion. No thresholds, policy observations, sensor-selection metrics or Overleaf results were changed by the witness.

Licensed human NPZs remain on the server. Only the small numeric/hash audit document_witness_remote_audit.json was stored locally; it contains summaries and hashes, not motion arrays. All existing outputs remain preserved.

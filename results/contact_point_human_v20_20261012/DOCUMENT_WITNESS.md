# V20 independent document witness

Verdict: **PASS_RECORDING_AND_V11_INVARIANCE_ONLY**. This does not establish metric correctness, corrected velocity interpretation, physical quality, candidate efficacy or innovation.

## Exact documented invocation

Read `docs/contact_point_human_v20_20261012.md`, then invoked the following exactly once, without source edits, retries or additional simulation:

```powershell
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_contact_point_human_v20_20261012.py --check
```

Exec session **32224** completed with exit **0** and both sentinels:

- `CONTACT_SAMPLING_BATCH_PASS repeat_a`
- `CONTACT_SAMPLING_DOCUMENT_CHECK_PASS`

Remote state was `WITNESS_PASS`. The single measurement ledger entry charged **38.7164987064898 seconds**, exit 0. The witness did not invoke `--run` or execute repeat_b.

## Saved-output inspection

Read-only remote inspection confirmed:

- 12 metric rows and 120 recorded steps, indices 0–119, covering all 12 environments.
- **53,764** native contact records; **0** parser/native read errors.
- The structured dtype has `localPos0`/`localPos1` rather than snake-case names. Actual fields were `env0`, `env1`, `body0`, `body1`, `localPos0`, `localPos1`, `minDist`, `initialOverlap`, `normal`, `offset0`, `offset1`, `lambda`, `lambdaFriction`, `friction`, `torsionFriction`, `rollingFriction`.
- Observed body indices ranged from -1 (ground) to 23. Native records also contain self-contact pairs; all records must not be interpreted as ground contacts.
- Native force magnitudes were finite, nonnegative, from 0 to **681.279052734375 N**. **10,189** ground contact records had positive native force.
- All 12 environments saved 24 body mass/local-COM entries; properties were identical across environments. Foot-body local COMs are nonzero. This establishes availability, not the reference point of exported linear velocities.
- Initial reset consistency passed. All seven reset-state maximum differences were exactly zero.
- The saved native-contact file matched its metadata SHA256 binding. Source manifest and all **20** version source bindings matched current remote bytes.
- CPU tensor exports and GPU PhysX remained enabled as documented (`use_gpu_pipeline=false`, `physx_use_gpu=true`); policy observation mode was 1; 242 physics completion fences were recorded.

## Passive-instrumentation comparison

Compared the new repeat_a against V11 repeat_a remotely, without downloading human motion/manifold arrays. Maximum numerical difference was **0** for every saved trajectory field:

`actual`, `reference`, `quaternion`, `linvel`, `angvel`, `contact_force`, `valid`, `terminated`, `reference_frame`.

Maximum difference was also **0** for every sensor/policy diagnostic:

`sensor`, `root`, `dof_pos`, `dof_vel`, `action`.

Thus this recording instrumentation preserved V11 trajectories/actions for this fixed 12-input, seed-11 invocation. It does not prove invariance for arbitrary tasks or model settings.

## Evidence hashes

- Metadata: `269e42ff2fd4a49004ac7e95c87f5a2bb23f72ea56c9750edd56506188a51714`
- Native contacts: `60ea5f42e4fab9b6f93d239396ed9dd3f2aa80654391f6e24c5d2068ae0f5605`
- Versions: `4c4512799f11b2ab56635426ffe61aec579624c37e96177fa557b61d4894ae27`
- Local compact inspection JSON: `211ee359eed44be986557c36c49c7ebd5b0bfcc9733cb6d3ae7852b9c31ac2d4`

## Limits and next evidence needed

The original frozen metric was preserved (`thresholds_changed=false`, `metrics_replaced=false`, `innovation_confirmed=false`). Native read success does not establish articulated-support slip calibration, the velocity reference point, contact-force temporal equivalence, motion quality or downstream utility. Native `lambda` can reflect different temporal aggregation from the all-substep net-force tensor. A second repeat and separate known articulated contact/velocity calibration are still required before replacing any physical-quality metric.

No Overleaf writes, checkpoint changes, policy updates or raw human trajectory exports were performed. Only this independent report and compact inspection JSON were written locally.
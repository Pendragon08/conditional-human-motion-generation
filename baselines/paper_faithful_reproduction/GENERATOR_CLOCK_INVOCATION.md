# Generator clock/frame numerical witness

Scope: warm reuse of the existing local CPU environment; no installation,
environment edit, GPU allocation, RL training, physics call or data modification.
Provider ledger: `../phygile_unitree/docs/COMPUTE_LEDGER.md`, canonical spec hash
`a43f960bd3bcdcd3398655bffe291138b73de5ca7f30901765172ff9821e1954`.
The original authorized 40-sequence clock-gated SSM subset is read in place.
Restricted arrays are not copied or published. The output contains numerical
summaries, hashes and full unit-test logs.

Run the following verbatim in PowerShell. The output path must not exist; an
existing path deliberately fails rather than overwriting experiment evidence.

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:/phygile_unitree_env_20261022/Scripts/python.exe' 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/paper_faithful_reproduction/scripts/verify_phygile_generator_clock.py' --manifest 'D:/具身智能/PhyGile与ReactiveBFM_论文对齐复现_20261022/phygile_clock_gate_v1/manifest_clock_consistent.json' --phygile-code 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/phygile' --output 'D:/具身智能/PhyGile与ReactiveBFM_论文对齐复现_20261022/phygile_generator_clock_doc_witness_v1'
```

Expected: exit0, `PASSED_NUMERICAL_CLOCK_WITNESS`, six unit tests pass,
40 real sequences checked, 34 training sequences/1472 training frames,
`inputs_unchanged=true`; maximum world-pose codec context roundtrip <1e-5,
maximum unrepresented tail <=0.0100000001s, no extrapolation/time stretching.
The unit proposer uses an explicitly analytical oracle in place of diffusion
sampling. These checks certify clock/frame arithmetic and preservation of
committed actual prefix and terminal states; they do not certify learned
semantics, physical feasibility, trained generation or paper reproduction.

Implementation boundaries:

- New real training checkpoints bind every sequence's codec FPS/frame sidecar;
  legacy checkpoints without that contract fail and need an audited migration.
- A real trained backend uses paper CFG2.5; the random historical engineering
  backend remains explicitly unconditional, 50Hz and CFG1.
- Feature times end at the actual execution boundary. Poses are resampled with
  linear translation/joints and shortest-arc root quaternion SLERP. Velocities
  are recomputed at the output rate. Any extra generated tail is recorded.
- Cold start's unavailable history is explicitly a held-pose initialization.
- Final time must fall on the feature grid, and only a one-frame suffix is
  supported. At20/50Hz, a final control count must be a multiple of5. Off-grid
  terminal time or a multi-frame suffix fails before sampling.
- The codec integrates root velocities, so terminal world XY/orientation is
  an explicit output projection, independently tested in physics later; known
  feature inpainting alone cannot guarantee an absolute terminal root pose.
- The older safety bounds are retained and recorded. They are not a physical
  filter and do not establish successful tracking.

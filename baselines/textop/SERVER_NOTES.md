# TextOp environment ledger

### env: textop@92b1d1b1
how: isolated venv on nlp03, shared base read only.
tier: CPUs 4, GPU1 A6000, CUDA allocation cap 65%.
status: spec written BEFORE environment build; validation pending.
source: TeleHuman/TextOp ef6555fb174c9b5c44945a62c7ffc77b5ddbbf22
assets: Yochish/TextOp-Data bb9604ce36ad4c0a77b5cccde89ebbb7c3ca93c9

### env: textop@c5f7e5fa
Spec corrected: isaac_utils is at upstream/deps/isaac_utils, not under TextOpRobotMDAR. Original failed build log retained. Rebuild pending.

### env: textop@9d7f370d
Build metadata fix: pin setuptools68.2.2/wheel0.42.0 before editable packages. Older base setuptools incorrectly labeled RobotMDAR UNKNOWN. Isolated overlay only; base not modified. Validation pending.

validated: spec9d7f370d seeded official MVAE CUDA forward/backward passed; fresh independent doc witness_v2 passed verbatim SSH invocation, exit0, no divergence. No trained result claimed by environment witness.

validated: spec9d7f370d fresh independent doc witness_v3 passed verbatim SSH invocation after corrected training-observability documentation, exit0, no divergence. Shared environment unchanged; attempt1 preserved, attempt2 uses official skip behavior and separate counters.

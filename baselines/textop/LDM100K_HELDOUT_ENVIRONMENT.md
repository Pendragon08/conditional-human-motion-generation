# CURRENT — heldout scope helper tested; actual scope/generation/evaluator unaccepted

Keep the user endpoint100k, original LDM/observer frozen and all model jobs
serialized. At16:12 the original run was33800attempts/33800Adam/0skip and the
controller separately observed33900/0signals; newer real receipts take priority.
No100k fullstate, export or verification exists yet. Do not activate them early.

New `scripts/ldm100k_heldout_protocol.py` and
`provider_tools/test_ldm100k_heldout_scope.py` are deployed. The actual local and
server Python3.8 isolated fixtures passed16new rejection cases and segment/T-1/
history/overlap/subset boundaries; server child0/reaped/final streams. No original
arrays or per-source text, models, GPU, generated samples or metrics were read
or produced. Do not rerun these fixtures or old26/18/10 suites on a heartbeat.
Receipt: results/ldm100k_user_handoff_20261028/heldout_protocol_preparation_v1.json.

`configs/ldm100k-heldout-plan.json` SHA
fc30daa332971b6e56a35347d9463e5dee3772e0a78e0c7d0cc293b0fd074247
has scope_build_authorized=false, generation_authorized=false, model_bindings=null,
text_motion_feature_extractor=null and0actual jobs. This is not a ready runtime.
The actual metadata scope builder, signed activation, bounded model-generation
entry/supervisor, fresh real document invocation, TMR encoder acceptance and
physical endpoint entries still need implementation and actual verification.

## Declared heldout diagnostic protocol

Use only original, unmirrored, physically accepted BABEL validation families
from the accepted filtered corpus. Verify all prepared manifests/source split/
view receipt/data acceptance bindings. Keep source/mirror families together;
train families and quarantined training duplicates are not heldout inputs.
Do not rebuild the corpus, normalization or CLIP cache. Use TRAIN-only69D stats
and the exact existing training text embedding convention.

The pure scope helper receives signed metadata rows only. It preserves all
annotations separately, including concurrent labels and duplicate texts. Time
indices use floor(start_seconds*50), ceil(end_seconds*50), capped to T-1 pairwise
features. Targets start at least2 features into the sequence; the preceding
two features are the only reference context. Windows without a valid future
are recorded as omissions, not physical failures. Each target chunk is at most
200future frames, with a separately recorded final generated tail below8frames.

Each diagnostic segment starts from its own two past reference features, then
uses only generated history. No future posterior encode or ground-truth future
condition is allowed. This independently initialized diagnostic is not a
continuous full-sequence rollout or paper transition evaluation. Concurrent
annotations must not be stitched into an invented transition order.

The first diagnostic selects32families by a fixed source hash ordering, keeps
all their annotations/chunks, and uses five fixed seeds20261028..20261101 in
conditional CFG5 and unconditional modes. The same segment/seed gets the same
initial history and diffusion-noise sequence in both modes. No best-of seed,
failed-sample deletion, prompt rewriting or CFG tuning. A None family limit is
the scope for all validation families; it still does not establish the paper's
continuous rollout/transition protocol. Actual source/segment/trial/frame and
storage/runtime counts must be measured before enabling a real job. A planned
trial is never a completed sample; partial outputs and all failures persist.

## Primary-source evaluation boundary

The paper describes BABEL-validation segments capped at200frames, transition
clips extending15frames on each side, and TMR-based encoders trained on BABEL
text-motion pairs; retrieval/MM-Dist use32-pair batches.
[TextOp appendices B-A/B-B](https://arxiv.org/html/2602.07439v1#A2.SS2).

The inspected official repository tree ef6555fb174c9b5c44945a62c7ffc77b5ddbbf22
contains generation/visualization/frequency scripts; no TMR/retrieval-named entry
was located in that tree. This observation is not proof that no external
evaluator exists. Pinned, MIT-licensed source-only references and exact Git-blob
proofs are archived under sources/author_generation_eval_20261028 and
results/ldm100k_user_handoff_20261028/author_evaluation_source_check_v1.json.
[Official code](https://github.com/TeleHuman/TextOp/tree/ef6555fb174c9b5c44945a62c7ffc77b5ddbbf22/TextOpRobotMDAR/robotmdar/eval).

Actual TMR Rprecision/MMdist/FID requires an accepted robot-compatible learned
text-motion feature extractor, signed training/data/family-split/normalization
and loading evidence. None is currently bound. Do not label a CLIP-cache score,
VAE reconstruction loss, arbitrary human-feature checkpoint or random features
as those accepted metrics. Any independently trained evaluator must use TRAIN
families only, keep actual BABEL validation held out, disclose its separate
weights/protocol and have its own finite capacity/resource/recovery acceptance.
Do not overlap its model training with LDM or extend the user's generator100k.

Continue preparing actual source-bound scope/generation and compatible metric
inputs while LDM runs. After the genuine100k stop and actual CPU export/fresh
verification, execute real heldout diagnostics and then the complete planned
coverage, with signed artifacts and no missing/failed sample relabeling. WBC,
true physical tracking endpoints, streaming transitions and licensed films
remain separate unaccepted results. Follow the existing shared lock, GPU1/.65,
CPU0-3, per-filesystem2GiB floor and private-data boundaries throughout.

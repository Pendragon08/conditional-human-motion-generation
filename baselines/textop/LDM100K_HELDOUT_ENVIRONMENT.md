<!--ldm100k-live-1809-start-->
CURRENT实核：2026-10-28T18:09:02+08:00原LDM54200尝试/54200Adam/0skip，原supervisor304265/start1436142089及worker304272/start1436142105一致、RUNNING无exit、共同锁占用、native stage1，真实完整DDPM history生成4199次/选用114primitive。18:10:26唯一控制另采54400、WAITING/0信号；不是保存检查点。VAE100k私有D副本已通过，不重传；p997_v2原worker73588/SFTP15724创建身份实核仍匹配，48201728B partial、D6383742976B采样，仍PENDING。C写入救援已完成不重跑，C约0.50GB可用但仍轻量观察。
新实际CPU scope-builder/签名激活/独占派发/有界监督/聚合收集已部署禁用，specSHA303d7261db46476aa539d2a524011bba1e9386f6e1bec2cc9aa24e4fed300c26，scope_build=false/verified_model=null/generation=false。新31隔离契约负例远端CPU0/reaped/final streams，原活动源/spec冻结未动，0实际scope/模型/数组/GPU/信号，不是实际接受。读LDM100K_HELDOUT_SCOPE_ENVIRONMENT.md，真100k停止→CPU导出→fresh实际CPU summary闭合后才一次启用/执行。现有12/16/18/26不重跑。实际模型生成entry/监督、兼容TMR及物理端点仍需继续实现和实际验收，不能以本阶段准备停工或称完整效果。
<!--ldm100k-live-1809-end-->

<!--ldm100k-live-1748-start-->
CURRENT实核：2026-10-28T17:48:25+08:00原LDM51000尝试/51000Adam/0skip，同原supervisor304265/start1436142089及worker304272/start1436142105，RUNNING无exit/commonlock占用、实际stage1，真实DDPM history生成999次/选用2primitive；不是保存点。17:49:32唯一控制另采51200、WAITING/0信号，用户100k端点不变。VAE100k私有D备份p100k_v1真实SFTP0/worker0、完整513526765B/SHA通过，已释放备份锁，p997_v2唯一续原32997376B前缀，仍PENDING。
本机C盘空闲0导致源码测试文件写入失败，已保留失败字节；经D私有完整恢复包/SHA/ZIP CRC/所有权及GitHub恢复代码+聚合证明实核后，只移除499702381B可恢复下载/编译缓存，0数据/checkpoint/环境/训练源删除或修改。C恢复可写但余量有限，后续只轻量观察，不重跑本次清理。见reports/c_space_rescue_and_ldm_progress_20261028.md和results/storage_c_space_rescue_20261028/final_summary.json。私有缓存/清单绝不公开。scope-builder/verified-input已有本地源码及12隔离元数据负例本地0，但没有监督/激活、未部署/无真实scope、生成或模型作业，非ready；继续完成实际入口，勿称实际验收。
<!--ldm100k-live-1748-end-->

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

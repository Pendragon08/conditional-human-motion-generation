# Independent online document witness — online_document_v1

Executed the documented durable command exactly once after63 actual preflight checks. The unchanged author run had closed and the common lock was free. Shared-GPU authorization was respected: GPU1 free40726MiB; combined four CPU cores, generator0 and SDK1–3. No packages, old profiles, weights or execution source/configuration were changed.

```powershell
& 'C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' 'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop/provider_tools/launch_online_quick.py' --run online_document_v1
```

The wrapper executed `python3 scripts/supervise_online_quick.py --run online_document_v1` once on the original server. Actual outer, generator and SDK return codes were all0; both children were reaped, no timeout/supervisor signal, output writers closed and lock released. Native SDK close remained false and the bounded fallback was used.

Independent local examination of the SHA-verified private archive passed430 checks; postflight passed41 checks. This included all packet hashes/CRC, unchanged published reference prefixes,35 actual generated primitives' own-history hashes, real pretrained VAE2000/LDM3000/teacher5000 hashes and training receipts, source before/after hashes, actual current+future availability, pre-autoreset179-column binary against every NPZ field,258 state samples,29 named joints, all timing records, resource samples, and exact recomputation of error metrics. No extra SDK or GPU experiment was dispatched for the audit.

The actual run completed257 timed control calls/1028 physics substeps;16 warmup calls/64 substeps were counted separately. Every scheduled20ms deadline passed with zero tolerance:0 missed deadlines,0 underruns,0 physical failures. Control-loop elapsed5.131989819929004s for257×20ms nominal5.14s; postprocessing-inclusive elapsed5.161067120730877s must not be substituted for control timing. Text was submitted live at control90 and the new generated target consumed at114; measured submission-to-consumption0.49660042114555836s includes buffering. Input was the real pretrained model's own reference history, with no future posterior or executed-pose conditioning.

Current-target mean14-body error0.04274287074804306m and jointRMSE0.12755894660949707rad are recomputed diagnostics, not paper aggregate/terminal-endpoint scores. The profile explicitly uses single-robot CPUPhysX plus GPU1 neural inference. Training-only reward evaluation was omitted during timed inference; returned zeros are API placeholders, not measured rewards. Models, strict thresholds and ground configuration were unchanged.

No doc-versus-invocation divergence occurred in this witness. Previously documented local FK spec CRLF/server LF difference was handled by checking immutable actual server bytes, without reserializing it. This is a bounded scripted online-chain witness, not human-facing interactive UI delivery, unlimited hard-realtime guarantee, semantic-quality acceptance, terminal-reference acceptance, full-data filtering, final deployable WBC or paper-scale reproduction. The fixed LDM sampling-quality regression remains unresolved; zero timing misses does not resolve it. No online underrun/physical-failure branch occurred here, so that branch is not certified by this successful run.

Frozen online spec SHA:818d9e0dd951eaa5b2bb3f3e5d7858f2322cf7af740348f1d59190554fca4430.

Private raw archive:18,847,521 bytes,128 members,SHAe70fc987ba4fe1e8a926ca5f31160357b4b915491b7108e223b8dbf46fbd3548. It remains on the server and D:/具身智能/TextOp复现_20261024/diagnostics/online_quick_20261026/online_document_v1/evidence.private.tar.gz. Pre/postflight full source bytes and independent auditor/evidence are in D:/具身智能/TextOp复现_20261024/diagnostics/online_document_v1. Only this REPORT and the aggregate SUMMARY belong in public records.

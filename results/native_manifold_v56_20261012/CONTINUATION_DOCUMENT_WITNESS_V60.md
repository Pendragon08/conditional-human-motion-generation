# V60 continuation document witness

**Status:** FAILED_RETAINED

The independently documented V60 continuation command was executed exactly once. It failed during preflight before manifest creation or random replay. No retry and no original V56 --run invocation were made by the witness agent.

## Invocation

```text
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 env PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/continue_native_manifold_v60_20261012.py
```

- Invocation count: 1
- Exit code: 1
- Expected sentinel observed: false
- Failure stage: V60 preflight before manifest creation and random replay
- Failure: the transformed continuation passed a string to verify_execution_lineage_v39.sha, which requires a Path and calls path.open().

## Preserved evidence

| Artifact | SHA256 |
| --- | --- |
| V60 document | 29421bb2f7a1f703ccfe443bf2ae15f278f1a934b4ecada0b02ed62631f413eb |
| V60 source | 1868c969b9a0a75a522f5e9e32e55a7e1f705a9824e341052ba860a5c08331bd |
| Bound V58 source | 81f97153cde8d1be81452b00e629fc0dc06e897c33407e2901a56c3f7d9f82a0 |
| Bound V58 failure witness | 881186f774ee2006c8bc3108ea569fc1fbeb5d44ea30388beb0b8890860dfdac |
| Prior V56 document witness | b0c109cd6dd1fd9015fbdf4dbd20b59ccc877edcfa76a5d17bddf4351979f589 |
| Prior V56 runtime witness | 4a8a9a4ea35a479b429922b3d29c523df891cf40703b96b428cf58dcb878ce7a |
| First failure state | 656983246104825dbb7e7e845d73fd73479f7858c98f0a2e0df9fbbb1be5a8d9 |
| Feedback trajectory | 54e45f61944bd32e11c49185bbf1172432e0781f335e977bb98f3c2844c16bca |
| Unchanged one-row ledger | 0a559572436bae7467d882b971a95ae12b2ae0c213850a2a09bc941f1d2ac184 |

The prior hashes remain unchanged. The raw directory still contains only val_feedback_replay_s11, and the ledger still contains its single successful measurement charge.

## Missing continuation outputs

- Continuation manifest: absent
- Continuation receipt: absent
- Continuation summary: absent
- random_equality.json: absent
- Random batch started: false

## Exact failure output

```
Traceback (most recent call last):
  File "/data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/continue_native_manifold_v60_20261012.py", line 26, in <module>
    exec(compile(source,str(Path(__file__)),'exec'),globals())
  File "/data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/continue_native_manifold_v60_20261012.py", line 56, in <module>
  File "/data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/continue_native_manifold_v60_20261012.py", line 33, in main
  File "/data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/verify_execution_lineage_v39_20261012.py", line 29, in sha
    with path.open('rb') as handle:
AttributeError: 'str' object has no attribute 'open'
```

This witness records the failed continuation attempt only. It makes no metric, efficacy, C2/C3, innovation, or real-contact claim.

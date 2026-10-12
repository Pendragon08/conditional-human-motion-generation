# V61 continuation document witness

**Status:** PASS_DOCUMENT_FOLLOWING_V61_CONTINUATION

This independent witness certifies the single documented V61 continuation invocation and passive-export equality only. It does not certify efficacy, C2/C3, innovation, or real contact ground truth. The prior V56, V58, and V60 failed entry attempts remain recorded as failures.

## Invocation

```text
ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 env PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/continue_native_manifold_v61_20261012.py --run
```

- Invocation count: 1
- Exit code: 0
- Standard output: `NATIVE_MANIFOLD_V61_CONTINUATION_COMPLETE`
- Retry performed: false
- Prior failed entry invoked: false

## Bound outputs

| Artifact | SHA256 |
| --- | --- |
| V61 document | `0e0fbfadd9608efdb28d8afbfad992ad15fa9ed9c1dfab501f404649e443781b` |
| V61 source | `355511d142bea4bf3b2788bf39da3d2a9284b70ab06e64b23a96a45a1a93923f` |
| CPU preflight | `d10eea0cc045ec19d327c05ae6d5053312631d64745966b64abbd3aea0ded3f3` |
| Continuation manifest | `66dd45e8e792faade8be526e450c52c68e6d7ce4dac4563cab10a262212c08d0` |
| Continuation receipt | `24a9269e305e46de378d96660d2abc9388fcaca02d2a88a376997e1742f23725` |
| Continuation summary | `c1c0f3e05f0973fda51e693dde94840d1912e6d4dcd5ae32c1332ea98261846e` |
| Random equality | `4b3b15236c6967867b50649a2921683042c46a199fb8e6144a3b2d8aac1c1cef` |
| Final budget ledger | `a4f027f83a7c10fe5fffb27229d78aac9988a3778becfd82bda43b9e2ce99c0a` |

## Random replay checks

- Status: `PASS_PASSIVE_EXPORT_EQUALITY`
- Cases: 72
- All 33 recorded array comparisons: true
- Frozen/original metrics exactly equal: true
- Generation calls: 0
- Native contact records: 288411, nonzero
- Native parser errors: 0
- Native contact SHA256: `97dea899b1435e818d61b416b772cff0f41e73fe8d02aeecceb552ca2c8e2f35`

## Budget and preserved failures

- Ledger entries: 2
- Both entries are successful measurement charges: true
- Total passive measurement seconds: 162.78181958571076
- Raw batches: `val_feedback_replay_s11`, `val_random_replay_s11`
- Feedback trajectory SHA256: `54e45f61944bd32e11c49185bbf1172432e0781f335e977bb98f3c2844c16bca`
- First failure state SHA256: `656983246104825dbb7e7e845d73fd73479f7858c98f0a2e0df9fbbb1be5a8d9`
- V58 failure witness SHA256: `881186f774ee2006c8bc3108ea569fc1fbeb5d44ea30388beb0b8890860dfdac`
- V60 failure witness SHA256: `48366f43dfae9819dfcd7fc81aac0ee83c1e72d9a5f09f59c9c2e42c25616b2e`

All preserved hashes remained unchanged. No offline metric analysis was performed.

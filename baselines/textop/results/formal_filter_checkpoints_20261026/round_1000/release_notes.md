TextOp 29DoF privileged filtering-policy inference snapshots.

Training target: 40,000 iterations on 8,278 available motion sources. These are intermediate checkpoints, not completed paper or closed-loop results. The 557D privileged actor/critic uses a declared validated static solid-floor contact representation; exact author geometry is unavailable.

Public exports contain network and observation-normalizer tensors plus aggregate iteration metadata. They omit per-motion sampling statistics, optimizer state, raw motion arrays and text/source indexes. Full training-resume checkpoints remain privately backed up on the server and local D: drive. No server checkpoint is removed.

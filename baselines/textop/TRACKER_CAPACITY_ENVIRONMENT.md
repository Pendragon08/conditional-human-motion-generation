# 8192-environment capacity diagnostic, independent document check

Read SERVER_NOTES.md and configs/tracker-capacity-spec.json first. No packages
or shared sources changed. Warm backend canonicalb7a3bf0c, shape37c64682:
Python3.10.15, IsaacSim4.5.0, IsaacLab project2.1.0/distribution0.36.21,
Torch2.5.1+cu118, RSL-RL2.3.1, NumPy1.26.0, Gymnasium0.29.1.
GPU1 shared use, CPU4, Torch allocator configured0.65 are user-authorized.

Author_v1 passed one iteration with8192 environments,24 steps/environment,
196608 transitions and20 actual Adam updates. It uses completed643 original
sources+mirrors (1286 trajectories,946700 frames), not the final full corpus.
Peak Torch allocated10,292,943,872bytes; reserved11,414,798,336bytes. These are
Torch counters, not total driver/PhysX memory. The supervisor samples both
devices' total used memory including other processes. Exact peaks/checkpoint
hash may differ because contact dynamics need not be bitwise deterministic.

The random untrained policy has large regularization/contact spikes and value
loss. Finite gradients and memory success do NOT establish training stability,
tracking quality, physically accepted data or full40k training readiness. All
such flags remainfalse. The complete future library needs another memory test.

## Execute verbatim once in PowerShell

capacity_document_v1 must not exist. No retries, repairs, changing flags, other
process actions, source mutations, video claims or full training launches.

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -i 'C:/Users/Pende/.ssh/id_ed25519' ouyanghao@10.147.17.32 'python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/supervise_tracker_capacity.py --run capacity_document_v1'
```

Expect directexit0 and TEXTOP_8192_CAPACITY_SUPERVISED_EXIT0. Read:

- receipts/tracker_capacity_capacity_document_v1_exit.json: child_exit_code0,
  timed_outfalse, numerical_statusTEXTOP_8192_CAPACITY_PROBE_PASSED.
- tracker_validation/capacity_document_v1/receipt.json: num_envs8192,
  optimizer_updates20, completed_learning_iterations1, transitions196608,
  policy/critic_observation_dim557, reference_source_sequences643,
  reference_trajectories1286/reference_frames946700, observed affinity4 CPUs,
  torch_threads4, torch_allocator_fraction_configured0.65, parameter change>0,
  finite/memory assertions passed, saved checkpoint SHA independently verifiable,
  output_writers_closedtrue. Dataset/filter/generator/full-corpus flags allfalse.

Source SHA2569b579a2c5ab6bb427e88f81fc9aac12fa3b0f2d3c4511b7e0070fd2c8b7a9e7a.
Capacity spec file SHAd805fbb7c472a68392f68ad705302accf33370e47307b6aedfd54802b549a5bb.
Private reference manifest SHA5601a109e150f9ef991f4dda0b1f618b743c278db3b64c509642d36f2eca6158.

Native stop/close remains problematic. If the20s fallback is used after closed
writers and durable checkpoint/receipt, require native_close_returnedfalse,
teardown_fallback_usedtrue, shutdown_completion_verifiedtrue; disclose it rather
than calling it native close success. Supervisor retains the actual exit status.

Keep private arrays/checkpoints/USDs onserver. Save command/stdout, public
numerical/supervisor receipts, shutdown stack, relevant logs and REPORT.md in
work/textop_reproduction_20261024/tracker_capacity_doc_witness_v1. Do not copy
private arrays, publish, message other chats or repair divergences. Report the
large untrained losses/forces as a limitation, not successful motion tracking.

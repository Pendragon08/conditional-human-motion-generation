# Real PPO update and resume diagnostic

Read SERVER_NOTES.md and configs/tracker-ppo-diagnostic.json. Warm unchanged
backend b7a3bf0c: IsaacSim4.5.0, IsaacLab project2.1.0/distribution0.36.21,
Python3.10.15, Torch2.5.1+cu118, RSL-RL2.3.1, NumPy1.26.0, Gymnasium0.29.1.
No installations or shared changes. GPU1, CPU4, Torch allocator cap65%.

Author_v2 passed: two complete source motions plus mirrors,8 environments,
557D observations, paper MLP2048/1024/512,24steps,4minibatches,5epochs.
One learning iteration, save/load, one resumed iteration =40 actual Adam steps.
Model, Adam and both normalizers restore with exactly equal values/dtypes/shapes.
Adam scalar step CPU residency is allowed by PyTorch. RSL's last-index counter
is deliberately advanced to completed-count1 in the separate resume checkpoint;
this avoids repeating completed iteration0. Checkpoints/logs are retained.

Original author_v1 reached20 Adam steps before a comparison-device error; its
stale zero counter and the correction are both retained. No policy-quality,
physical acceptance,8192-env capacity or full training claim is permitted.

## Fresh document invocation, once only

Run exactly once in PowerShell. No repair, retry, process stopping, private-array
copying, checkpoint deletion or training beyond this diagnostic is authorized.

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -i 'C:/Users/Pende/.ssh/id_ed25519' ouyanghao@10.147.17.32 'python3 /data/code/ouyanghao/oyh/textop-reproduction-20261024/scripts/supervise_tracker_job.py --job ppo_witness --run ppo_document_v1'
```

Expected directexit0, TEXTOP_TRACKER_JOB_EXIT0. Supervisor receipt
receipts/tracker_job_ppo_document_v1_exit.json: child_exit_code0,timed_outfalse.
Numerical receipt tracker_validation/ppo_document_v1/receipt.json:
TEXTOP_TRACKER_PPO_RESUME_WITNESS_PASSED,optimizer_updates40,num_envs8,
environment_transitions384,completed_learning_iterations2,policy_observation_dim557,
model_restored_exactlytrue,adam_restored_exactlytrue,
both_normalizers_restored_exactlytrue,full_training_startedfalse,
dataset_acceptedfalse,physical_filter_executedfalse,output_writers_closedtrue.
Two verified checkpoints of90,600,722 bytes each are expected; their SHA values
need not equal author's because contact simulation need not be bitwise identical.

Source SHA256 cfff854196ebf022377b9a1abcf0698a038e45023d3abf3c84c66232d3068e32.
Diagnostic spec file SHA256 c978be1ace1d759f0020f946a6211c7ce94cda81e6ef0f3413399161c191681a.
The native stop/close path can hang. Only after test completion, durable
checkpoints/receipts and closed writers, a20s watchdog exits this private process.
If fallback is used, require shutdown_completion_verifiedtrue and report
native_close_returnedfalse/teardown_fallback_usedtrue explicitly. The supervisor
still rejects any failed receipt even if the native SDK returned processcode0.

Keep private references/checkpoints/USDs onserver. Save command/stdout/receipt,
supervisor proof, shutdown stack and a concise doc-vs-reality report locally.
Do not publish or message other chats. Local witness evidence folder:
work/textop_reproduction_20261024/tracker_ppo_doc_witness_v1.

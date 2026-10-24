# Unitree actual generation-execution replay witness

Read COMPUTE_LEDGER.md and configs/environment_plan_v2.json first; warm reuse, no dependency changes. User authorized headless simulation. No new PPO in this test. All original assets are read-only. ReactiveBFM must remain untouched.

Execute this one PowerShell command once, without repairing source or retrying. Source code_v11 is immutable. Fresh output must not exist:

~~~powershell
& 'C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/phygile_unitree/scripts/submit_closed_loop.py' --version code_v11 --name closure_unitree_v3
~~~

Read-only poll the returned run directory verification.json, logs/closure_unitree_v3_stdout.txt and stderr.txt. Poll every 30 seconds or slower; any wait <=60 seconds; limit 15 minutes. Never kill other jobs. Preserve failure and do not retry.

Acceptance: status COMPLETED_ENGINEERING_VALIDATION, actual Unitree-G1-PhyGile-v0 physics workers, sources unchanged, frozen GMT/generator unchanged, same-seed fresh-process replay witness, two accepted one-frame feedback proxy windows with history replay, actual diffusion sampling and rejection without invalid commit. The generator is untrained/unconditional; a rejected 50-frame generated window is an expected negative result, not semantic generation success. Proxy feedback is not a motion benchmark. Preserve raw private arrays on server; publish only hashes/receipts/metrics.

Save complete command/output/launch/process status, verification.json and loaded source manifest hashes in a fresh local witness directory. No external notifications or automations.

V2 failed before physics because CPU20-23 do not exist (allowed0-19). V3 validates CPU16-19 before launch and persists the real child exit code in logs/closure_unitree_v3_exit.json. Original V2 evidence is preserved.

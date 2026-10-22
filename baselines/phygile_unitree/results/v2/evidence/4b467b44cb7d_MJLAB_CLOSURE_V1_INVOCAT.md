# Unitree MuJoCo actual closed-loop migration witness

Read COMPUTE_LEDGER.md and configs/environment_plan_v2.json. Exact unchanged local environment, warm reuse via ASCII interpreter. No install/source changes. Bounded native prefix replay smoke already passed two actual frames with delta0; no learned-generation claim. This invocation uses an immutable source snapshot source_mj_closure_v1/source_manifest.json. If saving a .ps1 wrapper for Windows PowerShell5.1, write UTF-8 WITH BOM.

Execute exactly once:

~~~powershell
& 'D:/phygile_unitree_env_20261022/Scripts/python.exe' 'D:/具身智能/PhyGile复现/Unitree框架适配_20261022/source_mj_closure_v1/scripts/run_mjlab_closed_loop.py' --parent 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/phygile' --unitree-checkout 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_unitree_20261022/vendor_unitree_rl_mjlab' --assets 'D:/具身智能/PhyGile复现/Unitree框架适配_20261022/private_assets_contact_v3' --output 'D:/具身智能/PhyGile复现/Unitree框架适配_20261022/closure_mjlab_v1'
~~~

Save command/full stdout/stderr/actual exit code separately in fresh closure_mjlab_doc_witness_v1. Limit20 minutes, wait <=60sec each. Do not retry, repair sources, install dependencies or operate server processes.

Acceptance: actual process exit0 and verification.json COMPLETED_ENGINEERING_VALIDATION; actual_mujoco_warp_physics and actual_diffusion_sampling true; frozen checkpoint/generator and sources unchanged; fresh-scene replay deltas <=2e-6; two accepted one-frame held-initialization proxy windows with exact accepted-history replay; two attempted 50-frame diffusion windows, rejection allowed and failed motions never committed. This is untrained/unconditional generation wiring, not stable semantic motion generation. No truth fabricated from held-pose proxies. Preserve all negative outcomes and licensed raw arrays privately; publish hashes/metrics only.

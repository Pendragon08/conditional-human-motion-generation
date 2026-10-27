# Paired sliding-friction diagnostic

Warm reuse the exact local environment in COMPUTE_LEDGER.md and configs/environment_plan_v2.json. No install or dependency/source repair. Read the existing kernel and doc witness records. Use the ASCII interpreter junction required by MuJoCo on Windows. Original private_assets remains unchanged; private_assets_contact_v3 holds a new deterministic PhysX contract measured from material_contract_v2 (all source materials [1,1,0], ground [1,1,0], multiply combine).

Execute exactly once in PowerShell:

~~~powershell
& 'D:/phygile_unitree_env_20261022/Scripts/python.exe' 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/phygile_unitree/scripts/run_contact_comparison.py' --unitree-checkout 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_unitree_20261022/vendor_unitree_rl_mjlab' --assets 'D:/具身智能/PhyGile复现/Unitree框架适配_20261022/private_assets_contact_v3' --output 'D:/具身智能/PhyGile复现/Unitree框架适配_20261022/contact_comparison_v3'
~~~

Save full stdout, stderr, actual exit code and a command receipt in a separate fresh contact_doc_witness_v3 directory next to the comparison output. No retry or source repairs; preserve failure. Run eight actual CPU MuJoCo-Warp first episodes: official and matched_sliding_friction, four clips each, seed2026. No policy updates. Acceptance requires comparison.json status EIGHT_PAIRED_DIAGNOSTICS_COMPLETED; eight child exit codes0; unchanged executed sources; finite first-episode metrics; actual compiled foot coefficients0.6 and1.0, respectively. Improvement is not an acceptance condition. These four clips are development diagnostics, not a held-out benchmark. Never publish licensed raw motions or private recorded states. No server/ReactiveBFM process operations.

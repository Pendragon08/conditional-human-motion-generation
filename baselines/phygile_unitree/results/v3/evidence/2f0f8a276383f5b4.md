# Source collision primitive diagnostic

Read the run-experiment skill/compute contract and baselines/phygile_unitree/docs/COMPUTE_LEDGER.md plus configs/environment_plan_v2.json in the conditional-human-motion-generation checkout. Warm reuse, no package/model edits. Use a UTF-8 WITH BOM wrapper for Windows PowerShell5.1.

Execute exactly once in PowerShell:

~~~powershell
& 'D:/phygile_unitree_env_20261022/Scripts/python.exe' 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_unitree_20261022/run_geometry_comparison.py' --base-scripts 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/phygile_unitree/scripts' --unitree-checkout 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_unitree_20261022/vendor_unitree_rl_mjlab' --assets 'D:/具身智能/PhyGile复现/Unitree框架适配_20261022/private_assets_contact_v3' --source-urdf 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_unitree_20261022/source_main_g1.urdf' --output 'D:/具身智能/PhyGile复现/Unitree框架适配_20261022/source_geometry_comparison_v1'
~~~

This is an explicit experimental wrapper, not an edit of installed official assets. It derives 29 primitives from the source URDF SHA dbc8471004e9b0b85c27a4ebc1f0d90a727c1cf7ca9ef507b805596f2004d64b; cylinders become capsules as requested in the actual Isaac import. Preserve original articulated dynamics, PD/action/observation contract and official contact-dimension rules, with measured sliding friction1.0. It does not assert exact PhysX/MuJoCo equivalence. Evaluate all four development clips seed2026, original parent policy frozen. Do not select a best profile based on these four clips.

Accept actual process exit0, comparison status FOUR_GEOMETRY_DIAGNOSTICS_COMPLETED, four real physics first-episode receipts, finite raw metrics, matched source/profile hashes, unchanged executed sources and exactly29 source definitions. Early termination and no improvement are valid outcomes, not validation failures. Record raw-array metric recomputation and source provenance before/after. No retry/repair/install/server operations; preserve all failures. Save command/logs/exit/report in fresh D:/具身智能/PhyGile复现/Unitree框架适配_20261022/source_geometry_doc_witness_v1. Wait <=60sec each, <=15min total. Keep raw licensed arrays private.

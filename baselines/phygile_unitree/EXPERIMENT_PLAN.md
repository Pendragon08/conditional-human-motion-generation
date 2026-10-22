# PhyGile Unitree framework adaptation

Authorization: user requested unitree_rl_lab adaptation, then unitree_rl_mjlab
cross-simulation, preserving ReactiveBFM and all data/checkpoints/records.

1. Pin both official repositories and preserve their licenses/source manifests.
2. Register a Unitree RL Lab custom task derived from its G1 mimic environment.
   Carry the PhyGile scene, observations, commands, rewards, terminations, MoE,
   empirical normalization and PPO extensions into that task. Test semantic
   equality against the original configuration; don't silently use a default MLP.
3. Verify the parent checkpoint's SHA and exact policy/normalizer/optimizer load.
   Validate an actual rendered simulator window and local viewing connection
   before a bounded, GUI-enabled continuation into a fresh directory.
4. Export an explicit full-precision joint/action/PD/observation contract. Record
   source and runtime versions, before/after checkpoint hashes and Adam counts.
5. In an isolated local environment, instantiate the official Unitree Mjlab G1
   tracking task. Add PhyGile future-command observations and load the same frozen
   policy/normalizer with an explicit joint permutation. Check equal synthetic
   states numerically, then evaluate fixed held-out clips without optimizer steps.
6. Keep engine/asset differences visible; report failures and raw tracking errors,
   not a self-normalized score. Use same initial reference states, action rate,
   step timing and first-episode termination contract across engines where possible.
7. Run an independent documented witness and an independent integrity audit.
   Publish code, permitted generated results/checkpoints and report to GitHub;
   licensed motion/model files remain on authorized local/server storage.

No stopping/restarting ReactiveBFM. No shared package upgrades. No deletion.
No real G1 deployment. No scheduled automation or external notifications.

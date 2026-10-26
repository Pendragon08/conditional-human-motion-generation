# Independent CPU software renderer document witness v2

Executed the current EXECUTION_VIDEO_ENVIRONMENT.md PowerShell invocation exactly once, without alterations or retries. Actual command and supervisor exits were0; stderr was empty. No source, configuration, dependency, trainer or controller changes were made by this witness.

- Status: SOFTWARE_RENDERER_SYNTHETIC_WITNESS_PASSED;24 frames encoded and decoded,50fps.
- GL: Mesa/X.org / llvmpipe (LLVM 12.0.0, 256 bits); no libnvidia loaded.
- Exactly1 world plane checked in the COMPILED model; existing plane reused across2 XML worldbody sections.
- Original16 training file hashes and live quick-controller hashes unchanged.
- SDK calls, physics integration, optimizer updates and training signals: all0 for this witness.
- Actual-policy-result-video flag isfalse. This is a synthetic renderer fixture, not a tracking/generation experiment or closed-loop result.

Visual inspection of the saved first_frame.png confirms a recognizable G1 robot, clear RENDERER TEST / SYNTHETIC POSES - NOT AN EXPERIMENT RESULT labels, a clean checker ground and resolved previous mottling/z-fighting. Distinct light shadows are present; they are not coplanar ground corruption. Source inspection confirms the actual-mode caption now explicitly says RECORDED ISAACLAB, alongside pose replay/no MuJoCo simulation wording. No remaining document-vs-reality divergence was found in this bounded scope.

MP4: 68512 bytes; SHA256 b082dfd55601956dc22f2684f2edf0d2332f8e71cec01d8d6b987e40f0035845.
First PNG SHA256: 69aa0082bb23acddb573025115870d06d8c91860fcad2a6a110700515a6b3376.
Receipt SHA256: 471be8f47ae024c1f6575e4e4a78d3a529d651a6eff7bc9fe17aa2685bc68f11.
Exit SHA256: 0bf7cf99b71773ffe32e1eb6a6493bec0ed961d0bddee4c6916b0e91f9efb5ba.

Actual experiment videos still require genuine5000 handoff and closed pilot trajectories; they have not been produced by this witness. No GPU/SDK task was launched or interrupted.

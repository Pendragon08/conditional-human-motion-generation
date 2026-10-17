# Pilot input attribution

The pilot uses 24 G1 kinematic reference clips derived in an earlier local preprocessing pipeline from [LuluCao/KungfuAthleteBot](https://huggingface.co/datasets/LuluCao/KungfuAthleteBot), source revision `ff09f3795dc99bef73b21413b03fa8470917a22c`.

The dataset card at that revision declares Apache-2.0. Its exact retrieved text is retained in `KungfuAthleteBot_README_at_source.md`; the license text is in `Apache-2.0.txt`. Original authors' attribution and source URLs are retained in `results/data_provenance.json`.

The distributed pilot subset is transformed data: selected source prefixes, normalized input root quaternions, G1 joint-name remapping, interpolation/finite-difference velocities and Isaac forward kinematics. Each resulting clip has 149 frames at 50 Hz, named joints/bodies and Isaac scalar-first quaternions. They are **kinematic references**, not measured robot rollouts. The hashes identify the existing processed files exactly; these files are not the PhyGile authors' benchmark data.

The input archive is provided as an experiment release attachment, outside the Git source history. It contains the processed references and manifests, not source videos, SMPL model assets, Isaac Sim binaries or G1 meshes. Robot assets retain their separate upstream distribution terms.

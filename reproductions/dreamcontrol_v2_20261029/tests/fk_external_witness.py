# Append this file to src/g1_fk.py and execute with the existing server's
# pytorch_kinematics environment. The path is passed as the first argument.
import json
import sys
import torch
import pytorch_kinematics as pk

path = Path(sys.argv[1])
model = URDFFK(path)
xml = ET.parse(path).getroot()
# MuJoCo-specific metadata is not part of kinematics and the URDF parser in
# pytorch_kinematics does not support that extension tag.
for item in xml.findall("mujoco"):
    xml.remove(item)
chain = pk.build_chain_from_urdf(ET.tostring(xml, encoding="unicode")).to(dtype=torch.float64, device="cpu")
names = chain.get_joint_parameter_names()
rng = np.random.default_rng(42)
values = rng.uniform(-0.3, 0.3, size=(4, len(names)))
ours = model.forward(np.zeros((4, 3)), np.tile([0, 0, 0, 1], (4, 1)), values, names)
ref = chain.forward_kinematics(torch.from_numpy(values))
expected = np.stack([ref[name].get_matrix()[:, :3, 3].detach().cpu().numpy() for name in PAPER_22_LINKS], axis=1)
error = float(np.max(np.abs(ours - expected)))
tolerance_m = 1e-6
result = {"frames": 4, "joint_count": len(names), "keypoints": len(PAPER_22_LINKS),
          "max_abs_error_m": error, "tolerance_m": tolerance_m,
          "reference": "pytorch_kinematics CPU", "passed": error < tolerance_m}
print(json.dumps(result))
if not result["passed"]:
    raise SystemExit(1)

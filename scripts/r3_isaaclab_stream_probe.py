"""Isaac Lab/WebRTC physics preflight; no learning or metrics are produced."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="GPU physics and WebRTC preflight")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.livestream != 2:
    raise SystemExit("This preflight requires --livestream 2")

# Kit consumes these extra settings; argparse must run first.
sys.argv.extend(
    [
        "--/app/livestream/publicEndpointAddress=10.147.17.32",
        "--/app/livestream/port=49100",
    ]
)
launcher = AppLauncher(args)
app = launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport


sim = SimulationContext(sim_utils.SimulationCfg(device=args.device))
sim.set_camera_view(eye=[2.2, 2.2, 1.5], target=[0.0, 0.0, 0.5])
ground = sim_utils.GroundPlaneCfg()
ground.func("/World/Ground", ground)
light = sim_utils.DomeLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0))
light.func("/World/Light", light)

cube_cfg = RigidObjectCfg(
    prim_path="/World/FallingCube",
    spawn=sim_utils.CuboidCfg(
        size=(0.35, 0.35, 0.35),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.6, 1.0)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.4)),
)
cube = RigidObject(cfg=cube_cfg)
sim.reset()
print(f"R3_ISAACLAB_READY device={sim.device}", flush=True)

frame_dir = Path(os.environ["R3_PROJECT_ROOT"]) / "artifacts/videos/r3_isaaclab_probe_frames"
frame_dir.mkdir(parents=True, exist_ok=True)
viewport = get_active_viewport()
print(f"R3_ISAACLAB_VIEWPORT {viewport is not None}", flush=True)

step = 0
while app.is_running():
    if step % 180 == 0:
        pose = cube.data.default_root_state.clone()
        pose[:, 2] = 1.4
        cube.write_root_pose_to_sim(pose[:, :7])
        cube.write_root_velocity_to_sim(torch.zeros_like(pose[:, 7:]))
        cube.reset()
    cube.write_data_to_sim()
    sim.step()
    cube.update(sim.get_physics_dt())
    if viewport is not None and step % 8 == 0 and step <= 480:
        capture_viewport_to_file(viewport, str(frame_dir / f"frame_{step // 8:04d}.png"))
        if step == 480:
            print(f"R3_ISAACLAB_CAPTURE_REQUESTS_DONE {frame_dir}", flush=True)
    step += 1

app.close()

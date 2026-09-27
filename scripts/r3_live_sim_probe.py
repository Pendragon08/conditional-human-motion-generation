"""Small streamed Isaac Sim visual preflight; this is not a training run."""

from __future__ import annotations

import math
import os
from pathlib import Path

from isaacsim import SimulationApp


project_root = Path(os.environ["R3_PROJECT_ROOT"])
isaac_root = Path(os.environ["ISAACSIM_ROOT"])
robot_usd = (
    project_root
    / "third_party/engineai_rl_lab/source/engineai_rl_lab/engineai_rl_lab/assets/pm01/serial_pm01_edu.usd"
)
if not robot_usd.is_file():
    raise FileNotFoundError(robot_usd)

app = SimulationApp(
    {
        "headless": True,
        "hide_ui": False,
        "create_new_stage": False,
        "width": 1280,
        "height": 720,
        "multi_gpu": False,
        "extra_args": [
            "--/app/livestream/publicEndpointAddress=10.147.17.32",
            "--/app/livestream/port=49100",
        ],
    },
    experience=str(isaac_root / "apps/isaacsim.exp.full.streaming.kit"),
)

import numpy as np
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport, get_num_viewports
from pxr import Gf, UsdGeom, UsdLux

app.set_setting("/app/window/drawMouse", True)
app.set_setting("/app/livestream/publicEndpointAddress", "10.147.17.32")
app.set_setting("/app/livestream/port", 49100)
enable_extension("omni.kit.livestream.webrtc")


omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
mount = UsdGeom.Xform.Define(stage, "/World/RobotMount")
robot = stage.DefinePrim("/World/RobotMount/Model", "Xform")
robot.GetReferences().AddReference(str(robot_usd))
if not robot.IsValid():
    raise RuntimeError("Robot USD failed to load")

ground = UsdGeom.Cube.Define(stage, "/World/Ground")
ground.CreateSizeAttr(1.0)
ground.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.06))
ground.AddScaleOp().Set(Gf.Vec3f(5.0, 5.0, 0.1))

dome = UsdLux.DomeLight.Define(stage, "/World/FillLight")
dome.CreateIntensityAttr(1200.0)
dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
key.CreateIntensityAttr(2500.0)
key.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
key.AddRotateXYZOp().Set(Gf.Vec3f(35.0, -30.0, 0.0))

robot_bounds = UsdGeom.BBoxCache(0, ["default", "render", "proxy"]).ComputeWorldBound(robot)
print(f"R3_ROBOT_BOUNDS {robot_bounds.GetRange()}", flush=True)
set_camera_view(eye=np.array([2.4, 2.4, 2.0]), target=np.array([0.0, 0.0, 1.0]))
transform = UsdGeom.XformCommonAPI(mount)
print("R3_LIVE_PROBE_READY", flush=True)

frame = 0
while app.is_running():
    # Motion is deliberately visual only. It confirms live frames reach the
    # local client without claiming a learned policy or physics validation.
    phase = frame * 0.035
    transform.SetRotate((0.0, 0.0, 12.0 * math.sin(phase)))
    transform.SetTranslate((0.0, 0.0, 1.0 + 0.025 * math.sin(phase * 2.0)))
    app.update()
    frame += 1
    if frame == 120:
        viewport = get_active_viewport()
        print(f"R3_VIEWPORTS {get_num_viewports()} ACTIVE {viewport is not None}", flush=True)
        if viewport is not None:
            screenshot = project_root / "artifacts/monitoring/r3_robot_live_probe.png"
            capture_viewport_to_file(viewport, str(screenshot))
            print(f"R3_CAPTURE_REQUESTED {screenshot}", flush=True)

app.close()

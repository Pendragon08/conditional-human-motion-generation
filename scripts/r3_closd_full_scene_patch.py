"""Opt-in CLoSD/Isaac Gym full-scene browser viewer integration.

Only inserts a guarded hook before CLoSD constructs its ground and actors.
Existing headless runs remain unchanged.
"""

from pathlib import Path
import sys

source = Path(sys.argv[1])
text = source.read_text(encoding="utf-8")
marker = '        self._create_ground_plane()\n'
hook = '''        if __import__("os").environ.get("R3_CLOSD_WEB_FULL_SCENE") == "1":
            from sim_web_visualizer.isaac_visualizer_client import (
                create_isaac_visualizer,
                bind_visualizer_to_gym,
                set_gpu_pipeline,
            )
            scene_viewer = create_isaac_visualizer(
                port="6000", host="127.0.0.1", keep_default_viewer=False,
                max_env=int(__import__("os").environ.get("R3_CLOSD_WEB_MAX_ENVS", "2")),
            )
            import zmq
            scene_socket = scene_viewer.viz.window.zmq_socket
            scene_socket.setsockopt(zmq.SNDTIMEO, 1000)
            scene_socket.setsockopt(zmq.RCVTIMEO, 1000)
            scene_socket.setsockopt(zmq.LINGER, 0)
            self.gym = bind_visualizer_to_gym(self.gym, self.sim)
            self.gym.add_method("clear_lines", lambda *args, **kwargs: None)
            self.gym.add_method("add_lines", lambda *args, **kwargs: None)
            original_draw = self.gym.methods["draw_viewer"]
            original_color = self.gym.methods["set_rigid_body_color"]
            original_scale = self.gym.methods["set_actor_scale"]
            scene_status = {"enabled": True, "frame": 0}
            def update_color_without_stopping_sim(*args, **kwargs):
                if scene_status["enabled"]:
                    try:
                        return original_color(*args, **kwargs)
                    except Exception as exc:
                        scene_status["enabled"] = False
                        print("R3_CLOSD_SCENE_MONITOR_FAILED_SIM_CONTINUES:", repr(exc), flush=True)
                        return
                return scene_viewer.original_gym.set_rigid_body_color(*args, **kwargs)
            def update_scale_without_stopping_sim(*args, **kwargs):
                if scene_status["enabled"]:
                    try:
                        return original_scale(*args, **kwargs)
                    except Exception as exc:
                        scene_status["enabled"] = False
                        print("R3_CLOSD_SCENE_MONITOR_FAILED_SIM_CONTINUES:", repr(exc), flush=True)
                        return
                return scene_viewer.original_gym.set_actor_scale(*args, **kwargs)
            def draw_scene_without_stopping_sim(*args, **kwargs):
                if not scene_status["enabled"]:
                    return
                scene_status["frame"] += 1
                if scene_status["frame"] % max(1, int(__import__("os").environ.get("R3_CLOSD_WEB_RENDER_EVERY", "4"))):
                    return
                try:
                    original_draw(*args, **kwargs)
                except Exception as exc:
                    scene_status["enabled"] = False
                    print("R3_CLOSD_SCENE_MONITOR_FAILED_SIM_CONTINUES:", repr(exc), flush=True)
            self.gym.add_method("draw_viewer", draw_scene_without_stopping_sim)
            self.gym.add_method("poll_viewer_events", lambda *args, **kwargs: None)
            self.gym.add_method("set_rigid_body_color", update_color_without_stopping_sim)
            self.gym.add_method("set_actor_scale", update_scale_without_stopping_sim)
            set_gpu_pipeline(self.sim_params.use_gpu_pipeline)
            print("R3_CLOSD_FULL_SCENE_READY", flush=True)

'''
if "R3_CLOSD_FULL_SCENE_READY" in text:
    print("already installed")
    if 'self.gym.add_method("clear_lines"' not in text:
        updated = text.replace(
            '            self.gym = bind_visualizer_to_gym(self.gym, self.sim)\n',
            '            self.gym = bind_visualizer_to_gym(self.gym, self.sim)\n'
            '            self.gym.add_method("clear_lines", lambda *args, **kwargs: None)\n'
            '            self.gym.add_method("add_lines", lambda *args, **kwargs: None)\n',
        )
        if updated == text:
            raise SystemExit("Could not install Isaac Gym line adapter")
        source.write_text(updated, encoding="utf-8")
        print("installed line adapter")
        text = updated
    if "R3_CLOSD_SCENE_MONITOR_FAILED_SIM_CONTINUES" not in text:
        old = '''            create_isaac_visualizer(
                port="6000", host="127.0.0.1", keep_default_viewer=False,
                max_env=int(__import__("os").environ.get("R3_CLOSD_WEB_MAX_ENVS", "2")),
            )
            self.gym = bind_visualizer_to_gym(self.gym, self.sim)
            self.gym.add_method("clear_lines", lambda *args, **kwargs: None)
            self.gym.add_method("add_lines", lambda *args, **kwargs: None)
'''
        new = hook[hook.index('            scene_viewer = '):hook.index('            set_gpu_pipeline(')]
        updated = text.replace(old, new)
        if updated == text:
            raise SystemExit("Could not install fail-open scene adapter")
        source.write_text(updated, encoding="utf-8")
        print("installed fail-open scene adapter")
        text = updated
    if "update_color_without_stopping_sim" not in text:
        old = '''            original_draw = self.gym.methods["draw_viewer"]
            scene_status = {"enabled": True, "frame": 0}
'''
        new = hook[hook.index('            original_draw = '):hook.index('            def draw_scene_without_stopping_sim')]
        updated = text.replace(old, new)
        if updated == text:
            raise SystemExit("Could not install fail-open color adapter")
        updated = updated.replace(
            '            self.gym.add_method("poll_viewer_events", lambda *args, **kwargs: None)\n',
            '            self.gym.add_method("poll_viewer_events", lambda *args, **kwargs: None)\n'
            '            self.gym.add_method("set_rigid_body_color", update_color_without_stopping_sim)\n'
            '            self.gym.add_method("set_actor_scale", update_scale_without_stopping_sim)\n',
        )
        source.write_text(updated, encoding="utf-8")
        print("installed fail-open color adapter")
else:
    if text.count(marker) != 1:
        raise SystemExit(f"Expected exactly one insertion point, found {text.count(marker)}")
    source.with_suffix(source.suffix + ".r3_full_scene_backup").write_text(text, encoding="utf-8")
    source.write_text(text.replace(marker, hook + marker), encoding="utf-8")
    print("installed")

camera_source = source.with_name("humanoid_amp.py")
camera_text = camera_source.read_text(encoding="utf-8")
camera_marker = "    def _update_camera(self):\n"
camera_hook = '''    def _update_camera(self):
        if __import__("os").environ.get("R3_CLOSD_WEB_FULL_SCENE") == "1":
            return  # Meshcat owns its browser camera; no native Isaac Gym Viewer exists.
'''
if "Meshcat owns its browser camera" not in camera_text:
    if camera_text.count(camera_marker) != 1:
        raise SystemExit("Expected one CLoSD camera hook")
    camera_source.with_suffix(camera_source.suffix + ".r3_full_scene_backup").write_text(
        camera_text, encoding="utf-8"
    )
    camera_source.write_text(
        camera_text.replace(camera_marker, camera_hook), encoding="utf-8"
    )
    print("installed camera adapter")

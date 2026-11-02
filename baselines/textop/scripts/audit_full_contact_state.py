"""Read-only cached/native PhysX state comparisons before first rollout."""
import json
import pathlib
import torch


def audit(env, private_dir, output):
    robot = env.unwrapped.scene['robot']
    view = robot.root_physx_view
    original = torch.load(pathlib.Path(private_dir) / 'initial_state.private.pth', map_location='cpu', weights_only=True)
    alerts = json.loads((pathlib.Path(private_dir) / 'first_state_alert.private.json').read_text())

    def stats(x):
        x = x.detach().cpu()
        assert torch.isfinite(x).all()
        return {'min': float(x.min()), 'max': float(x.max()), 'abs_max': float(x.abs().max())}

    trans = view.get_root_transforms().clone()
    trans = torch.cat([trans[:, :3], trans[:, 6:7], trans[:, 3:6]], dim=1)
    dof_pos = view.get_dof_positions().clone()
    dof_vel = view.get_dof_velocities().clone()
    native_root = torch.cat([trans, view.get_root_velocities().clone()], dim=1)
    mass = view.get_masses().clone()
    inertia = view.get_inertias().clone().reshape(-1, 3, 3)
    result = {'cached_vs_original_root': stats(robot.data.root_state_w.cpu() - original['root_state_w']),
              'cached_vs_original_joint_position': stats(robot.data.joint_pos.cpu() - original['joint_pos']),
              'cached_vs_original_joint_velocity': stats(robot.data.joint_vel.cpu() - original['joint_vel']),
              'native_vs_cached_root': stats(native_root - robot.data.root_state_w),
              'native_vs_cached_joint_position': stats(dof_pos - robot.data.joint_pos),
              'native_vs_cached_joint_velocity': stats(dof_vel - robot.data.joint_vel),
              'mass': stats(mass), 'inertia_eigenvalues': stats(torch.linalg.eigvalsh(inertia)),
              'maximum_mass_difference_between_clones': float((mass - mass[:1]).abs().max()),
              'maximum_inertia_difference_between_clones': float((inertia.reshape(8192, -1) - inertia.reshape(8192, -1)[:1]).abs().max()),
              'first_action_match_checked': False}
    ids = [x['environment'] for x in alerts]
    directory = pathlib.Path(output) / 'full_contact_private'
    directory.mkdir()
    import numpy as np
    np.savez(directory / 'selected_initial.private.npz',
             body_state=robot.data.body_state_w[ids].detach().cpu().numpy(),
             native_root=native_root[ids].detach().cpu().numpy(),
             native_joint_position=dof_pos[ids].detach().cpu().numpy(),
             native_joint_velocity=dof_vel[ids].detach().cpu().numpy())
    original_process = env.unwrapped.action_manager.process_action
    def process(actions):
        if not result['first_action_match_checked']:
            expected = torch.tensor([x['action'] for x in alerts], device=actions.device)
            result['selected_first_action_difference'] = stats(actions[ids] - expected)
            result['first_action_match_checked'] = True
        return original_process(actions)
    env.unwrapped.action_manager.process_action = process
    return result

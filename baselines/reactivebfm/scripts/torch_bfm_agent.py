"""Native PyTorch ScaleBFM adapter; official weights/math, no TensorRT artifact.

Supplementary implementation under CC-BY-NC-SA-4.0. Upstream network, task
adapter, robot constants and tracking stack retain their original notices.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys
import torch

UPSTREAM = Path(__file__).resolve().parents[1] / 'upstream'
for p in [UPSTREAM, UPSTREAM / 'deploy']:
    sys.path.insert(0, str(p))
from scalebridge.agent.base_agent import BaseAgent
from scalebridge.agent.bfm_agent import BFMAgent
from scalebridge.agent.qpos_adapter import ScaleBFMQposTaskAdapter


class TorchBFMAgent(BFMAgent):
    def __init__(self, config, device):
        # Bypass only BFMAgent's TensorRT/CUDA loader and CUDA-only assertion.
        BaseAgent.__init__(self, config, device)
        self.control_mode = torch.tensor([config.control_mode], dtype=torch.long, device=device)

    def _load_policy(self):
        checkpoint = Path(self.checkpoint)
        h = hashlib.sha256()
        with checkpoint.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        expected = '88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3'
        if h.hexdigest() != expected:
            raise ValueError('Expected pinned official ScaleBFM M checkpoint')
        spec = importlib.util.spec_from_file_location('native_scalebfm_export', UPSTREAM / 'scripts/export_scalebfm_tensorrt.py')
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        device = torch.device(self.device)
        state = m._state_dict(checkpoint)
        arch = m._infer_architecture(state, None)
        actor, task = m._build_models(state, arch, device, m._load_network_module())
        names, _, parents, translations, rotations, axes, selected, mapping = m._parse_kinematics(m.DEFAULT_XML, device)
        default, scale, stiffness, damping, effort = m._robot_parameters()
        table = m._mode_table(device); mappings = m._mode_mappings(table)
        self.policy = m.ScaleBFMTransformerCore(actor, task,
            torch.tensor([default], dtype=torch.float32, device=device),
            torch.tensor([scale], dtype=torch.float32, device=device), 3).eval()
        self.meta_data_dict = {
            'schema_version': 2, 'input_contract': 'reactivebfm_qpos_v1',
            'engine_input_contract': 'scalebfm_transformer_core_v1',
            'execution_backend': 'native_pytorch_supplementary',
            'source_checkpoint_sha256': expected, 'policy_architecture': arch,
            'selected_body_names': m.SELECTED_BODY_NAMES, 'body_names': names,
            'joint_names': m.POLICY_JOINT_NAMES, 'action_names': m.POLICY_JOINT_NAMES,
            'target_joint_names': m.POLICY_JOINT_NAMES, 'history_buffer_size': 3,
            'future_idx': m.FUTURE_IDX, 'mode_feature_dims': m.MODE_FEATURE_DIMS,
            'mode_mapping_with_time': True,
            'qpos_adapter': {'mode_table': table.cpu().tolist(), 'mode_mappings': mappings.cpu().tolist(),
                'parents': parents.cpu().tolist(), 'translations': translations.cpu().tolist(),
                'rotations': rotations.cpu().tolist(), 'axes': axes.cpu().tolist(),
                'selected': selected.cpu().tolist(), 'policy_to_xml': mapping.cpu().tolist()},
            'default_dof_pos': default, 'action_scale': scale, 'stiffness': stiffness,
            'damping': damping, 'torque_limit': effort}
        self.qpos_adapter = ScaleBFMQposTaskAdapter.from_metadata(self.meta_data_dict, device)

    # get_action/get_meta_data are inherited unchanged from official BFMAgent.

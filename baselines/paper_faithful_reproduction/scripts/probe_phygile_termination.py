"""Add read-only pre-reset termination observations to the fixed Mjlab runner.

Run three selected existing pairs with identical source/seed/policy and compare
every actual state with its preserved baseline. No changes to control or rules.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

p = argparse.ArgumentParser(add_help=False)
p.add_argument('--base-scripts', type=Path, required=True)
p.add_argument('--trace-output', type=Path, required=True)
a, remaining = p.parse_known_args()
checkout = Path(remaining[remaining.index('--unitree-checkout')+1])
sys.path.insert(0, str(checkout.resolve()))
sys.path.insert(0, str(a.base_scripts.resolve()))
import mjlab_task
import source_provenance

capture = source_provenance.capture
def bound_capture(output, roots, inputs=()):
    return capture(output, roots, [*inputs, __file__])
source_provenance.capture = bound_capture
original = mjlab_task.PhyGileMjlabEnv


class TracedEnv(original):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._read_only_trace = []
        compute = self.termination_manager.compute
        def traced_compute():
            result = compute()
            if int(self.episode_length_buf[0]) > 0:
                motion = self.command_manager.get_term('motion')
                names = ['left_ankle_roll_link', 'right_ankle_roll_link',
                         'left_wrist_yaw_link', 'right_wrist_yaw_link']
                ids = [motion.cfg.body_names.index(n) for n in names]
                self._read_only_trace.append({
                    'frame': int(motion.time_steps[0]),
                    'terms': {n: bool(self.termination_manager.get_term(n)[0])
                              for n in self.termination_manager.active_terms},
                    'anchor_z_error_m': float((motion.anchor_pos_w[0, 2]-motion.robot_anchor_pos_w[0, 2]).abs()),
                    'reference_anchor_quat_wxyz': motion.anchor_quat_w[0].detach().cpu().tolist(),
                    'robot_anchor_quat_wxyz': motion.robot_anchor_quat_w[0].detach().cpu().tolist(),
                    'end_effector_z_errors_m': dict(zip(names,
                        (motion.body_pos_relative_w[0, ids, 2]-motion.robot_body_pos_w[0, ids, 2]).abs().detach().cpu().tolist()))})
            return result
        self.termination_manager.compute = traced_compute

    def close(self):
        if hasattr(self, '_read_only_trace') and not getattr(self, '_trace_written', False):
            data = {'kind': 'actual_pre_reset_termination_observations',
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'control_or_termination_rules_changed': False,
                'records': self._read_only_trace}
            with a.trace_output.open('x', encoding='utf-8') as stream:
                json.dump(data, stream, indent=2); stream.write('\n')
            self._trace_written = True
        super().close()


mjlab_task.PhyGileMjlabEnv = TracedEnv
sys.argv = [str(a.base_scripts/'mjlab_evaluate.py'), *remaining]
runpy.run_path(sys.argv[0], run_name='__main__')

"""Materialize an opt-in repaired copy, preserving the archived upstream bytes.

These are independent integrity repairs, not author-confirmed algorithm changes.
Unsafe evaluators are disabled, not represented as repaired benchmark protocols.
"""
from pathlib import Path
import argparse
import difflib
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
MODULE = 'source/whole_body_tracking/whole_body_tracking/'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Pinned source no longer matches: {old[:90]}')
    return text.replace(old, new, 1)


def patched_sources():
    output = {}
    name = MODULE + 'tasks/tracking/mdp/commands.py'
    text = (ROOT/'upstream'/name).read_text(encoding='utf-8')
    text = replace_once(text, '        errors = torch.nan_to_num(errors, nan=0.0, posinf=0.0, neginf=0.0)',
        '        # Independent repair: invalid states must not become perfect tracking.\n'
        '        if not all(bool(torch.isfinite(x).all()) for x in (errors, mpjpe, mpjae)):\n'
        '            raise FloatingPointError("Nonfinite tracking error; stop before sampler statistics")\n'
        '        errors = torch.nan_to_num(errors, nan=0.0, posinf=0.0, neginf=0.0)')
    text = replace_once(text, '        clip_lengths = torch.clamp(self.clip_step_count[env_ids], min=1.0)',
        '        if not all(bool(torch.isfinite(x[env_ids]).all()) for x in (self.clip_error_accum,\n'
        '                self.clip_mpjpe_accum, self.clip_mpjae_accum, self.clip_step_count)):\n'
        '            raise FloatingPointError("Nonfinite accumulated clip statistics")\n'
        '        clip_lengths = torch.clamp(self.clip_step_count[env_ids], min=1.0)')
    start = text.index('        if self._curriculum is None:', text.index('    def run_curriculum_evaluation('))
    text = text[:start] + ('        raise RuntimeError("Unsafe in-process evaluator disabled: double frame advance and post-reset sampling. "\n'
                          '                           "Use scripts/evaluate_prefix_feedback.py for bounded pre-reset diagnostics.")\n') + text[start:]
    output[name] = text
    name = MODULE + 'utils/my_on_policy_runner.py'
    text = (ROOT/'upstream'/name).read_text(encoding='utf-8')
    # This public scheduler samples after automatic reset; blocking is deliberate.
    start = text.index('        if not eval_files:', text.index('    def _run_curriculum_evaluation_subprocess('))
    text = text[:start] + ('        raise RuntimeError("Unverified public subprocess evaluator disabled by integrity repairs. "\n'
                          '                           "A complete pre-reset curriculum evaluator remains required.")\n') + text[start:]
    text = replace_once(text, '        success_ratio = 0.0\n        \n        if use_subprocess',
        '        success_ratio = float("nan")\n        min_success_ratio = float("nan")\n'
        '        min_success_file = ""\n        \n        if use_subprocess')
    text = replace_once(text, '            success_ratio = 1.0  # subprocess 模式不检查成功率',
        '            success_ratio = float("nan")  # Unknown, never fabricated as 100%.')
    text = replace_once(text, '            self.writer.add_scalar("Curriculum/eval_success_ratio", success_ratio, iteration)',
        '            if math.isfinite(success_ratio):\n'
        '                self.writer.add_scalar("Curriculum/eval_success_ratio", success_ratio, iteration)')
    text = 'import math\n' + text if 'from __future__' not in text else text
    if 'from __future__' in text:
        text = replace_once(text, 'from __future__ import annotations', 'from __future__ import annotations\nimport math')
    text = replace_once(text, '        passed = condition1 or condition2',
        '        metrics_valid = not bad_files and all(math.isfinite(x) and x >= 0 for x in (mpjpe, mpjae))\n'
        '        passed = metrics_valid and (condition1 or condition2)')
    text = replace_once(text, '        force_due = timeout_due or convergence_due',
        '        force_due = metrics_valid and (timeout_due or convergence_due)')
    text = replace_once(text, '        valid_mask = (mpjpe_values > 1e-6) & (mpjae_values > 1e-6)',
        '        valid_mask = torch.isfinite(mpjpe_values) & torch.isfinite(mpjae_values) & (mpjpe_values >= 0) & (mpjae_values >= 0)\n'
        '        # Missing/insufficient files cannot silently disappear from the denominator.\n'
        '        minimum = max(1, int(getattr(motion_term.cfg, "curriculum_min_attempts", 5)))\n'
        '        if len(file_indices) != len(eval_files) or success_ratio_per_file is None or not bool(valid_mask.all()):\n'
        '            return float("inf"), float("inf"), mean_success_ratio, min_success_ratio, min_success_file\n'
        '        if not bool(((success + failed) >= minimum).all()) or not bool(torch.isfinite(success + failed).all()):\n'
        '            return float("inf"), float("inf"), mean_success_ratio, min_success_ratio, min_success_file')
    text = replace_once(text,
        '        is_converged = imp_mpjpe < self._metric_conv_threshold and imp_mpjae < self._metric_conv_threshold',
        '        valid = all(math.isfinite(x) and x >= 0 for pair in window for x in pair)\n'
        '        changes = [(a[k] - b[k]) / max(a[k], 1e-4)\n'
        '                   for a, b in zip(window, window[1:]) for k in (0, 1)]\n'
        '        # Independent conservative interpretation: every consecutive eval is stable\n'
        '        # and non-worsening. Endpoint cancellation/negative improvement is not convergence.\n'
        '        is_converged = valid and all(0 <= x < self._metric_conv_threshold for x in changes)')
    output[name] = text
    name = MODULE + 'utils/curriculum_manager.py'
    text = (ROOT/'upstream'/name).read_text(encoding='utf-8')
    text = replace_once(text, '        passed_by_metric = mpjpe < self.cfg.mpjpe_threshold or mpjae < self.cfg.mpjae_threshold',
        '        import math\n'
        '        if not all(math.isfinite(x) and x >= 0 for x in (mpjpe, mpjae)):\n'
        '            return False  # Even timeout must not promote invalid evaluation.\n'
        '        passed_by_metric = mpjpe < self.cfg.mpjpe_threshold or mpjae < self.cfg.mpjae_threshold')
    output[name] = text
    return output


def materialize(destination):
    dest = Path(destination).resolve()
    if dest.exists():
        raise FileExistsError(dest)
    if dest == ROOT/'upstream' or (ROOT/'upstream') in dest.parents:
        raise ValueError('Never modify the archived upstream snapshot')
    sources = patched_sources()
    shutil.copytree(ROOT/'upstream', dest, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    records = {}
    patch = []
    for name, text in sources.items():
        original = (ROOT/'upstream'/name).read_bytes()
        (dest/name).write_bytes(text.encode('utf-8'))
        records[name] = {'original_sha256': hashlib.sha256(original).hexdigest(),
                         'patched_sha256': hashlib.sha256((dest/name).read_bytes()).hexdigest()}
        patch.extend(difflib.unified_diff(original.decode().splitlines(True), text.splitlines(True),
                                        fromfile='a/'+name, tofile='b/'+name))
    (dest/'INDEPENDENT_REPAIRS.patch').write_bytes(''.join(patch).encode('utf-8'))
    receipt = {'status': 'MATERIALIZED_NOT_TRAINED', 'destination': str(dest),
               'scope': 'Integrity repairs; unsafe evaluators disabled; full curriculum not validated',
               'sources': records, 'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (dest/'INDEPENDENT_REPAIRS.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', required=True)
    print(json.dumps(materialize(parser.parse_args().destination), indent=2))

"""Read-only, hash-bound preparation for the scheduled SuperMimic task adapter.

Uses only Python's standard library. Does not import the framework, load arrays,
initialize CUDA, alter TextOp, or change the running Pick training process.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def scalar(node, names):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        return names[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return scalar(node.left, names) + scalar(node.right, names)
    raise ValueError('unsupported scalar expression; inspect changed source')


def assigned(tree, name, names):
    for item in tree.body:
        if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in item.targets):
            return scalar(item.value, names)
    raise ValueError('missing assignment '+name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    source = Path(plan['existing_textop_environment']['source_root'])
    receipt_path = source/'source_receipt.json'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(receipt_path) == plan['existing_textop_environment']['source_receipt_sha256']
    receipt = json.loads(receipt_path.read_text())
    expected = {item['path']: item['sha256'] for item in receipt['files']}
    selected = ['dimensions.py', 'mdp.py', 'networks.py', 'env.py', 'train.py', 'buffer.py']
    sources = {}
    trees = {}
    texts = {}
    for name in selected:
        path = source/'supermimic'/name
        digest = sha(path)
        assert digest == expected['supermimic/'+name], 'source changed: '+name
        sources[name] = {'path': str(path), 'sha256': digest}
        texts[name] = path.read_text()
        trees[name] = ast.parse(texts[name])

    names = {key: assigned(trees['dimensions.py'], key, {}) for key in ['STATE_DIM', 'XI_DIM']}
    names['NUM_J'] = assigned(trees['mdp.py'], 'NUM_J', {})
    obs = assigned(trees['mdp.py'], 'OBS_DIM', names)
    inv = assigned(trees['mdp.py'], 'INV_DIM', names)
    wm_out = assigned(trees['networks.py'], 'WM_OUT', names)
    assert (names['STATE_DIM'], names['NUM_J'], obs, inv, wm_out) == (71, 29, 99, 71, 70)

    defaults = {}
    for node in ast.walk(trees['train.py']):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            if node.args and isinstance(node.args[0], ast.Constant):
                for kw in node.keywords:
                    if kw.arg == 'default':
                        try:
                            defaults[node.args[0].value] = ast.literal_eval(kw.value)
                        except (ValueError, TypeError):
                            pass
    assert 'step_acc += 1.0 / args.utd' in texts['train.py']
    assert 'collect(max(args.warmup_steps, args.wm_window), stats)' in texts['train.py']
    matched = plan['matched_configuration']
    target = matched['total_vectorized_control_steps_per_env']
    warmup = max(defaults['--warmup_steps'], defaults['--wm_window'])
    total, counter, iterations = warmup, 0., 0
    at_3000 = None
    final_native_collect, final_capped_collect = None, None
    while total < target:
        counter += 1./defaults['--utd']
        n = int(counter)
        counter -= n
        final_native_collect, final_capped_collect = n, min(n, target-total)
        total += n
        iterations += 1
        if iterations == 3000:
            at_3000 = total
    assert warmup == 64 and at_3000 == 7564 and iterations == 28775 and total == 72001

    # Byte counts for the installed native replay fields, not measured peak VRAM.
    fields = next(n.value for n in trees['buffer.py'].body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'FIELDS' for t in n.targets))
    dtype_bytes = {'float32': 4, 'long': 8, 'int32': 4, 'bool': 1}
    layout = []
    for key, value in zip(fields.keys, fields.values):
        shape, dtype = value.elts
        width = 1 if isinstance(shape, ast.Tuple) and not shape.elts else scalar(shape, names)
        layout.append({'name': key.value, 'width': width, 'bytes': width*dtype_bytes[dtype.attr]})
    row_bytes = sum(x['bytes'] for x in layout)
    assert row_bytes == 949
    envs = matched['num_envs']
    native_rows = defaults['--buffer_steps']
    report = {
        'status': 'static_preparation_only_not_task_validation',
        'source_commit': receipt['commit'], 'source_binding_passed': True, 'sources': sources,
        'native_interface': {'body_actions': names['NUM_J'], 'flat_state': names['STATE_DIM'],
            'single_frame_actor_observation': obs, 'history_frames': defaults['--hist'],
            'future_reference_frames': defaults['--goal_frames'], 'world_model_output': wm_out},
        'required_pick_interface': {'body_dof': 29, 'binary_hands': 2, 'actions': 31,
            'actor_observation': 102, 'actor_reference_input': False,
            'object_and_physical_hand_dynamics': 'not represented by the native71 state; must be added'},
        'native_budget_example_not_a_launch_config': {
            'utd': defaults['--utd'], 'warmup_vector_steps': warmup,
            'vector_steps_for_native3000_iterations': at_3000,
            'matched_target_vector_steps': target,
            'minimum_native_iterations_to_reach_target_at_default_utd': iterations,
            'uncapped_steps_at_that_iteration': total,
            'last_native_collect_steps': final_native_collect,
            'last_collect_steps_needed_for_exact_budget': final_capped_collect,
            'required_change': 'cap actual collection at exact remaining budget, including warmup; track optimizer updates separately'},
        'replay_allocation_lower_bound_only': {
            'native_field_layout': layout, 'native_bytes_per_transition': row_bytes,
            'native1000_rows_at8912_envs_bytes': native_rows*envs*row_bytes,
            'candidate112_state31_action_with_same_other_fields_bytes_per_transition': row_bytes+2*(112-71)*4+2*(31-29)*4,
            'candidate1000_rows_bytes': native_rows*envs*(row_bytes+2*(112-71)*4+2*(31-29)*4),
            'candidate_layout_unvalidated': True,
            'scope': 'excludes simulator, models/Adam, gradients, sampled windows and imagined-rollout activations; not a safe peak bound'},
        'remaining': ['reference-free actor and live object pose adaptation',
            '31 actions with binary hands and contact-aware state transitions',
            'same scene/rewards/reset and differentiable imagined-task reward',
            'time/joint/asset and simulator-backend alignment',
            'exact interaction-budget cap and bounded replay allocation',
            'real task simulation/world-model/policy update plus checkpoint round trip after PPO evaluation'],
        'framework_imports': False, 'gpu_used': False, 'policy_updates': 0,
        'training_launched': False, 'task_ready_for_training': False,
    }
    args.output_dir.mkdir(exist_ok=False)
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

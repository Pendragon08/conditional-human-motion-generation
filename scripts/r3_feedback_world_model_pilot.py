"""Bounded real-physics data / learned-dynamics ablation, no PPO or extra IK.

Prediction on replay is deliberately separate from control improvement. Raw
state/reference arrays and fitted weights are kept only in ignored artifacts.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn

from r3_sim2sim_mujoco import Transfer, rotation, multiply, render_frame, VideoWriter, sha

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / 'configs/feedback_world_model_pilot_20261006.json'
FITCFG = ROOT / 'configs/feedback_world_model_fit_v2_20261006.json'
ART = ROOT / 'artifacts/feedback_world_model_20261006'
OUT = ROOT / 'results/feedback_world_model_20261006'
DT = .02


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def state(t):
    q = t.data.qpos[3:7].copy()
    if q[0] < 0:
        q *= -1
    return np.r_[t.data.qpos[:3], q, t.data.qvel[:6],
                 t.data.qpos[t.qadr], t.data.qvel[t.vadr]].astype(np.float32)


def normalize_quat(s):
    out = s.clone()
    q = out[..., 3:7]
    q = q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-6)
    q = torch.where(q[..., :1] < 0, -q, q)
    # Concatenate rather than in-place mutation of a differentiable view.
    return torch.cat([out[..., :3], q, out[..., 7:]], dim=-1)


def witness():
    torch.manual_seed(42)
    reports = []
    for task in ('Kick', 'Punch'):
        t = Transfer(task, 'cuda:0')
        parity = t.verify()
        assert parity['passed']
        t.reset(0)
        initial = state(t)
        for k in range(30):
            t.advance(t.action(t.obs(0, k*DT)))
        final = state(t)
        assert final.shape == (67,) and np.isfinite(final).all()
        assert np.max(np.abs(final-initial)) > .001
        renderer = mujoco.Renderer(t.model, height=480, width=640)
        frame = render_frame(renderer, t, t.ref.state(0, 0)['goal'])
        assert frame.shape == (480, 640, 3) and frame.std() > 1
        renderer.close()
        model = nn.Sequential(nn.Linear(98, 32), nn.SiLU(), nn.Linear(32, 67)).cuda()
        prediction = model(torch.randn(8, 98, device='cuda'))
        prediction.square().mean().backward()
        assert torch.isfinite(prediction).all()
        reports.append({'task': task, 'parity': parity, 'moving_state': True,
                        'cuda_forward_backward': True, 'nonempty_egl_frame': True,
                        'provenance': t.provenance})
    report = {'status': 'PASS', 'config_sha256': sha(CFG), 'reports': reports,
              'no_formal_rl_started': True, 'no_package_changes': True}
    write(OUT / 'environment_witness.json', report)
    print('FEEDBACK_WORLD_MODEL_WITNESS_PASS', json.dumps(report), flush=True)


def collect(probe=False):
    c = json.loads(CFG.read_text())
    private = ART / ('probe' if probe else 'data')
    private.mkdir(parents=True, exist_ok=True)
    if list(private.glob('*.npz')):
        raise FileExistsError('Existing collection must be preserved; no duplicate collection')
    records = []
    start = time.time()
    for ti, task in enumerate(c['tasks']):
        t = Transfer(task, 'cuda:0')
        assert t.verify()['passed']
        nrefs = 5 if probe else 66
        conditions = ['nominal', 'dither_123'] if probe else c['collection_conditions'] + [c['heldout_condition']]
        for cond in conditions:
            for i in range(nrefs):
                split = next(name for name, (lo, hi) in c['reference_splits'].items() if lo <= i < hi)
                if cond == c['heldout_condition'] and split != 'test':
                    continue
                seed = int(cond.split('_')[-1]) if cond.startswith('dither') else 123
                rng = np.random.default_rng(seed + i*1009 + ti*100003)
                t.reset(i)
                states, actions, forces = [state(t)], [], []
                squared_errors, endpoint_errors = [], []
                fallen = False
                for k in range(c['episode_steps']):
                    action = t.action(t.obs(i, k*DT))
                    if cond.startswith('dither'):
                        action[:27] += np.clip(rng.normal(0, c['action_dither_std'], 27),
                                              -c['action_dither_clip'], c['action_dither_clip'])
                    force = np.zeros(3)
                    if cond == c['heldout_condition'] and 50 <= k < 60:
                        force[1] = 80
                    t.data.xfrc_applied[t.robot_body, :3] = force
                    actions.append(action.copy())
                    forces.append(force.copy())
                    t.advance(action)
                    s = state(t)
                    assert np.isfinite(s).all()
                    states.append(s)
                    ref = t.ref.state(i, (k+1)*DT)
                    squared_errors.append(float(np.mean((s[13:40]-ref['joints'])**2)))
                    endpoint_errors.append(float(np.linalg.norm(t.point()-ref['goal'])))
                    if t.fallen():
                        fallen = True
                        break
                name = f'{task}_{i:03d}_{cond}.npz'
                np.savez_compressed(private/name, states=np.asarray(states), actions=np.asarray(actions),
                                    external_forces=np.asarray(forces), task=ti, reference=i)
                record = {'task': task, 'task_index': ti, 'reference': i, 'split': split,
                          'condition': cond, 'file': name, 'transitions': len(actions), 'fallen': fallen,
                          'duration_s': len(actions)*DT,
                          'joint_tracking_rmse_rad': float(np.sqrt(np.mean(squared_errors))),
                          'minimum_fixed_goal_error_m': float(min(endpoint_errors)),
                          'states_sha256': sha(private/name)}
                records.append(record)
                if len(records) % 20 == 0:
                    print('COLLECT', len(records), 'transitions', sum(r['transitions'] for r in records),
                          'wall_s', round(time.time()-start, 2), flush=True)
    stats = {split: sum(r['transitions'] for r in records
                        if r['split'] == split and r['condition'] != c['heldout_condition'])
             for split in c['reference_splits']}
    report = {'status': 'COMPLETE', 'module_only': True, 'config_sha256': sha(CFG),
              'wall_s': time.time()-start, 'episodes': len(records), 'transitions': sum(stats.values()),
              'transitions_by_split': stats, 'unseen_push_test_transitions': sum(r['transitions'] for r in records
                  if r['condition'] == c['heldout_condition']), 'records': records,
              'policies_frozen': True, 'no_extra_ik': True,
              'external_force_not_given_to_world_model': True,
              'force_steps_50_to_59_in_unseen_push': True}
    write(OUT / ('throughput_probe.json' if probe else 'collection.json'), report)
    print('COLLECTION_COMPLETE', {k: v for k, v in report.items() if k != 'records'}, flush=True)


def load_episodes():
    report = json.loads((OUT/'collection.json').read_text())
    eps = []
    for r in report['records']:
        a = dict(np.load(ART/'data'/r['file']))
        eps.append({**r, 's': a['states'].astype(np.float32), 'a': a['actions'].astype(np.float32)})
    return eps, report


def windows(eps, split, horizon, stride=1, push=False, metadata=False):
    ss, aa, tasks = [], [], []
    labels = []
    for e in eps:
        if e['split'] != split or (e['condition'].startswith('push') != push):
            continue
        for k in range(0, len(e['a'])-horizon+1, stride):
            ss.append(e['s'][k:k+horizon+1])
            aa.append(e['a'][k:k+horizon])
            tasks.append(e['task_index'])
            k_end = k+horizon
            phase = ('active_force' if 50 <= k < 60 else
                     'post_force_recovery_60_149' if 60 <= k < 150 else
                     'late_recovery' if k >= 150 else
                     'force_onset_crossing' if k_end > 50 else 'pre_force')
            labels.append((e['task_index'], e['reference'], e['condition'], phase if push else 'nominal_and_dither'))
    result = (np.asarray(ss), np.asarray(aa), np.asarray(tasks))
    return result+(labels,) if metadata else result


class Dynamics(nn.Module):
    def __init__(self, norm, width=256, state_only=False, task_free=False):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(98, width), nn.SiLU(), nn.Linear(width, width),
                                 nn.SiLU(), nn.Linear(width, 67))
        self.state_only = state_only
        self.task_free = task_free
        for k, v in norm.items():
            self.register_buffer(k, torch.tensor(v, dtype=torch.float32))

    def features(self, s, a, task):
        s = torch.cat([torch.zeros_like(s[..., :2]), s[..., 2:]], -1)
        an = (a-self.am)/self.ast
        # Only these 27 outputs drive the same-step physical PD. The final
        # two hand commands are not applied by this frozen transfer adapter.
        an = torch.cat([an[..., :27], torch.zeros_like(an[..., 27:])], -1)
        if self.state_only:
            an = torch.zeros_like(an)
        bits = torch.nn.functional.one_hot(task, 2).to(s.dtype)
        if self.task_free:
            bits = torch.zeros_like(bits)
        return torch.cat([(s-self.sm)/self.sst, an, bits], -1)

    def forward(self, s, a, task):
        d = self.net(self.features(s, a, task))*self.dst+self.dm
        return normalize_quat(s+d)


def state_metric(p, y):
    delta = p-y
    dot = np.abs(np.sum(p[:, 3:7]*y[:, 3:7], axis=-1))
    return {'root_xyz_rmse_m': float(np.sqrt(np.mean(delta[:, :3]**2))),
            'joint_position_rmse_rad': float(np.sqrt(np.mean(delta[:, 13:40]**2))),
            'root_velocity_rmse': float(np.sqrt(np.mean(delta[:, 7:13]**2))),
            'joint_velocity_rmse_rad_s': float(np.sqrt(np.mean(delta[:, 40:]**2))),
            'orientation_geodesic_mean_rad': float(np.mean(2*np.arccos(np.clip(dot, 0, 1))))}


def constant_velocity(s):
    pred = s.copy()
    pred[:, :3] += DT*s[:, 7:10]
    pred[:, 13:40] += DT*s[:, 40:]
    for i in range(len(s)):
        w = s[i, 10:13]
        angle = np.linalg.norm(w)*DT
        dq = np.r_[np.cos(angle/2), w*DT*.5*np.sinc(angle/(2*np.pi))]
        pred[i, 3:7] = multiply(s[i, 3:7], dq)
    pred[:, 3:7] /= np.linalg.norm(pred[:, 3:7], axis=-1, keepdims=True)
    pred[:, 3:7] *= np.where(pred[:, 3:4] < 0, -1, 1)
    return pred


def fit_witness():
    torch.manual_seed(42)
    raw = np.load(ART/'probe/Kick_000_nominal.npz')
    ss = np.stack([raw['states'][k:k+6] for k in range(8)])
    aa = np.stack([raw['actions'][k:k+5] for k in range(8)])
    x = ss[:,0].copy()
    x[:,:2] = 0
    d = ss[:,1]-ss[:,0]
    norm = {'sm':x.mean(0),'sst':np.maximum(x.std(0),.01),
            'am':aa[:,0].mean(0),'ast':np.maximum(aa[:,0].std(0),.01),
            'dm':d.mean(0),'dst':np.maximum(d.std(0),.001)}
    s,a = torch.tensor(ss,device='cuda'),torch.tensor(aa,device='cuda')
    task = torch.zeros(8,dtype=torch.long,device='cuda')
    reports=[]
    for state_only,task_free in ((False,False),(True,False),(False,True)):
        model = Dynamics(norm,256,state_only,task_free).cuda()
        features = model.features(s[:,0],a[:,0],task)
        assert torch.all(features[:,67+27:67+29]==0)
        prediction=s[:,0]
        loss=0
        for k in range(5):
            prediction=model(prediction,a[:,k],task)
            loss+=(((prediction-s[:,k+1])/model.dst)**2).mean()/5
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters())
        reports.append({'state_only':state_only,'task_free':task_free,'loss':loss.item(),
                        'parameters':sum(p.numel() for p in model.parameters())})
    write(OUT/'fit_kernel_witness.json',{'status':'PASS','causal_actions27':True,
          'fit_protocol_sha256':sha(FITCFG),'reports':reports,'no_saved_trained_model':True})
    print('FIT_KERNEL_WITNESS_PASS',json.dumps(reports),flush=True)


def fit():
    c = json.loads(CFG.read_text())
    fitting = json.loads(FITCFG.read_text())
    kernel_witness = json.loads((OUT/'fit_kernel_witness.json').read_text())
    assert kernel_witness['status']=='PASS' and kernel_witness['fit_protocol_sha256']==sha(FITCFG), 'Fresh fit-kernel witness required'
    c['models'] = fitting['models']
    eps, collection = load_episodes()
    device = 'cuda:0'
    train_s, train_a, train_t = windows(eps, 'train', c['training_window'], stride=2)
    val_s, val_a, val_t = windows(eps, 'validation', c['training_window'], stride=10)
    x = train_s[:, 0].copy()
    x[:, :2] = 0
    delta = train_s[:, 1]-train_s[:, 0]
    norm = {'sm': x.mean(0), 'sst': np.maximum(x.std(0), .01),
            'am': train_a[:, 0].mean(0), 'ast': np.maximum(train_a[:, 0].std(0), .01),
            'dm': delta.mean(0), 'dst': np.maximum(delta.std(0), .001)}
    ts, ta, tt = [torch.tensor(x, device=device) for x in (train_s, train_a, train_t)]
    vs, va, vt = [torch.tensor(x, device=device) for x in (val_s, val_a, val_t)]
    # Linear ridge is fitted with exactly the same training-window start rows.
    shell = Dynamics(norm, c['hidden_width']).to(device)
    with torch.no_grad():
        features = shell.features(ts[:, 0], ta[:, 0], tt)
        features = torch.cat([features, torch.ones(len(features), 1, device=device)], -1).double()
        target = ((ts[:, 1]-ts[:, 0]-shell.dm)/shell.dst).double()
        ridge = torch.linalg.solve(features.T@features + c['ridge_lambda']*torch.eye(99, device=device),
                                   features.T@target).float()
    private = ART/'models'
    private.mkdir(parents=True, exist_ok=True)
    if list(private.glob('*.pt')):
        raise FileExistsError('Fitted models already exist; do not train a duplicate pilot')
    torch.save({'norm': norm, 'ridge': ridge.cpu()}, private/'ridge.pt')
    models = {}
    curve = []
    training_start = time.time()
    for arm in c['models']:
        for seed in c['training_seeds']:
            torch.manual_seed(seed)
            model = Dynamics(norm, c['hidden_width'], arm.startswith('state_only'),
                             arm.startswith('mlp_notask')).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=c['learning_rate'])
            generator = torch.Generator(device=device).manual_seed(seed)
            best, best_state = float('inf'), None
            for update in range(1, c['updates']+1):
                ids = torch.randint(len(ts), (c['batch_size'],), generator=generator, device=device)
                s, a, task = ts[ids], ta[ids], tt[ids]
                prediction = s[:, 0]
                loss = 0
                for k in range(c['training_window']):
                    initial = s[:, k] if arm == 'mlp_single' else prediction
                    prediction = model(initial, a[:, k], task)
                    loss = loss + (((prediction-s[:, k+1])/model.dst)**2).mean()/c['training_window']
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step()
                if update % c['validation_every'] == 0:
                    with torch.no_grad():
                        prediction = vs[:, 0]
                        validation = 0
                        for k in range(c['training_window']):
                            prediction = model(prediction, va[:, k], vt)
                            validation += (((prediction-vs[:, k+1])/model.dst)**2).mean().item()/c['training_window']
                    if validation < best:
                        best = validation
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    row = {'arm': arm, 'seed': seed, 'update': update,
                           'train_loss': loss.item(), 'validation_rollout_loss': validation}
                    curve.append(row)
                    print('FIT', json.dumps(row), flush=True)
            model.load_state_dict(best_state)
            model.eval()
            torch.save({'state_dict': best_state, 'norm': norm, 'state_only': model.state_only,
                        'task_free': model.task_free,
                        'width': c['hidden_width'], 'best_validation_loss': best}, private/f'{arm}_{seed}.pt')
            models[(arm, seed)] = model
    write(OUT/'training_curves.json', curve)
    training_wall = time.time()-training_start
    eval_start = time.time()
    rows = []
    reference_rows = []
    fk_transfer = Transfer('Kick', device)
    fk_data = mujoco.MjData(fk_transfer.model)
    fk_ids = [mujoco.mj_name2id(fk_transfer.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in
        ('pelvis','left_hip_pitch_link','right_hip_pitch_link','left_knee_link','right_knee_link',
         'left_ankle_roll_link','right_ankle_roll_link','left_shoulder_roll_link','right_shoulder_roll_link',
         'left_elbow_link','right_elbow_link','left_wrist_yaw_link','right_wrist_yaw_link')]

    def forward_kinematics(states):
        positions = []
        for s in states:
            mujoco.mj_resetData(fk_transfer.model, fk_data)
            fk_data.qpos[:3], fk_data.qpos[3:7] = s[:3], s[3:7]
            fk_data.qpos[fk_transfer.qadr] = s[13:40]
            mujoco.mj_forward(fk_transfer.model, fk_data)
            positions.append(fk_data.xpos[fk_ids].copy())
        return np.asarray(positions)
    for push in (False, True):
        for horizon in c['evaluation_horizons']:
            s, a, task, labels = windows(eps, 'test', horizon, c['evaluation_stride'], push, True)
            st, at, taskt = [torch.tensor(x, device=device) for x in (s, a, task)]
            # No autoregressive action recomputation: this is recorded-action replay prediction.
            predictions = {'persistence': s[:, 0].copy(), 'constant_velocity': s[:, 0].copy(),
                           'ridge': s[:, 0].copy()}
            for k in range(horizon):
                predictions['constant_velocity'] = constant_velocity(predictions['constant_velocity'])
                with torch.no_grad():
                    curr = torch.tensor(predictions['ridge'], device=device)
                    f = shell.features(curr, at[:, k], taskt)
                    f = torch.cat([f, torch.ones(len(f), 1, device=device)], -1)
                    curr = normalize_quat(curr+(f@ridge)*shell.dst+shell.dm)
                    predictions['ridge'] = curr.cpu().numpy()
            neural_predictions = {}
            with torch.no_grad():
                for (arm, seed), model in models.items():
                    pred = st[:, 0]
                    for k in range(horizon):
                        pred = model(pred, at[:, k], taskt)
                    neural_predictions[(arm, seed)] = pred.cpu().numpy()
                for arm in c['models']:
                    predictions[arm+'_ensemble'] = np.mean([neural_predictions[(arm, seed)]
                        for seed in c['training_seeds']], axis=0)
                    predictions[arm+'_ensemble'][:, 3:7] /= np.linalg.norm(
                        predictions[arm+'_ensemble'][:, 3:7], axis=-1, keepdims=True)
            predictions.update({f'{arm}_seed{seed}': p for (arm, seed), p in neural_predictions.items()})
            true_fk = forward_kinematics(s[:, -1])
            for name, pred in predictions.items():
                assert np.isfinite(pred).all()
                fk_squared = np.mean((forward_kinematics(pred)-true_fk)**2,axis=(1,2))
                groups = {}
                for wi,label in enumerate(labels):
                    groups.setdefault(label,[]).append(wi)
                for (ti,reference,condition,phase), ids in groups.items():
                    reference_rows.append({'model':name,'task':c['tasks'][ti],
                        'reference':reference,'condition':condition,'phase':phase,'horizon_steps':horizon,
                        'windows':len(ids),'body_site_fk_rmse_m':float(np.sqrt(np.mean(fk_squared[ids])))})
                for ti, task_name in enumerate(c['tasks']):
                    mask = task == ti
                    rows.append({'model': name, 'task': task_name, 'horizon_steps': horizon,
                                 'condition_set': 'unseen_push' if push else 'nominal_and_dither',
                                 'windows': int(mask.sum()),
                                 'body_site_fk_rmse_m':float(np.sqrt(np.mean(fk_squared[mask]))),
                                 **state_metric(pred[mask], s[mask, -1])})
    write(OUT/'prediction_metrics.json', rows)
    write(OUT/'reference_metrics.json', reference_rows)
    primary = [r for r in reference_rows if r['horizon_steps']==10 and r['phase']=='nominal_and_dither']
    keys = sorted({(r['task'],r['reference']) for r in primary})
    means = {}
    for model_name in sorted({r['model'] for r in primary}):
        means[model_name] = np.asarray([np.mean([r['body_site_fk_rmse_m'] for r in primary
            if r['model']==model_name and (r['task'],r['reference'])==key]) for key in keys])
    neural = np.mean([means[f'mlp_multi5_seed{seed}'] for seed in c['training_seeds']],axis=0)
    state_only = np.mean([means[f'state_only_multi5_seed{seed}'] for seed in c['training_seeds']],axis=0)
    simple = [means[name] for name in ('persistence','constant_velocity','ridge')]
    best = min(np.mean(m) for m in simple)
    improvement = 1-float(neural.mean()/best)
    action_improvement = 1-float(neural.mean()/state_only.mean())
    rng = np.random.default_rng(fitting['bootstrap_seed'])
    trials, action_trials = [], []
    # Balance tasks while clustering every condition/window within a reference.
    task_groups = [np.asarray([i for i,k in enumerate(keys) if k[0]==task_name]) for task_name in c['tasks']]
    for _ in range(fitting['bootstrap_repetitions']):
        ids = np.concatenate([rng.choice(group,len(group),replace=True) for group in task_groups])
        simple_best = min(np.mean(m[ids]) for m in simple)
        trials.append(1-float(neural[ids].mean()/simple_best))
        action_trials.append(1-float(neural[ids].mean()/state_only[ids].mean()))
    ci = np.quantile(trials,[.025,.975]).tolist()
    action_ci = np.quantile(action_trials,[.025,.975]).tolist()
    n_train = collection['transitions_by_split']['train']
    data_pass = n_train >= c['gates']['minimum_training_transitions_for_predictive_gate']
    primary_pass = improvement >= fitting['minimum_relative_H10_improvement'] and ci[0]>0
    action_pass = action_ci[0]>0
    predictive_gate = {'status':'PASS' if data_pass and primary_pass and action_pass else 'FAIL',
        'primary':'H10 reference-macro body-site FK RMSE, nominal+dither test',
        'test_reference_clusters':len(keys),'training_transitions':n_train,'data_gate_pass':data_pass,
        'macro_means_m':{k:float(v.mean()) for k,v in means.items()},
        'relative_improvement_vs_best_simple':improvement,'reference_bootstrap_95ci':ci,
        'relative_improvement_vs_state_only':action_improvement,'action_bootstrap_95ci':action_ci,
        'primary_improvement_gate_pass':primary_pass,'action_input_gate_pass':action_pass,
        'ensemble_not_used_for_gate':True,'control_benefit_gate_pass':False,
        'full_innovation_validated':False}
    write(OUT/'predictive_gate.json',predictive_gate)
    summary = {'status': 'COMPLETE', 'config_sha256': sha(CFG), 'fit_protocol_sha256':sha(FITCFG), 'module_only': True,
               'training_transition_count': collection['transitions_by_split']['train'],
               'training_windows': len(ts), 'validation_windows': len(vs),
               'same_data_and_batch_schedule_for_neural_arms': True,
               'same_parameter_count_for_state_only': True, 'training_wall_s': training_wall,
               'evaluation_wall_s': time.time()-eval_start,
               'checkpoint_hashes': {p.name: sha(p) for p in sorted(private.glob('*.pt'))},
               'no_policy_update': True, 'no_diffusion_feedback_executed': True,
               'control_benefit_evaluated': False,
               'prediction_is_replay_action_conditioned_not_on_policy_planning': True,
               'minimum_data_gate_pass': collection['transitions_by_split']['train'] >= c['gates']['minimum_training_transitions_for_predictive_gate'],
               'predictive_gate':predictive_gate,
               'full_innovation_validated': False}
    write(OUT/'summary.json', summary)
    print('WORLD_MODEL_PREDICTION_COMPLETE', json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['witness', 'fit_witness', 'probe', 'collect', 'fit'], required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'witness':
        witness()
    elif args.stage == 'fit_witness':
        fit_witness()
    elif args.stage in ('probe', 'collect'):
        collect(args.stage == 'probe')
    else:
        fit()


if __name__ == '__main__':
    main()

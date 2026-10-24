"""Independent pre-reset evaluation service and single curriculum decision rule.

Policies are reconstructed in child processes. No EMA estimate, skipped file,
post-reset sample or unknown success rate is accepted as a completed assessment.
Threshold AND/success gates are conservative implementation choices, not an
author-confirmed reproduction of their inconsistent public scheduler rules.
"""
from pathlib import Path
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(request, records):
    requested=[str(Path(p).resolve()) for p in request['files']]
    if not requested or len(set(requested))!=len(requested):
        raise ValueError('Nonempty unique files required')
    if request['attempts_per_file']<1:
        raise ValueError('Positive number of attempts required')
    errors=[]; by_file={}
    for r in records:
        name=str(Path(r['motion_file']).resolve())
        if name in by_file:raise ValueError('Duplicate evaluation record')
        by_file[name]=r
    if set(by_file)!=set(requested):errors.append('missing_or_extra_files')
    per_file=[]
    for name in requested:
        if name not in by_file:continue
        r=by_file[name]; problems=[]
        if r['checkpoint_sha256']!=request['checkpoint_sha256']:problems.append('checkpoint_mismatch')
        if r['active_experts']!=request['active_experts']:problems.append('active_expert_mask_mismatch')
        if r['motion_sha256']!=sha(name):problems.append('motion_changed')
        if r['episodes_completed']!=request['attempts_per_file'] or r['num_envs']!=request['attempts_per_file']:
            problems.append('attempt_count_mismatch')
        if len(r['episode_lengths'])!=request['attempts_per_file'] or min(r['episode_lengths'],default=0)<=0:
            problems.append('invalid_episode_lengths')
        if not 0<=r['successes']<=request['attempts_per_file']:problems.append('invalid_success_count')
        metric=r['metrics']; p=metric['mpjpe_m_until_termination']; a=metric['mpjae_rad_until_termination']
        if not all(math.isfinite(x) and x>=0 for x in (p,a)):problems.append('invalid_metrics')
        per_file.append({'motion_file':name,'motion_sha256':r['motion_sha256'],
            'attempts':r['episodes_completed'],'successes':r['successes'],
            'success_fraction':r['successes']/request['attempts_per_file'],
            'mpjpe_m_until_termination':p,'mpjae_rad_until_termination':a,
            'episode_lengths':r['episode_lengths'],'problems':problems})
        errors.extend(name+':'+e for e in problems)
    valid=not errors and len(per_file)==len(requested)
    return {'status':'COMPLETED' if valid else 'INVALID','valid':valid,'problems':errors,
        'evaluation_type':'simulation_only','reference_type':'dataset-derived kinematic reference',
        'aggregation':'equal weight per requested file; errors censored at termination; success counts full clips',
        'requested_files':requested,'attempts_per_file':request['attempts_per_file'],
        'active_experts':request['active_experts'],
        'checkpoint_sha256':request['checkpoint_sha256'],'per_file':per_file,
        'mpjpe_m':sum(r['mpjpe_m_until_termination'] for r in per_file)/len(per_file) if valid else None,
        'mpjae_rad':sum(r['mpjae_rad_until_termination'] for r in per_file)/len(per_file) if valid else None,
        'success_fraction':sum(r['success_fraction'] for r in per_file)/len(per_file) if valid else None,
        'minimum_file_success_fraction':min((r['success_fraction'] for r in per_file),default=None) if valid else None}


@dataclass(frozen=True)
class PromotionPolicy:
    mpjpe_m:float=.12
    mpjae_rad:float=.06
    minimum_success:float=.1
    minimum_iterations:int=3000
    convergence_evaluations:int=3
    convergence_fraction:float=.03
    timeout_iterations:int=0
    convergence_enabled:bool=True

    def __post_init__(self):
        if not all(math.isfinite(x) and x>0 for x in (self.mpjpe_m,self.mpjae_rad,self.convergence_fraction)):
            raise ValueError('Invalid thresholds')
        if not 0<=self.minimum_success<=1 or self.minimum_iterations<0 or self.convergence_evaluations<1 or self.timeout_iterations<0:
            raise ValueError('Invalid promotion policy')

    def decide(self, report, history, elapsed):
        reason=[]
        if not report.get('valid'):return {'promote':False,'reason':'invalid_evaluation'}
        values=(report['mpjpe_m'],report['mpjae_rad'],report['minimum_file_success_fraction'])
        if not all(x is not None and math.isfinite(x) and x>=0 for x in values):
            return {'promote':False,'reason':'invalid_metrics'}
        if not 0<=values[2]<=1:return {'promote':False,'reason':'invalid_success'}
        if values[2]<self.minimum_success:return {'promote':False,'reason':'minimum_file_success_below_threshold'}
        if elapsed<self.minimum_iterations:return {'promote':False,'reason':'minimum_training_iterations'}
        if values[0]<self.mpjpe_m and values[1]<self.mpjae_rad:reason.append('both_error_thresholds')
        series=list(history)+[values[:2]]
        n=self.convergence_evaluations
        if self.convergence_enabled and len(series)>=n+1:
            tail=series[-n-1:]
            valid=all(math.isfinite(x) and x>=0 for pair in tail for x in pair)
            changes=[(a[k]-b[k])/max(a[k],1e-8) for a,b in zip(tail,tail[1:]) for k in (0,1)]
            if valid and all(0<=x<self.convergence_fraction for x in changes):reason.append('non_worsening_convergence')
        if self.timeout_iterations and elapsed>=self.timeout_iterations:reason.append('configured_timeout')
        return {'promote':bool(reason),'reason':'+'.join(reason) if reason else 'error_thresholds_not_met'}


def evaluate_files(request, output, root):
    output=Path(output);root=Path(root).resolve()
    output.mkdir(parents=True,exist_ok=False)
    if sha(request['checkpoint'])!=request['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    (output/'request.json').write_text(json.dumps(request,indent=2)+'\n')
    records=[]; started=time.time()
    for i,name in enumerate(request['files']):
        run=output/f'file_{i:04d}'
        command=['bash',str(root/'scripts/run_isaac.sh'),str(root/'cache/closed_loop_evaluate.py'),
            '--checkpoint',request['checkpoint'],'--motion_file',str(Path(name).resolve()),
            '--output',str(run),'--num_envs',str(request['attempts_per_file']),
            '--seed',str(request['seed']+i),'--headless','--device',request['device'],
            '--g1_training_cfg',request['training_config'],
            '--active_experts',str(request['active_experts'])]
        with (output/f'file_{i:04d}.log').open('w') as log:
            result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,
                                  timeout=request.get('timeout_per_file_s',600))
        if result.returncode:raise RuntimeError(f'Evaluation failed: {name}, code {result.returncode}')
        records.append(json.loads((run/'metrics.json').read_text()))
    report=aggregate(request,records)
    report.update(started_unix=started,finished_unix=time.time(),source_sha256=sha(__file__),
        worker_sha256=sha(root/'cache/closed_loop_evaluate.py'),
        request_sha256=sha(output/'request.json'),
        raw_files=[{'path':str((output/f'file_{i:04d}/raw_pre_reset.npz').resolve()),
                    'sha256':sha(output/f'file_{i:04d}/raw_pre_reset.npz')} for i in range(len(records))])
    (output/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def install_runner_hook(runner_class, root):
    """Replace the old EMA/subprocess decision path with one explicit service.

    The parent runner owns learning-rate/expert transitions on the next loop;
    this hook only advances the manager using the same validated decision.
    No synthetic labels are created. A real curriculum CSV remains required.
    """
    def evaluate(self, iteration):
        motion=self._get_motion_term()
        if motion is None:return
        motion.set_training_iteration(iteration)
        if not motion.should_run_curriculum_evaluation():return
        manager=motion._curriculum
        if manager is None:return
        level=motion.get_curriculum_level()
        files=motion.get_curriculum_files_for_evaluation(self._get_frozen_files(motion),level)
        if not files:raise ValueError('Empty requested curriculum level')
        if not self.log_dir:raise ValueError('Independent curriculum evaluator requires a persistent log directory')
        destination=Path(self.log_dir)/'independent_curriculum'/f'iteration_{iteration:08d}'
        destination.mkdir(parents=True,exist_ok=False)
        checkpoint=destination/'policy_snapshot.pt'
        self.save(str(checkpoint))
        cfg=motion.cfg
        import os
        request={'checkpoint':str(checkpoint.resolve()),'checkpoint_sha256':sha(checkpoint),
            'active_experts':int(self.alg.policy._max_active_experts),
            'files':[str(Path(p).resolve()) for p in files],
            'attempts_per_file':int(getattr(manager.cfg,'eval_runs_per_file',10)),
            'seed':int(getattr(self.env.unwrapped.cfg,'seed',2026)),
            'device':str(self.device),'training_config':str(Path(os.environ['G1_TRAINING_CFG_PATH']).resolve())}
        report=evaluate_files(request,destination/'assessment',root)
        timeout=getattr(cfg,'curriculum_auto_upgrade_per_level',None)
        timeout=int(timeout[min(level-1,len(timeout)-1)]) if timeout else int(getattr(cfg,'curriculum_auto_upgrade_after_iters',0))
        policy=PromotionPolicy(float(cfg.curriculum_mpjpe_threshold),float(cfg.curriculum_mpjae_threshold),
            float(getattr(cfg,'curriculum_success_threshold',.1)),
            int(getattr(cfg,'curriculum_metric_convergence_min_iters',3000)),
            int(getattr(cfg,'curriculum_metric_convergence_evals',3)),
            float(getattr(cfg,'curriculum_metric_convergence_threshold',.03)),timeout,
            bool(getattr(cfg,'curriculum_metric_convergence_enabled',True)))
        history=getattr(self,'_independent_curriculum_history',[])
        elapsed=iteration-manager.get_level_unlocked_iter(level)
        decision=policy.decide(report,history,elapsed)
        if report['valid']:history.append((report['mpjpe_m'],report['mpjae_rad']))
        if decision['promote'] and level<manager.cfg.max_difficulty:
            manager.max_level=level+1;manager._level_unlocked_iter[level+1]=iteration
            manager._current_iter=iteration
            motion._curriculum_current_level=manager.max_level
            motion._rebuild_file_list_for_curriculum()
            history=[]
        self._independent_curriculum_history=history
        decision.update(iteration=iteration,old_level=level,new_level=manager.max_level,
                        report_sha256=sha(destination/'assessment/evaluation.json'))
        (destination/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
        print('PHYGILE_INDEPENDENT_CURRICULUM_DECISION',json.dumps(decision),flush=True)
    runner_class._run_curriculum_evaluation_if_needed=evaluate
    runner_class._independent_curriculum_evaluator_installed=True

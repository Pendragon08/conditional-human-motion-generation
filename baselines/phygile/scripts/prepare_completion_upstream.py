"""Opt-in copy of integrity guards plus independent curriculum hook."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'cache/guarded_upstream';dest=ROOT/'cache/closed_loop_upstream'
rel=Path('source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py')
text=(source/rel).read_text()
footer='''
# Independently implemented, explicitly opt-in evaluation service.
if os.environ.get("PHYGILE_INSTALL_CURRICULUM_EVALUATOR") == "1":
    import sys as _phygile_sys
    from pathlib import Path as _PhygilePath
    _phygile_root = _PhygilePath(os.environ["PHYGILE_REPRODUCTION_ROOT"])
    _phygile_sys.path.insert(0, str(_phygile_root / "reimplementation"))
    from curriculum_evaluator import install_runner_hook as _install_hook
    _install_hook(MotionOnPolicyRunner, _phygile_root)
    print("PHYGILE_CURRICULUM_HOOK_ACTIVE", __file__, flush=True)
'''
patched=(text+footer).encode()
if dest.exists():
 if (dest/rel).read_bytes()!=patched:raise FileExistsError('Existing completion copy differs')
else:
 shutil.copytree(source,dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 (dest/rel).write_bytes(patched)
manifest={'original_guarded_runner_sha256':hashlib.sha256((source/rel).read_bytes()).hexdigest(),
 'completion_runner_sha256':hashlib.sha256(patched).hexdigest(),
 'hook_sha256':hashlib.sha256((ROOT/'reimplementation/curriculum_evaluator.py').read_bytes()).hexdigest()}
# Evaluation must reconstruct the parent's active MoE mask. Loading a state_dict
# alone does not restore this Python attribute, so the old worker would silently
# evaluate only the initial expert after a curriculum level transition.
worker=(ROOT/'scripts/evaluate_prefix_feedback.py').read_text()
worker=worker.replace('p=argparse.ArgumentParser()',
    "p=argparse.ArgumentParser()\np.add_argument('--active_experts',type=int,required=True)",1)
old='runner.load(str(checkpoint),load_optimizer=False)'
new=old+'''
if not 1 <= a.active_experts <= runner.alg.policy._num_experts:
 raise ValueError('Requested active expert mask incompatible with policy architecture')
runner.alg.policy.set_max_active_experts(a.active_experts)
runner.alg.policy.set_current_max_level(a.active_experts)
'''
if worker.count(old)!=1:raise ValueError('Pinned evaluation entrypoint changed')
worker=worker.replace(old,new,1)
target=ROOT/'cache/closed_loop_evaluate.py'
if target.exists() and target.read_text()!=worker:
 archive=ROOT/'results/completion_v1/development_versions'/hashlib.sha256(target.read_bytes()).hexdigest()/'closed_loop_evaluate.py'
 archive.parent.mkdir(parents=True,exist_ok=True);archive.write_bytes(target.read_bytes())
target.write_text(worker)
manifest['evaluation_worker_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
(dest/'COMPLETION_HOOK.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('PHYGILE_COMPLETION_COPY_READY',json.dumps(manifest))

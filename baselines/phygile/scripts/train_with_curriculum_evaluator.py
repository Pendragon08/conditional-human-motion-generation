"""Opt-in training entrypoint; preserves archived upstream and earlier patches."""
from pathlib import Path
import runpy,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from guarded_bootstrap import activate
activate()
# Imported by upstream only after AppLauncher initializes Isaac Sim.
import os
os.environ['PHYGILE_INSTALL_CURRICULUM_EVALUATOR']='1'
os.environ['PHYGILE_REPRODUCTION_ROOT']=str(ROOT)
target=ROOT/'cache/closed_loop_upstream/scripts/rsl_rl/train.py'
if not target.exists():raise FileNotFoundError('Run prepare_completion_upstream.py first')
sys.path.insert(0,str(ROOT/'cache/closed_loop_upstream/source/whole_body_tracking'))
sys.path.insert(0,str(target.parent))
sys.argv[0]=str(target)
runpy.run_path(str(target),run_name='__main__')

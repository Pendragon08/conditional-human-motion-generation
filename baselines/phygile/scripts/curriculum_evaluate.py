"""Independent curriculum evaluation entrypoint; no Isaac app in the parent."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reimplementation'))
from curriculum_evaluator import evaluate_files
p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
result=evaluate_files(json.loads(a.request.read_text()),a.output,ROOT)
print('PHYGILE_CURRICULUM_EVALUATION_COMPLETE',json.dumps({k:v for k,v in result.items() if k!='per_file'}))
if not result['valid']:raise SystemExit(2)

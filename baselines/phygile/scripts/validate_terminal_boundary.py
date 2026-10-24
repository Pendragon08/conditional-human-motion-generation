"""Supplemental bounded test: one-second generated window with standing suffix."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--name',choices=['closure_final','closure_recovery'],default='closure_final')
a=p.parse_args()
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/completion_v1'/a.name
if OUT.exists():raise FileExistsError(OUT)
pipeline=json.loads((ROOT/'results/completion_v1/document_run/pipeline.json').read_text())
assert pipeline['status']=='COMPLETED'
command=['bash','scripts/run_isaac.sh','scripts/run_generated_closed_loop.py',
 '--checkpoint','logs/rsl_rl/phygile_moe_pilot/2026-10-17_11-25-01_extended2000_seed42/model_1999.pt',
 '--motion_file','data/train/sample_000_actor0.npz',
 '--normalization','results/completion_v1/document_run/codec/output/normalization.npz',
 '--output',str(OUT/'output'),'--headless','--device','cuda:0',
 '--total_frames','50','--terminal_reference','configs/nominal_standing_reference.json']
code=subprocess.call([sys.executable,'scripts/run_prefix_recorded.py','--name','completion_v1/'+a.name+'/run',
 '--timeout','600','--',*command],cwd=ROOT)
if code:raise SystemExit(code)
report_path=OUT/'output/verification.json';report=json.loads(report_path.read_text())
assert report['status']=='COMPLETED_ENGINEERING_VALIDATION'
assert report['requested_new_frames']==50
assert 'nominal G1 standing' in report['terminal_target_source']
import numpy as np
with np.load(OUT/'output/nominal_standing_target.npz',allow_pickle=False) as d:target={k:d[k] for k in d.files}
assert np.all(target['states'][:,7:13]==0) and np.all(target['states'][:,42:]==0)
for proposal in sorted((OUT/'output/proposals').glob('*.npz')):
 with np.load(proposal,allow_pickle=False) as d:
  np.testing.assert_array_equal(d['candidate_native'][-1:],target['states'])
  np.testing.assert_array_equal(d['candidate_body_positions'][-1:],target['body_positions'])
proof={'status':'COMPLETED','primary_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),
 'nominal_suffix_preserved_in_every_candidate':True,'raw_target_is_not_measured_standing_success':True}
(OUT/'verification.json').write_text(json.dumps(proof,indent=2)+'\n')
print('PHYGILE_TERMINAL_BOUNDARY_COMPLETE',flush=True)

"""Paired four-clip diagnostic, with the measured sliding coefficient only."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--unitree-checkout',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);runs=[]
for profile in ['official','matched_sliding_friction']:
 for clip in range(4):
  out=a.output/(profile+'_clip'+str(clip))
  command=[sys.executable,str(Path(__file__).with_name('mjlab_evaluate.py')),'--unitree-checkout',str(a.unitree_checkout),'--assets',str(a.assets),'--output',str(out),'--clip',str(clip),'--seed','2026','--contact-profile',profile]
  with (a.output/(out.name+'_stdout.txt')).open('x',encoding='utf-8') as so,(a.output/(out.name+'_stderr.txt')).open('x',encoding='utf-8') as se:
   r=subprocess.run(command,stdout=so,stderr=se)
  receipt=json.loads((out/'receipt.json').read_text()) if (out/'receipt.json').exists() else None
  runs.append({'command':command,'exit_code':r.returncode,'receipt':receipt})
  (a.output/'progress.json').write_text(json.dumps(runs,indent=2)+'\n',encoding='utf-8')
  if r.returncode or receipt['status']!='FIRST_EPISODE_EVALUATED':raise RuntimeError('Actual physics diagnostic failed: '+out.name)
  print(json.dumps({'profile':profile,'clip':clip,'frames':receipt['frames_evaluated'],'success':receipt['full_clip_success']}),flush=True)
result={'status':'EIGHT_PAIRED_DIAGNOSTICS_COMPLETED','evaluation_type':'simulation_only','seeds':[2026],'clips_are_development_diagnostics':True,'optimizer_updates':0,'selected_by_test_performance':False,'change':'Measured source sliding coefficient 1.0 replaces official foot coefficient 0.6; other geometry/contact/solver settings retained. PhysX and MuJoCo contacts are not asserted equivalent.','runs':runs}
(a.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')

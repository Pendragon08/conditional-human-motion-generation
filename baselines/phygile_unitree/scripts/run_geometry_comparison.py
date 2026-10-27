from pathlib import Path
import argparse,json,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--base-scripts',required=True);p.add_argument('--unitree-checkout',required=True);p.add_argument('--assets',required=True);p.add_argument('--source-urdf',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);rows=[]
for clip in range(4):
 out=a.output/('clip'+str(clip))
 command=[sys.executable,str(Path(__file__).with_name('evaluate_source_geometry.py')),'--source-urdf',a.source_urdf,'--base-scripts',a.base_scripts,'--unitree-checkout',a.unitree_checkout,'--assets',a.assets,'--output',str(out),'--clip',str(clip),'--contact-profile','matched_sliding_friction']
 with (a.output/('clip'+str(clip)+'_stdout.txt')).open('x',encoding='utf-8') as so,(a.output/('clip'+str(clip)+'_stderr.txt')).open('x',encoding='utf-8') as se:r=subprocess.run(command,stdout=so,stderr=se)
 receipt=json.loads((out/'receipt.json').read_text(encoding='utf-8')) if (out/'receipt.json').exists() else None
 rows.append({'command':command,'exit_code':r.returncode,'receipt':receipt});(a.output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
 if r.returncode or receipt['status']!='FIRST_EPISODE_EVALUATED':raise RuntimeError('Actual geometry-profile run failed')
 print(json.dumps({'clip':clip,'frames':receipt['frames_evaluated'],'full_clip_success':receipt['full_clip_success']}),flush=True)
(a.output/'comparison.json').write_text(json.dumps({'status':'FOUR_GEOMETRY_DIAGNOSTICS_COMPLETED','development_clips':True,'seed':2026,'parameter_sweep_or_best_profile_selection':False,'policy_updates':0,'runs':rows},indent=2)+'\n',encoding='utf-8')

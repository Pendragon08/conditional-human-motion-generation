from pathlib import Path
import subprocess,json,datetime
root=Path(__file__).resolve().parents[1]
out=root/'results/reimplementation_text'
record={'status':'RUNNING','stages':[]}
path=out/'pipeline.json';path.write_text(json.dumps(record,indent=2)+'\n')
for name,script,timeout in [('text_download','fetch_text_encoder.py',900),('text_interface','test_text_encoder.py',120)]:
 result=subprocess.run(['python3','scripts/run_recorded.py','--name','reimplementation_text/'+name,'--timeout',str(timeout),'--','bash','scripts/run_isaac.sh','reimplementation/'+script],cwd=root)
 record['stages'].append({'script':script,'exit_code':result.returncode})
 path.write_text(json.dumps(record,indent=2)+'\n')
 if result.returncode:break
record['status']='COMPLETED' if len(record['stages'])==2 and all(x['exit_code']==0 for x in record['stages']) else 'FAILED'
record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
path.write_text(json.dumps(record,indent=2)+'\n')

"""One durable continuation of the unchanged paper teacher; no retry."""
import json
from session import remote
print(remote('''import json,pathlib,subprocess
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
s=json.loads((r/'configs/full-teacher-resume-spec.json').read_text());assert s['resume_enabled']
run=s['run'];prefix='full_resume_'+run
driver=r/'receipts'/(prefix+'_driver.py');outer=r/'receipts'/(prefix+'_outer_exit.json')
assert not driver.exists() and not outer.exists() and not (r/'tracker_validation'/run).exists()
code="import pathlib,subprocess,json\\nr=pathlib.Path(%r)\\np=subprocess.run(['python3','scripts/supervise_full_teacher_resume.py'],cwd=r,timeout=7*86400+120)\\n(r/'receipts'/%r).write_text(json.dumps({'actual_returncode':p.returncode}))\\n"%(str(r),outer.name)
driver.write_text(code)
with (r/'logs'/(prefix+'_outer.log')).open('xb') as f:
 p=subprocess.Popen(['python3',str(driver)],cwd=r,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'driver_pid':p.pid,'run':run,'driver':str(driver),'outer_exit':str(outer)}))
''',timeout=60))

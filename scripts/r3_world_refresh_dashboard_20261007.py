"""Refresh the independent read-only dashboard; never touch compute/viewer."""
from r3_world_ssh_transport import execute
PROGRAM=r'''
from pathlib import Path
import subprocess,json,shlex,time
r=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation');m=r/'artifacts/monitoring'
marker=m/'r3_world_dashboard_stage_labels_20261007.json'
assert not marker.exists(),'Dashboard already refreshed; inspect rather than repeat'
name='r3_i1_viewport_dashboard'
old=subprocess.run(['tmux','has-session','-t',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
if old:subprocess.run(['tmux','kill-session','-t',name],check=True)
cmd='cd '+shlex.quote(str(r))+' && /usr/bin/python3 -u scripts/r3_i1_viewport_dashboard.py --port 6013 > artifacts/monitoring/r3_world_dashboard_20261007.log 2>&1'
subprocess.run(['tmux','new-session','-d','-s',name,cmd],check=True)
marker.write_text(json.dumps({'changed':'read-only dashboard stage/budget labels','old_dashboard_session':old,'viewer_or_compute_signals':0,'unix':time.time()})+'\n')
print('INDEPENDENT_DASHBOARD_REFRESHED_NO_TRAINING_SIGNALS')
'''
if __name__=='__main__':print(execute(PROGRAM),end='')

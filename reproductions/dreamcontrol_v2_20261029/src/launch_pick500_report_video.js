const {execFileSync}=require('child_process');
const script=String.raw`import json,os,subprocess
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103';O=S/'reference_scale500_20261104'
launch=O/'report_video_v1_launch.json'
assert not launch.exists(),'existing launch receipt: inspect instead of duplicating'
guard=O/'report_video_v1_guard.py'
guard.write_text('''import json,os,subprocess,shutil,time,signal
from pathlib import Path
R=Path("/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029");S=R/"stage3_pick29_20261103";O=S/"reference_scale500_20261104"
spec=json.loads((R/"stage3_supermimic29_20261103/runtime_bridge_spec.json").read_text())
manifest=json.loads((O/"library_progress.json").read_text())["current_partial_manifest"]
out=O/"report_video_v1"
cmd=[spec["python"],"-u",str(S/"src/stage3_pick500_report_video.py"),"--headless","--enable_cameras","--device","cuda:1","--manifest",manifest,"--output-dir",str(out)]
state={"status":"rendering","scope":"reference pose playback, not policy or grasp success","command":cmd}
def save():
 p=O/"report_video_v1_guard.json";t=p.with_suffix(".tmp");t.write_text(json.dumps(state,indent=2));t.replace(p)
assert shutil.disk_usage("/data").free>8*1024**3+456*1024**2+100*1024**2
with (O/"report_video_v1.log").open("x") as log:
 child=subprocess.Popen(cmd,cwd=spec["cwd"],env={**os.environ,**spec["env"]},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 state["child_pid"]=child.pid;began=time.monotonic();save()
 while child.poll() is None:
  rows={}
  for l in subprocess.check_output(["ps","-eo","pid=,ppid=,rss=,args="],text=True).splitlines():
   f=l.strip().split(None,3)
   if len(f)==4:rows[int(f[0])]=(int(f[1]),int(f[2]),f[3])
  own={p for p,v in rows.items() if str(R) in v[2] and any(s in v[2] for s in ["stage3_","isaacsim"]) };own.add(child.pid)
  while True:
   n=len(own);own.update(p for p,v in rows.items() if v[0] in own)
   if n==len(own):break
  rss=sum(rows[p][1] for p in own if p in rows)*1024;gpu=0
  for l in subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,used_gpu_memory","--format=csv,noheader,nounits"],text=True,timeout=10).splitlines():
   pid,mem=[x.strip() for x in l.split(",")]
   if int(pid) in own:
    assert mem.isdigit();gpu+=int(mem)*1024**2
  free=shutil.disk_usage("/data").free
  state.update(elapsed_seconds=time.monotonic()-began,own_gpu_bytes=gpu,own_rss_bytes=rss,free_bytes=free);save()
  if gpu>12*1024**3 or rss>32*1024**3 or free<8*1024**3+456*1024**2 or time.monotonic()-began>900:
   state["status"]="protected_stop_preserved";os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   break
  time.sleep(3)
 code=child.wait();state["returncode"]=code
 if state["status"]=="rendering":state["status"]="render_complete" if code==0 else "failed_preserved"
 save()
''')
with (O/'report_video_guard_launcher.log').open('x') as f:
 p=subprocess.Popen(['nice','-n','10','python3','-u',str(guard)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
launch.open('x').write(json.dumps({'guard_pid':p.pid,'guard':str(guard),'output':str(O/'report_video_v1')},indent=2))
print(launch.read_text())
`;
process.stdout.write(execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -'],{input:script,encoding:'utf8',timeout:30000}));

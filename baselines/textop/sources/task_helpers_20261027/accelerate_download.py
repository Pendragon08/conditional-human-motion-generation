import pathlib,os,signal,json,subprocess,time
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
pid=json.loads((r/'receipts/download_pid.json').read_text())['pid']
cmd=pathlib.Path(f'/proc/{pid}/cmdline')
if cmd.exists():
    assert 'download_assets.py' in cmd.read_text()
    os.kill(pid,signal.SIGTERM)
    time.sleep(1)
(r/'logs/download.log').rename(r/'logs/download_parallel4.log')
env=os.environ.copy();env['TEXTOP_RANGE_WORKERS']='12'
p=subprocess.Popen(['python3',str(r/'scripts/download_assets.py')],cwd=r,env=env,stdout=open(r/'logs/download.log','ab'),stderr=subprocess.STDOUT,start_new_session=True)
(r/'receipts/download_pid.json').write_text(json.dumps({'pid':p.pid,'range_workers':12,'prior_pid':pid}))
print('DOWNLOAD_PID',p.pid)

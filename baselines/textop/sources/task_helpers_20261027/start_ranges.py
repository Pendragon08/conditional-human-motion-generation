import pathlib,os,signal,json,subprocess
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
pid=json.loads((r/'receipts/preparation_pids.json').read_text())['download']
cmd=pathlib.Path(f'/proc/{pid}/cmdline')
if cmd.exists() and 'download_assets.py' in cmd.read_text():os.kill(pid,signal.SIGTERM)
(r/'logs/download.log').rename(r/'logs/download_initial_stream.log')
p=subprocess.Popen(['python3',str(r/'scripts/download_assets.py')],cwd=r,stdout=open(r/'logs/download.log','ab'),stderr=subprocess.STDOUT,start_new_session=True)
(r/'receipts/download_pid.json').write_text(json.dumps({'pid':p.pid,'method':'parallel HTTP ranges with journal and full LFS SHA256 validation'}))
print('DOWNLOAD_PID',p.pid)

import json, os, pathlib, subprocess, shutil, hashlib, datetime
root=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
commands={
 'gpu': ['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,memory.total','--format=csv,noheader,nounits'],
 'gpu_processes': ['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],
 'personal_processes': ['ps','-u',str(os.getuid()),'-o','pid,ppid,stat,lstart,args']
}
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':os.uname().nodename,'uid':os.getuid(),'self_pid':os.getpid()}
for key,cmd in commands.items():
 p=subprocess.run(cmd,text=True,capture_output=True)
 out[key]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
out['data_disk']=dict(zip(['total','used','free'],shutil.disk_usage(root)))
out['meminfo']=pathlib.Path('/proc/meminfo').read_text()
out['document_run_exists']=(root/'tracker_validation/full_library_document_v1').exists()
out['document_exit_exists']=(root/'receipts/full_library_capacity_full_library_document_v1_exit.json').exists()
files=['configs/full-library-capacity-spec.json','scripts/full_library_inputs.py','scripts/full_library_capacity_probe.py','scripts/supervise_full_library_capacity.py','tracker_validation/full_library_author_v1/receipt.json','receipts/full_library_capacity_full_library_author_v1_exit.json','receipts/preprocessing_after_kit_v1.json']
out['file_sha256']={p:hashlib.sha256((root/p).read_bytes()).hexdigest() if (root/p).is_file() else None for p in files}
print(json.dumps(out,indent=2))
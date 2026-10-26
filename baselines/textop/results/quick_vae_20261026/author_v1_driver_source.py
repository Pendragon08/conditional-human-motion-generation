import subprocess,pathlib,json,time
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
z=subprocess.run(['python3',str(r/'scripts/supervise_quick_vae.py'),'--mode','witness','--run','vae_kernel_author_v1'],capture_output=True,text=True,timeout=950)
(r/'receipts/quick_vae_kernel_author_v1_supervisor_execution.json').write_text(json.dumps(dict(exit_code=z.returncode,stdout=z.stdout,stderr=z.stderr,time_unix=time.time()),indent=2))

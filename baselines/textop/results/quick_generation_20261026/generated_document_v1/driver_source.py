import subprocess,pathlib,json,time
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
z=subprocess.run(['python3',str(r/'scripts/supervise_quick_generation.py'),'--run','generated_document_v1'],capture_output=True,text=True,timeout=1050)
(r/'receipts'/('generated_document_v1'+'_supervisor_execution.json')).write_text(json.dumps(dict(exit_code=z.returncode,stdout=z.stdout,stderr=z.stderr,time_unix=time.time()),indent=2))

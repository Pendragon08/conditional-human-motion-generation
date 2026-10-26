import subprocess,pathlib,json,time
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
z=subprocess.run(['python3',str(r/'scripts/supervise_quick69_corpus.py'),'--mode','build','--run','quick_corpus_author_v3'],capture_output=True,text=True,timeout=950)
(r/'receipts/quick69_corpus_author_v3_supervisor_execution.json').write_text(json.dumps(dict(exit_code=z.returncode,stdout=z.stdout,stderr=z.stderr,time_unix=time.time()),indent=2))

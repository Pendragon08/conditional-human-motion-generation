const {execFileSync}=require('child_process');
const script=String.raw`import ast,json,os,subprocess,hashlib,shutil
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103';O=S/'reference_scale500_20261104'
source=S/'src/stage3_pick500_replacement_queue.py';assert source.exists();ast.parse(source.read_text())
receipt=O/'replacement_queue128_v1_launch.json';assert not receipt.exists() and not (O/'replacement_queue128_v1').exists(),'inspect existing queue, do not relaunch'
assert shutil.disk_usage('/data').free>8*1024**3+456*1024**2+100*1024**2
active=json.loads((O/'active_native_controller.json').read_text());proc=Path('/proc')/str(active['pid'])/'cmdline'
assert proc.exists() and b'stage3_pick500_native_drain.py' in proc.read_bytes(),'inspect native continuation first'
registry_path=O/'native_pipeline_registry.json';registry=json.loads(registry_path.read_text())
backup=O/'native_pipeline_registry_before_replacement128_v1.json';backup.open('x').write(registry_path.read_text())
sample=O/'candidate_replacement128_v1';opt=O/'postopt_replacement128_v1'
assert not sample.exists() and not opt.exists()
registry['producer_dirs'].append(str(sample));registry['optimizer_record_paths'].append(str(opt/'records.jsonl'));registry['optimizer_terminal_dirs'].append(str(opt))
registry['planned_replacement_queue']={'source':str(source),'receipt':str(receipt),'candidate_start':500,'candidate_count':128,'starts_only_after_original_producer_and_optimizer_exit_and_candidate_audit':True}
tmp=registry_path.with_suffix('.tmp');tmp.write_text(json.dumps(registry,indent=2));tmp.replace(registry_path)
with (O/'replacement_queue128_v1.log').open('x') as log:
 child=subprocess.Popen(['nice','-n','10','python3','-u',str(source)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
value={'queue_pid':child.pid,'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'status':'sequential_continuation_queued_not_second_sampler_started','candidate_start':500,'candidate_count':128,
 'native_pipeline_registry':str(registry_path),'does_not_modify_existing_sampling_or_optimizer':True,'formal_PPO_started':False}
receipt.open('x').write(json.dumps(value,indent=2));print(json.dumps(value))
`;
process.stdout.write(execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -'],{input:script,encoding:'utf8',timeout:30000}));

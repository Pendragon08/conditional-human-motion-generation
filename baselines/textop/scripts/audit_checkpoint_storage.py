"""CPU-only, read-only size forecast from existing trusted diagnostic weights.

Serialization goes to a counting sink, never to a new checkpoint. No model
execution, update, CUDA allocation or deletion occurs.
"""
import gc,hashlib,json,math,os,pathlib,re,shutil
os.environ['CUDA_VISIBLE_DEVICES']=''
os.sched_setaffinity(0,{0,1,2,3})
import torch
torch.set_num_threads(4)
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
OUT=ROOT/'receipts/checkpoint_storage_20261026.json'
assert not OUT.exists() and os.getuid()==1013

class CountingSink:
 def __init__(self):self.count=0
 def write(self,data):self.count+=len(data);return len(data)
 def flush(self):pass

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()

def serialized_size(value):
 target=CountingSink();torch.save(value,target);return target.count

result={'status':'CHECKPOINT_SIZE_PLANNING_ONLY','torch_version':torch.__version__,'cpu_only':True,
        'source_sha256':sha(pathlib.Path(__file__)),'new_checkpoint_files_written':0,'optimizer_updates':0,
        'physical_filter_executed':False,'dataset_accepted':False,'formal_training_started':False,
        'stages':{},'limitations':['Existing diagnostic weights only; accepted-corpus binding and future state may differ.',
         'CPU-mapped serialization changes device metadata; formal GPU checkpoint size has not been measured.',
         'One percent serialization allowance is a planning margin, not a proven upper bound.',
         'Does not include final WBC, generated augmentation, full corpus/cache or all logs.']}
for stage,config_name,steps in [('vae','mvae',200000),('ldm','dar',300000)]:
 config=ROOT/('upstream/TextOpRobotMDAR/robotmdar/config/train/'+config_name+'.yaml')
 text=config.read_text();assert re.search(r'use_ema:\s*False',text)
 interval=int(re.search(r'save_every:\s*(\d+)',text).group(1))
 p=ROOT/('generator_validation/generator_state_author_v1/'+stage+'/state_after4.pth')
 before=sha(p);value=torch.load(p,map_location='cpu')
 assert value['schema']=='textop.generator_training_state.v1' and value['step']==4 and value['ema_models']
 original_virtual=serialized_size(value)
 ema_bytes=sum(t.numel()*t.element_size() for state in value['ema_models'].values() for t in state.values() if torch.is_tensor(t))
 value['ema_models']={};without=serialized_size(value);del value;gc.collect()
 assert sha(p)==before and without<original_virtual and ema_bytes>0
 budget=math.ceil(without*1.01)
 result['stages'][stage]={'existing_checkpoint_sha256':before,'existing_file_bytes':p.stat().st_size,
  'cpu_serialization_ema_on_bytes':original_virtual,'ema_tensor_bytes':ema_bytes,
  'cpu_serialization_ema_off_bytes':without,'per_checkpoint_planning_bytes_with_one_percent_margin':budget,
  'checkpoint_count':math.ceil(steps/interval),'scheduled_planning_bytes':budget*math.ceil(steps/interval),
  'formal_ema_enabled':False,'official_config_sha256':sha(config),'source_checkpoint_unchanged':True}
result['disk_free_bytes']={p:shutil.disk_usage(p).free for p in ['/data','/home']}
OUT.write_text(json.dumps(result,indent=2));print(json.dumps(result))

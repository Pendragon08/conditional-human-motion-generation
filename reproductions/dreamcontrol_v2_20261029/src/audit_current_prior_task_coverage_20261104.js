const fs=require('fs'),path=require('path'),{execFileSync}=require('child_process');
const script=String.raw`import json,re,hashlib
from pathlib import Path
from collections import Counter
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');D=R/'stage2_prior29_20261103'
cfg=json.loads((D/'config.json').read_text());manifest=Path(cfg['manifest']);h=hashlib.sha256(manifest.read_bytes()).hexdigest();assert h==cfg['manifest_sha256']
j=json.loads(manifest.read_text());rows=j['records'];train=[r for r in rows if r['split']=='train']
print('ROWS',len(rows),len(train),'ROW_KEYS',list(train[0]),'DEX_ROW',json.dumps(next(r for r in train if 'GRAB' in r['dataset_source']),ensure_ascii=False)[:2200])
patterns={
 'pick_grasp_lift':r'\b(pick(?:s|ed|ing)?|grasp(?:s|ed|ing)?|grab(?:s|bed|bing)?|lift(?:s|ed|ing)?)\b',
 'pour':r'\bpour(?:s|ed|ing)?\b',
 'wipe':r'\bwip(?:e|es|ed|ing)\b',
 'vertical_surface':r'\b(vertical|wall|window)\b',
 'drawer':r'\bdrawer(?:s)?\b',
 'squat_crouch':r'\b(squat(?:s|ted|ting)?|crouch(?:es|ed|ing)?)\b',
 'bend':r'\bbend(?:s|ing)?\b|\bbent\b',
 'punch_boxing':r'\b(punch(?:es|ed|ing)?|boxing)\b',
 'step_walk':r'\b(step(?:s|ped|ping)?|walk(?:s|ed|ing)?)\b'}
def captions(r):
 values=[]
 for k in ['caption','candidate_captions','captions','text','texts']:
  value=r.get(k)
  if isinstance(value,str):values.append(value)
  elif isinstance(value,list):values.extend(v for v in value if isinstance(v,str))
 return list(dict.fromkeys(values))
result={}
for key,pattern in patterns.items():
 hits=[r for r in train if any(re.search(pattern,c,re.I) for c in captions(r))]
 result[key]={'training_trajectories_with_text_match':len(hits),'by_source':dict(Counter(r['dataset_source'] for r in hits)),
  'examples':[{'sample_id':r['sample_id'],'source':r['dataset_source'],'caption':next(c for c in captions(r) if re.search(pattern,c,re.I))} for r in hits[:8]]}
sources={}
for source in sorted({r['dataset_source'] for r in train}):
 rr=[r for r in train if r['dataset_source']==source]
 sources[source]={'training_trajectories':len(rr),'unique_caption_strings':len({c for r in rr for c in captions(r)}),
  'most_common_captions':Counter(c for r in rr for c in captions(r)).most_common(20)}
out=R/'stage2_prior29_20261103/task_coverage_recheck_20261104';out.mkdir(exist_ok=True)
report={'scope':'exact training manifest including all candidate_captions sampled by the actual training loader; lexical evidence only, not semantic acceptance or physical success',
 'manifest':str(manifest),'manifest_sha256':h,'checkpoint_sha256':'109fcfebf2a805831444ca8fb55d69674ef7cb19c61219e3e4ed18876d9527a9',
 'dataset_trajectories':len(rows),'training_trajectories':len(train),'sources':sources,'queries':result,
 'AMASS_in_current_finetune':False,'Nymeria_in_current_finetune':False,
 'mdm_pretraining_note':'MDM initialization may retain human motion abilities; that is not evidence of robot/task-specific fine-tune coverage'}
(out/'caption_retrieval_all_captions_v2.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps(report,ensure_ascii=False))
`;
const output=execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -'],{input:script,encoding:'utf8',timeout:30000});
process.stdout.write(output);
const j=JSON.parse(output.trim().split('\n').pop()),L=path.resolve(__dirname,'../reports/current_prior_task_coverage_20261104');fs.mkdirSync(L,{recursive:true});fs.writeFileSync(path.join(L,'caption_retrieval_all_captions_v2.json'),JSON.stringify(j,null,2));

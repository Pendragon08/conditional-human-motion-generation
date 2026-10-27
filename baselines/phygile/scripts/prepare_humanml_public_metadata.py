from pathlib import Path
import csv,datetime,hashlib,io,json,math,urllib.request,zipfile
r=Path(__file__).resolve().parents[1]
out=r/'cache/humanml3d_public_20261018';out.mkdir(parents=True,exist_ok=False)
def get(url):
 req=urllib.request.Request(url,headers={'User-Agent':'PhyGile-reproduction-data-inspection'})
 with urllib.request.urlopen(req,timeout=45) as res:return res.read(20_000_001)
commit=json.loads(get('https://api.github.com/repos/EricGuo5513/HumanML3D/commits/main'))['sha']
names=['README.md','LICENSE','index.csv','HumanML3D/texts.zip',
       'HumanML3D/all.txt','HumanML3D/train.txt','HumanML3D/val.txt','HumanML3D/test.txt','HumanML3D/train_val.txt']
files=[]
for name in names:
 url=f'https://raw.githubusercontent.com/EricGuo5513/HumanML3D/{commit}/{name}'
 b=get(url)
 if len(b)>20_000_000:raise ValueError('Unexpectedly large public metadata file')
 p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
 files.append({'path':name,'url':url,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
 print('Downloaded',name,len(b),flush=True)
split={n:set((out/f'HumanML3D/{n}.txt').read_text().split()) for n in ('all','train','val','test','train_val')}
overlap={a+'_'+b:len(split[a]&split[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
if any(overlap.values()):raise ValueError(('Official split overlap',overlap))
mirror_mismatches=[]
for n in ('train','val','test'):
 for item in split[n]:
  if item.startswith('M') and item[1:] not in split[n]:mirror_mismatches.append((n,item))
captions=0;segments=0;caption_ids=set();invalid=[]
with zipfile.ZipFile(out/'HumanML3D/texts.zip') as z:
 if z.testzip() is not None:raise ValueError('Text archive CRC failed')
 for entry in z.infolist():
  p=Path(entry.filename)
  if p.suffix!='.txt' or p.name.startswith('._') or '__MACOSX' in p.parts:continue
  caption_ids.add(p.stem)
  for line in z.read(entry).decode('utf-8-sig').splitlines():
   if not line.strip():continue
   fields=line.split('#')
   if len(fields)!=4:invalid.append({'file':entry.filename,'error':'caption_fields'});continue
   captions+=1
   try:
    start,end=map(float,fields[2:])
    if not math.isfinite(start) or not math.isfinite(end):
     invalid.append({'sample_id':p.stem,'issue':'nonfinite_time_tags'});continue
    if start!=0 or end!=0:segments+=1
   except ValueError:invalid.append({'file':entry.filename,'error':'time_tags'})
index=list(csv.DictReader((out/'index.csv').open(encoding='utf-8-sig',newline='')))
missing={n:sorted(v-caption_ids) for n,v in split.items()}
result={'status':'PUBLIC_METADATA_DOWNLOADED_NOT_A_G1_PAIRED_TRAINING_SET',
 'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source_repository':'https://github.com/EricGuo5513/HumanML3D','pinned_commit':commit,
 'files':files,'split_counts':{n:len(v) for n,v in split.items()},'split_overlap':overlap,
 'mirrored_split_mismatches':mirror_mismatches,'caption_file_count':len(caption_ids),
 'caption_line_count':captions,'nonzero_time_tag_captions':segments,'invalid_caption_lines':invalid,
 'split_ids_missing_text':missing,'source_index_rows':len(index),'source_index_columns':list(index[0]),
 'robot_motion_files_obtained':0,'g1_262d_paired_samples_prepared':0,'generator_training_started':False,
 'still_needed':['AMASS motion sequences mapped to HumanML3D IDs and time ranges',
  'Body model files compatible with the chosen retargeting route; official GMR SMPL-X route requires SMPL-X model and motion inputs',
  'Time-aligned G1 retargeting, semantic/kinematic quality checks, 262D extraction and manifest preparation'],
 'official_data_routes':{'humanml':'https://github.com/EricGuo5513/HumanML3D#how-to-obtain-the-data',
   'amass':'https://amass.is.tue.mpg.de/download.php','smplx':'https://smpl-x.is.tue.mpg.de/',
   'gmr':'https://github.com/YanjieZe/GMR#data-preparation'},
 'note':'Public text/index/split acquisition is not acquisition of paired robot-native motion. No original/robot motion was downloaded or trained here.'}
proof=r/'results/paired_data_20261018';proof.mkdir(exist_ok=False)
(proof/'public_metadata_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('files','still_needed','official_data_routes','split_ids_missing_text','mirrored_split_mismatches','invalid_caption_lines')},ensure_ascii=False,indent=2))

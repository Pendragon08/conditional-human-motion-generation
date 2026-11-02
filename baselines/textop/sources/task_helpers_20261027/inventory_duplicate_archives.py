import collections,hashlib,json,pathlib,re,subprocess
TASK=pathlib.Path(__file__).resolve().parent
roots=[TASK.parent.parent,pathlib.Path('D:/具身智能')]
files=[]
for root in roots:
    command=['rg','--files','--hidden','--no-ignore','-g','*.zip','-g','*.tar.gz','-g','*.mp4','-g','!**/.git/**','-g','!**/site-packages/**','-g','!**/node_modules/**','-g','!**/env/**','-g','!**/venv/**',str(root)]
    result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace')
    files.extend(pathlib.Path(line) for line in result.stdout.splitlines())
groups=collections.defaultdict(list)
for p in files:
    if re.search(r'(?i)AMASS|SMPL|HumanML|checkpoint|ckpt|dataset|原始资产|body_model|[\\/]data[\\/]|[\\/]envs[\\/]|\.env[\\/]',str(p)):continue
    if not re.search(r'(?i)PhyGile|ReactiveBFM|conditional-human-motion-generation|服务器磁盘清理',str(p)):continue
    if p.is_file() and not p.is_symlink() and p.stat().st_size>=1024**2:groups[p.stat().st_size].append(p)
duplicates=[]
for size,paths in groups.items():
    if len(paths)<2:continue
    by_hash=collections.defaultdict(list)
    for p in paths:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
        by_hash[h.hexdigest()].append(str(p))
    for digest,matches in by_hash.items():
        if len(matches)>1:duplicates.append({'sha256':digest,'bytes_each':size,'paths':matches,'potential_duplicate_bytes':size*(len(matches)-1)})
report={'duplicates':sorted(duplicates,key=lambda x:x['potential_duplicate_bytes'],reverse=True),'potential_bytes':sum(x['potential_duplicate_bytes'] for x in duplicates),'deleted_files':0,'github_verification_required_before_deletion':True}
(TASK/'duplicate_archives_inventory.private.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'groups':len(duplicates),'potential_gib':report['potential_bytes']/1024**3,'top':report['duplicates'][:10]},ensure_ascii=True))

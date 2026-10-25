"""Verify the two completed private archives and publish metadata only."""
import hashlib,json,pathlib
from session import REPO
BASE=REPO/'baselines/textop';LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024')
out={'scope':'Completed500+143 candidate preprocessing, not physical tracking acceptance','original_failure':'Transient SSH connection abort left a zero-byte partial; original failure record preserved','data_deleted':False,'server_retargeting_interrupted':False,'archives':[]}
for batch in ['available500_v1','available143_v1']:
    receipt=json.loads((LOCAL/(batch+'_private_data_backup_receipt.json')).read_text(encoding='utf-8'));p=pathlib.Path(receipt['path'])
    h=hashlib.sha256()
    with p.open('rb') as source:
        for block in iter(lambda:source.read(8*1024**2),b''):h.update(block)
    assert h.hexdigest()==receipt['sha256'] and p.stat().st_size==receipt['bytes']
    out['archives'].append({k:receipt[k] for k in ['bytes','sha256','motion_files','time_utc']})
out['failed_partials_preserved']=[{'filename':p.name,'bytes':p.stat().st_size} for p in (LOCAL/'datasets').glob('*.partial.failed_*')]
out['verified_candidate_motions']=sum(x['motion_files'] for x in out['archives'])
(BASE/'results/data_preparation_20261025/backup_recovery_verified.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
(LOCAL/'数据备份修复核验_20261025.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({'verified_candidate_motions':out['verified_candidate_motions'],'archives':len(out['archives']),'data_deleted':False}))

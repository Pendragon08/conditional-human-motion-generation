"""Keep private reconstruction indexes with durable original archives locally."""
import json,pathlib,shutil,sys
from session import REPO
from restore_indexed_cache import sha
BASE=REPO/'baselines/textop';TASK=REPO.parent.parent/'textop_reproduction_20261024'
CACHE=TASK/'private_assets/indexed_raw_expansion_v1'
LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024')
KIT=LOCAL/'datasets/recovery1653_private';KIT.mkdir(parents=True,exist_ok=True)
rows=[]
for source,name in [(CACHE/'ready.private.json','ready.private.json'),(CACHE/'raw_index.private.jsonl','raw_index.private.jsonl'),(pathlib.Path(__file__).with_name('restore_indexed_cache.py'),'restore_indexed_cache.py')]:
    target=KIT/name
    if target.exists():assert sha(target)==sha(source)
    else:shutil.copy2(source,target)
    assert sha(target)==sha(source)
    rows.append({'file':name,'bytes':target.stat().st_size,'sha256':sha(target)})
command='python restore_indexed_cache.py --index ready.private.json --archive-dir "D:/具身智能/PhyGile复现/后续AMASS下载_20261021/archives" --output "<private cache directory with enough free space>"'
(KIT/'恢复说明.txt').write_text('全部1653条原始动作可从保留的完整许可压缩包按此索引恢复。保留6GiB空余；--limit 1可只恢复首条。\n'+command+'\n原始动作和逐条文本只供许可允许的个人研究，不公开上传GitHub。\n',encoding='utf-8')
receipt={'private_recovery_indexes_verified_on_D':True,'files':rows,'cache_files_present_on_C':len(list(CACHE.glob('*_stageii.npz'))),
         'current_durable_inputs':'complete D-drive licensed archives plus private indexes on C and D',
         'server_RAM_is_working_copy_only':True,'unique_dataset_files_deleted':0,'private_indexes_published':False}
(BASE/'results/data_preparation_20261025/cache_residency.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt),flush=True)

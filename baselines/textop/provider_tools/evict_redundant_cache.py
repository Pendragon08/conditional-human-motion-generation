"""Remove only proven-recoverable temporary extraction duplicates, not datasets.

Original licensed tar.bz2 packages, private indexes, server working inputs and
all prepared outputs remain. Requires published recovery code/evidence first.
No recursive deletion; every absolute path is checked against one exact root.
"""
import argparse,datetime,json,pathlib,shutil,subprocess,sys
from session import REPO,GH
from restore_indexed_cache import sha,ARCHIVES
BASE=REPO/'baselines/textop';TASK=REPO.parent.parent/'textop_reproduction_20261024'
CACHE=(TASK/'private_assets/indexed_raw_expansion_v1').resolve()
ADIR=pathlib.Path('D:/具身智能/PhyGile复现/后续AMASS下载_20261021/archives')
sys.stdout.reconfigure(encoding='utf-8')

def main(commit,evict):
    plan=json.loads((TASK/'cache_eviction_plan.private.json').read_text(encoding='utf-8'))
    proof=plan['receipt'];rows=plan['files']
    assert len(rows)==1653 and proof['cache_files_verified']==1653
    assert sha(CACHE/'ready.private.json')==proof['index_sha256']
    assert proof['fresh_archive_restore_witness']['all_member_sha256_match']
    assert proof['server_working_copy']['files_verified']==1653
    for original in proof['retained_original_archives']:
        p=ADIR/original['file'];assert p.is_file() and p.stat().st_size==original['bytes']
        assert sha(p)==original['sha256'],'Original dataset backup failed verification'
    # GitHub confirms the immutable commit really exists. Verify byte-for-byte
    # contents of the recovery implementation and proof at that backed-up commit.
    subprocess.run([GH,'api','repos/Pendragon08/conditional-human-motion-generation/git/commits/'+commit],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    for relative in ['provider_tools/restore_indexed_cache.py','provider_tools/evict_redundant_cache.py','results/data_preparation_20261025/cache_recovery_verified.json']:
        rel='baselines/textop/'+relative
        recorded=subprocess.check_output(['git','-C',str(REPO),'show',commit+':'+rel])
        current=(BASE/relative).read_bytes()
        if relative.endswith('.json'):
            # Windows text writes CRLF; Git stores LF. Compare parsed evidence,
            # preserving every key/value rather than weakening source checks.
            assert json.loads(recorded)==json.loads(current),'Recovery evidence differs from backed-up commit'
        else:
            assert recorded==current,'Recovery source differs from backed-up commit'
    for row in rows:
        p=pathlib.Path(row['path']).resolve()
        assert p.parent==CACHE and p.name.endswith('_stageii.npz')
        assert not p.is_symlink() and p.is_file() and p.stat().st_size==row['bytes']
        assert sha(p)==row['sha256'],'Cache changed; refusing cleanup'
    before=shutil.disk_usage('C:/').free;removed=[]
    if evict:
        with (TASK/'cache_eviction.private.jsonl').open('x',encoding='utf-8') as log:
            for row in rows:
                p=pathlib.Path(row['path']).resolve()
                assert p.parent==CACHE and not p.is_symlink()
                p.unlink();removed.append(row['sha256'])
                log.write(json.dumps({'path':str(p),'sha256':row['sha256'],'recover_from_original_archive':True})+'\n');log.flush()
        assert not any(CACHE.glob('*_stageii.npz'))
    result={'time_beijing':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            'status':'REDUNDANT_EXTRACTION_CACHE_EVICTED' if evict else 'RECOVERY_VERIFIED_DRY_RUN',
            'files_unlinked':len(removed),'recoverable_logical_bytes':sum(x['bytes'] for x in rows),
            'previous_cache_physical_bytes':proof['cache_physical_bytes'],
            'C_free_before':before,'C_free_after':shutil.disk_usage('C:/').free,
            'recovery_code_and_evidence_backed_up_github_commit':commit,
            'original_dataset_archives_deleted':0,'prepared_dataset_files_deleted':0,
            'checkpoints_deleted':0,'experiment_records_deleted':0,
            'private_indexes_retained':True,'server_working_copy_retained':True,
            'restore_command':'restore_indexed_cache.py --index <retained ready.private.json> --archive-dir <retained original archives> --output <private cache> --limit <optional count>',
            'restore_helper_sha256':proof['restore_helper_sha256']}
    (BASE/'results/data_preparation_20261025/redundant_cache_cleanup.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (pathlib.Path('D:/具身智能/TextOp复现_20261024')/'重复解压缓存清理_20261025.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--backed-up-commit',required=True);a.add_argument('--evict',action='store_true')
    x=a.parse_args();main(x.backed_up_commit,x.evict)

"""Record explicit user authorization and observed access; no credentials."""
import datetime,json,pathlib,shutil
from session import REPO
BASE=REPO/'baselines/textop';DEST=BASE/'results/data_preparation_20261025'
LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024')
archive_dir=pathlib.Path('D:/具身智能/PhyGile复现/后续AMASS下载_20261021/archives')
expected={'BMLrub':12919496284,'CMU':11572134418,'KIT':13182414052}
rows=[]
for group,size in expected.items():
    partial=archive_dir/(group+'_smplx_g.tar.bz2.part')
    local_partial=REPO.parent.parent/'textop_reproduction_20261024/private_assets/incoming_amass'/(group+'_smplx_g.tar.bz2.part')
    local_final=local_partial.with_suffix('')
    have=max(p.stat().st_size if p.exists() else 0 for p in [partial,local_partial,local_final])
    rows.append({'subset':group,'expected_archive_bytes_from_prior_official_manifest':size,'current_partial_bytes':have,'remaining_bytes_estimate':size-have})
receipt={'recorded_beijing':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
         'amass':{'user_reported_login_complete':True,'observed_main_download_table_accessible':True,'file_host_session_verified_this_turn':False,'existing_approved_subsets_remain_authorized':True,'totalcapture_skipped':True},
         'babel':{'user_explicit_license_authorization':True,'license_url':'https://babel.is.tue.mpg.de/license.html','scope':'non-commercial research; personal reproduction storage; no redistribution','observed_download_redirect':'https://babel.is.tue.mpg.de/login.php','download_completed':False,'status':'WAITING_FOR_INDEPENDENT_BABEL_LOGIN'},
         'storage':{'C_free_bytes':shutil.disk_usage('C:/').free,'D_free_bytes':shutil.disk_usage('D:/').free,'remaining_archives':rows,'total_additional_archive_bytes_estimate':sum(x['remaining_bytes_estimate'] for x in rows),'status':'NEEDS_MORE_DURABLE_STORAGE_BEFORE_COMPLETE_BATCH','requested_free_capacity_gb':50},
         'credentials_recorded':False,'new_raw_assets_published':False}
babel_acquisition=DEST/'babel_acquisition.json';babel_alignment=DEST/'babel_alignment.json';babel_server=DEST/'babel_server_install.json'
if babel_acquisition.exists():
    acquired=json.loads(babel_acquisition.read_text(encoding='utf-8'))
    receipt['babel'].update(user_reported_login_complete=True,observed_download_redirect=None,
        observed_download_page='https://babel.is.tue.mpg.de/download.php',download_completed=True,
        status=acquired['status'],archive_sha256=acquired['archive_sha256'],
        server_installed=babel_server.exists(),original_alignment_checked=babel_alignment.exists())
cmu=DEST/'cmu_resume.json'
if cmu.exists():
    receipt['amass']['file_host_session_verified_this_turn']=True
    receipt['storage']['status']='CMU_RESUMING_WITH_6GIB_C_RESERVE_OTHER_ARCHIVES_STILL_PENDING'
DEST.mkdir(parents=True,exist_ok=True);(DEST/'asset_authorization_and_storage.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
(LOCAL/'数据授权与空间核查_20261025.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'babel_authorized':True,'babel_download_complete':receipt['babel']['download_completed'],'remaining_archive_gib':receipt['storage']['total_additional_archive_bytes_estimate']/1024**3,'storage_status':receipt['storage']['status']}))

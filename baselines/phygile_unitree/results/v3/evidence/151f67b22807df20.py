from pathlib import Path
import sys,json,hashlib,datetime,xml.etree.ElementTree as ET,shutil
E=Path('D:/具身智能/PhyGile复现/Unitree框架适配_20261022/source_geometry_doc_witness_v1')
TASK=Path('C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_unitree_20261022')
BASE=Path('C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/phygile_unitree')
ASSETS=Path('D:/具身智能/PhyGile复现/Unitree框架适配_20261022/private_assets_contact_v3')
OUT=E.parent/'source_geometry_comparison_v1'
EXPECTED='dbc8471004e9b0b85c27a4ebc1f0d90a727c1cf7ca9ef507b805596f2004d64b'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def dump(name,data):
    (E/name).write_text(json.dumps(data,sort_keys=True,indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
if sys.argv[1]=='before':
    assert not OUT.exists(),'Documented output directory exists; experiment not launched'
    spec=json.loads((BASE/'configs/environment_plan_v2.json').read_text(encoding='utf-8-sig'))
    spec_hash=canonical(spec)
    assert spec_hash=='a43f960bd3bcdcd3398655bffe291138b73de5ca7f30901765172ff9821e1954',spec_hash
    assert sha(TASK/'source_main_g1.urdf')==EXPECTED
    root=ET.parse(TASK/'source_main_g1.urdf').getroot()
    count=sum(len(link.findall('collision')) for link in root.findall('link'))
    assert count==29,count
    roots=[BASE,TASK/'vendor_unitree_rl_mjlab',Path('D:/phygile_unitree_env_20261022/Lib/site-packages/mjlab')]
    files=set()
    for folder in roots:
        for suffix in ('*.py','*.sh'):
            files.update(p.resolve() for p in folder.rglob(suffix) if p.is_file())
    files.update(p.resolve() for p in TASK.glob('*.py') if p.is_file())
    files.update(p.resolve() for p in ASSETS.iterdir() if p.is_file())
    files.update(p.resolve() for p in [TASK/'source_main_g1.urdf',TASK/'GEOMETRY_INVOCATION_V1.md',BASE/'configs/environment_plan_v2.json',BASE/'docs/COMPUTE_LEDGER.md',E/'verify_evidence.py',E/'documented_command.ps1'])
    hashes={str(p):sha(p) for p in sorted(files)}
    dump('before.json',{'utc':utc(),'canonical_spec_sha256':spec_hash,'source_urdf_sha256':EXPECTED,'source_collision_count':count,'files':hashes,'files_count':len(hashes),'scope':'Warm local CPU; full source/input file hashes only, private array values are never published'})
    print(json.dumps({'preflight':'PASS','source_collision_count':count,'hashed_files':len(hashes),'spec_hash':spec_hash}))
else:
    import numpy as np
    before=json.loads((E/'before.json').read_text(encoding='utf-8'))
    report={'status':'FAILED','validated_utc':utc(),'documented_invocation_count':1,'raw_arrays_published':False,'server_operations':False,'environment_warm_reuse':True,'runs':[]}
    try:
        actual_exit=json.loads((E/'process_exit.json').read_text(encoding='utf-8-sig'))
        assert actual_exit['actual_python_exit_code']==0,actual_exit
        comparison=json.loads((OUT/'comparison.json').read_text(encoding='utf-8'))
        assert comparison['status']=='FOUR_GEOMETRY_DIAGNOSTICS_COMPLETED'
        assert comparison['seed']==2026 and comparison['policy_updates']==0
        assert comparison['development_clips'] is True and comparison['parameter_sweep_or_best_profile_selection'] is False
        assert len(comparison['runs'])==4
        definitions_hashes=set();manifest_counts=[];executed=set()
        for i,run in enumerate(comparison['runs']):
            folder=OUT/f'clip{i}';receipt_path=folder/'receipt.json'
            receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
            assert run['exit_code']==0 and run['receipt']==receipt
            assert receipt['status']=='FIRST_EPISODE_EVALUATED'
            assert receipt['clip']==i and receipt['seed']==2026 and receipt['optimizer_steps']==0 and receipt['device']=='cpu'
            assert receipt['action_scale_offset_exact'] and receipt['post_physics_forward_before_termination'] and receipt['executed_sources_unchanged']
            assert receipt['geometry_profile']=='source_urdf_capsule_primitives'
            geometry=receipt['geometry_provenance']
            assert geometry['source_urdf_sha256']==EXPECTED
            assert geometry['source_collision_count']==29 and len(geometry['source_definitions'])==29
            definitions_hash=canonical(geometry['source_definitions']);definitions_hashes.add(definitions_hash)
            assert receipt['contact_profile']=='matched_sliding_friction'
            assert receipt['checkpoint_sha256']==before['files'][str((ASSETS/'parent_checkpoint.pt').resolve())]
            assert receipt['motion_sha256']==before['files'][str((ASSETS/f'test_{i:03d}.npz').resolve())]
            proof=receipt['source_provenance'];manifest_path=Path(proof['manifest'])
            assert sha(manifest_path)==proof['sha256']
            manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
            assert len(manifest['files'])==proof['source_files']
            for name,digest in manifest['files'].items():
                resolved=str(Path(name).resolve())
                assert resolved in before['files'],'Executed source missing from pre-snapshot: '+resolved
                assert before['files'][resolved]==digest==sha(name),'Executed source/input changed: '+name
                executed.add(resolved)
            profile_path=str((TASK/'source_geometry_profile.py').resolve())
            assert manifest['files'].get(profile_path)==before['files'][profile_path]
            with np.load(folder/'raw_pre_reset.npz',allow_pickle=False) as raw:
                assert all(np.isfinite(raw[k]).all() for k in raw.files)
                assert np.array_equal(raw['frame'],np.arange(len(raw['frame'])))
                assert len(raw['frame'])==receipt['frames_evaluated']
                computed={'mpjpe_m_until_termination':float(np.linalg.norm(raw['reference_body_pos']-raw['robot_body_pos'],axis=-1).mean()),'mpjae_rad_until_termination':float(np.abs(raw['reference_joint_pos']-raw['robot_joint_pos']).mean()),'mpjve_rad_s_until_termination':float(np.abs(raw['reference_joint_vel']-raw['robot_joint_vel']).mean())}
                assert all(np.isfinite(v) for v in computed.values())
                assert all(np.isclose(v,receipt['metrics'][k],rtol=1e-12,atol=1e-12) for k,v in computed.items()),computed
                assert receipt['terminated']==bool(raw['terminated'][-1]) and receipt['timeout']==bool(raw['timeout'][-1])
            for contact in receipt['compiled_contact_parameters']:
                assert contact['friction'][0]==1.0
                assert contact['condim']==3
            shutil.copyfile(receipt_path,E/f'clip{i}_receipt.json')
            shutil.copyfile(manifest_path,E/f'clip{i}_executed_sources.json')
            report['runs'].append({'clip':i,'exit_code':run['exit_code'],'receipt_sha256':sha(receipt_path),'source_manifest_sha256':proof['sha256'],'source_file_count':proof['source_files'],'source_urdf_sha256':geometry['source_urdf_sha256'],'geometry_profile_source_sha256':before['files'][profile_path],'source_definitions_sha256':definitions_hash,'source_collision_count':29,'raw_pre_reset_sha256':sha(folder/'raw_pre_reset.npz'),'policy_io_sha256':sha(folder/'policy_io_private.npz'),'frames_evaluated':receipt['frames_evaluated'],'motion_frames':receipt['motion_frames'],'terminated':receipt['terminated'],'timeout':receipt['timeout'],'full_clip_success':receipt['full_clip_success'],'metrics_recomputed':computed,'metrics_match':True,'raw_finite':True,'optimizer_steps':receipt['optimizer_steps'],'checkpoint_sha256':receipt['checkpoint_sha256']})
            manifest_counts.append(proof['source_files'])
        assert len(definitions_hashes)==1
        after={name:sha(name) for name in before['files']}
        changed=[name for name,digest in before['files'].items() if after[name]!=digest]
        dump('after.json',{'utc':utc(),'files':after,'all_presnapshotted_files_unchanged':not changed,'changed_files':changed})
        assert not changed,'Pre-snapshotted source/input changed: '+repr(changed)
        shutil.copyfile(OUT/'comparison.json',E/'comparison.json')
        report.update(status='PASS',comparison_status=comparison['status'],comparison_sha256=sha(OUT/'comparison.json'),actual_python_exit_code=actual_exit['actual_python_exit_code'],all_executed_sources_unchanged=True,executed_unique_files=len(executed),all_presnapshotted_files_unchanged=True,presnapshotted_files=before['files_count'],source_collision_count=29,source_definitions_sha256=next(iter(definitions_hashes)),full_clip_successes=sum(r['full_clip_success'] for r in report['runs']),policy_updates=0,frozen_policy_evidence='Original checkpoint and normalizer inputs unchanged, exact checkpoint load, model.eval, torch.inference_mode, optimizer_steps=0, comparison policy_updates=0')
    except BaseException as exc:
        import traceback
        report.update(error=repr(exc),traceback=traceback.format_exc())
    dump('report.json',report)
    print(json.dumps({k:report.get(k) for k in ['status','comparison_status','actual_python_exit_code','full_clip_successes','source_collision_count','error']}))
    if report['status']!='PASS':sys.exit(2)
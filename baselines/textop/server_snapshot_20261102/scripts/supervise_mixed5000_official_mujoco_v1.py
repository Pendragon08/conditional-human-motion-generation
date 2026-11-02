"""Resource-bounded, exclusive MuJoCo evaluation supervisor."""
import argparse,fcntl,hashlib,json,os,pathlib,shutil
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
RUN='user_mixed5000_official_mujoco_dance_v1'
ENTRY=ROOT/'scripts/run_mixed5000_official_mujoco_v1.py'
FLOOR=2*1024**3+128*1024**2+512*1024**2


def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--trusted-request-sha256',required=True);a=p.parse_args()
    request=pathlib.Path(a.request)
    if not __debug__ or os.getuid()!=1013 or request.resolve()!=request or request.parent!=ROOT/'receipts' or sha(request)!=a.trusted_request_sha256:raise ValueError('request identity')
    job=json.loads(request.read_text())
    if job['run']!=RUN or job['authorized'] is not True or sha(__file__)!=job['supervisor_sha256'] or sha(ENTRY)!=job['executor_sha256']:raise ValueError('wrong job')
    for name,h in job['bound_sources'].items():
        path=ROOT/name
        if pathlib.PurePosixPath(name).is_absolute() or '..' in pathlib.PurePosixPath(name).parts or path.resolve()!=path or path.stat().st_uid!=1013 or sha(path)!=h:raise ValueError('source differs: '+name)
    from contact_filter_resources import sample_resources
    from contact_process_supervision import run_supervised
    from mixed5000_dance_tracking_inputs_v1 import accept
    os.sched_setaffinity(0,{0,1,2,3});parent=ROOT/'step34_20261029';out=parent/RUN;logs=parent/(RUN+'_supervision')
    with (ROOT/'receipts/full_library_capacity.lock').open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if out.exists() or logs.exists() or shutil.disk_usage(parent).free<FLOOR:raise ValueError('existing output or reserve')
        accept(job['input_binding'])
        onnx=parent/'user_mixed5000_onnx_export_v1/policy.onnx';receipt=parent/'user_mixed5000_onnx_export_v1/receipt.json'
        if sha(onnx)!=job['onnx_sha256'] or sha(receipt)!=job['onnx_receipt_sha256']:raise ValueError('ONNX provenance')
        asset=pathlib.Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/GMR/assets/unitree_g1/g1_mocap_29dof.xml')
        if asset.resolve()!=asset or sha(asset)!=job['asset_sha256']:raise ValueError('G1 asset differs')
        for name,digest in job['mesh_hashes'].items():
            mesh=pathlib.Path(name)
            if mesh.resolve()!=mesh or asset.parent not in mesh.parents or sha(mesh)!=digest:raise ValueError('G1 mesh differs')
        sample_resources(ROOT,None,8*1024**3,FLOOR)
        logs.mkdir(mode=0o700,exist_ok=False)
        def sample(pid):
            if shutil.disk_usage(parent).free<FLOOR:raise RuntimeError('output reserve unavailable')
            return sample_resources(ROOT,pid,8*1024**3,FLOOR)
        py='/data/code/ouyanghao/oyh/phygile-reproduction-20261017/envs/gmr_amass_20261021/bin/python'
        site=parent/'mixed5000_ort_overlay_v1/site'
        cmd=['taskset','-c','0-3','/usr/bin/env','-u','PYTHONOPTIMIZE','CUDA_VISIBLE_DEVICES=',
             'PYTHONDONTWRITEBYTECODE=1','PYTHONNOUSERSITE=1','PYTHONPATH='+str(site)+':'+str(ROOT/'scripts'),
             'OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4',py,'-B',str(ENTRY),'--request',str(request),
             '--trusted-request-sha256',a.trusted_request_sha256]
        proof=run_supervised(cmd,logs/'child.log',logs/'resources.jsonl',logs/'process_exit.json',sample,7200,15)
        recpath=out/'receipt.json';rec=json.loads(recpath.read_text()) if recpath.exists() else {}
        proof.update(run=RUN,request_sha256=sha(request),receipt_sha256=sha(recpath) if recpath.exists() else None,
                     resource_sha256=sha(logs/'resources.jsonl'),process_exit_sha256=sha(logs/'process_exit.json'))
        with (logs/'exit.json').open('x') as f:json.dump(proof,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
        if not(proof['supervision_passed'] and rec.get('status')=='MIXED5000_OFFICIAL_MUJOCO_DANCE_FINISHED'
               and rec.get('output_writers_closed') is True and rec.get('completed_trials')==16 and rec.get('optimizer_updates')==0):
            raise RuntimeError('evaluation failed; retain evidence')


if __name__=='__main__':main()

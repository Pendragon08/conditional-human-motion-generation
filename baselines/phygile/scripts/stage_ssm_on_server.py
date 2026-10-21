"""Stage a locally acquired SSM archive and verify it on the owner's SSH host.

Never exports browser credentials or publishes licensed motions to GitHub.
Preserves existing remote files and uses the existing retarget environment.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage(args):
    base=Path(__file__).resolve().parents[1]
    archive=args.archive.resolve(strict=True)
    root=PurePosixPath(args.remote_root)
    if not root.is_absolute() or len(root.parts)<5 or '..' in root.parts:
        raise ValueError('An explicit dedicated absolute remote data directory is required')
    if args.output.exists():raise FileExistsError('Existing deployment receipt preserved')
    args.output.mkdir(parents=True)
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-i',str(args.identity),args.host]
    scp=['scp','-o','BatchMode=yes','-o','ConnectTimeout=15','-i',str(args.identity)]
    started=time.monotonic()
    record={'status':'RUNNING','date':'2026-10-21','host':args.host,'remote_root':str(root),
            'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
            'body_model_weights_available':False,'g1_pairs_generated':0,
            'new_environment_installed':False,'github_dataset_upload_performed':False}

    def remote(name,source,timeout=90):
        run=subprocess.run(ssh+['python3','-'],input=source,text=True,encoding='utf-8',
                           stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
        (args.output/(name+'.stdout.log')).write_text(run.stdout,encoding='utf-8')
        (args.output/(name+'.stderr.log')).write_text(run.stderr,encoding='utf-8')
        if run.returncode:raise RuntimeError(name+' failed: '+run.stderr[-1800:])
        return json.loads(run.stdout)

    def transfer(local,destination):
        run=subprocess.run(scp+[str(local),args.host+':'+str(destination)],
                           stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',timeout=300)
        if run.returncode:raise RuntimeError('SCP failed: '+run.stderr[-1800:])

    try:
        probe=remote('preflight', '''import os,pathlib,shutil,json
r=pathlib.Path('''+repr(str(root))+''')
existing=r
while not existing.exists():existing=existing.parent
u=shutil.disk_usage(existing)
instructions=[]
for p in [r,*r.parents]:
 f=p/'AGENTS.md'
 if f.is_file():instructions.append(str(f))
print(json.dumps({'free_bytes':u.free,'total_bytes':u.total,'instructions':instructions,
 'retarget_python_exists':pathlib.Path('''+repr(args.remote_python)+''').is_file(),
 'root_exists':r.exists()}))
''')
        record['preflight']=probe
        if probe['instructions']:raise RuntimeError('Read remote AGENTS.md before deployment: '+str(probe['instructions']))
        if probe['free_bytes'] < archive.stat().st_size*5 + 5*1024**3:
            raise RuntimeError('Insufficient free space for bounded SSM staging plus 5 GiB reserve')
        if not probe['retarget_python_exists']:raise RuntimeError('Existing retarget Python is absent')
        remote('create_destination','''import pathlib,json
r=pathlib.Path('''+repr(str(root))+''')
r.mkdir(parents=True,exist_ok=True)
for name in ('SSM_smplx_g.tar.bz2','archive.incoming','ingestion','extracted','ssm_verification.json'):
 if (r/name).exists():raise FileExistsError('Preserve existing remote artifact: '+name)
print(json.dumps({'created':str(r)}))
''')
        transfer(archive,root/'archive.incoming')
        record['archive_remote_verification']=remote('verify_archive','''import pathlib,hashlib,json
r=pathlib.Path('''+repr(str(root))+''')
p=r/'archive.incoming'
h=hashlib.sha256(p.read_bytes()).hexdigest()
if h!='''+repr(record['archive_sha256'])+''':raise ValueError('Archive checksum differs')
p.rename(r/'SSM_smplx_g.tar.bz2')
print(json.dumps({'sha256':h,'bytes':(r/'SSM_smplx_g.tar.bz2').stat().st_size}))
''')
        files=['scripts/ingest_ssm_archive.py','scripts/plan_humanml_amass.py',
               'scripts/test_humanml_amass_plan.py','configs/g1_codec_skeleton.json',
               'reimplementation/codec262.py','reimplementation/features.py',
               'results/paired_data_20261018/public_metadata_manifest.json']
        files += ['cache/humanml3d_public_20261018/'+n for n in
                  ('README.md','LICENSE','index.csv','HumanML3D/texts.zip','HumanML3D/all.txt',
                   'HumanML3D/train.txt','HumanML3D/val.txt','HumanML3D/test.txt','HumanML3D/train_val.txt')]
        package=args.output/'public_metadata_and_ingestion.tar.gz'
        with tarfile.open(package,'w:gz') as tar:
            for name in files:tar.add(base/name,arcname=name,recursive=False)
        record['deployed_source_hashes']={n:sha(base/n) for n in files}
        package_sha=sha(package)
        transfer(package,root/'ingestion_payload.tar.gz')
        remote('extract_payload','''import pathlib,tarfile,hashlib,json
r=pathlib.Path('''+repr(str(root))+''')
p=r/'ingestion_payload.tar.gz'
if hashlib.sha256(p.read_bytes()).hexdigest()!='''+repr(package_sha)+''':raise ValueError('Payload checksum differs')
dest=r/'ingestion';dest.mkdir()
count=0
with tarfile.open(p,'r:gz') as tar:
 for item in tar.getmembers():
  name=pathlib.PurePosixPath(item.name)
  if not item.isfile() or name.is_absolute() or '..' in name.parts:raise ValueError('Unsafe payload member')
  out=dest.joinpath(*name.parts);out.parent.mkdir(parents=True,exist_ok=True)
  out.write_bytes(tar.extractfile(item).read());count+=1
print(json.dumps({'files':count,'payload_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
''')
        run=remote('ingest_ssm','''import pathlib,subprocess,json
r=pathlib.Path('''+repr(str(root))+''')
p='''+repr(args.remote_python)+'''
command=[p,str(r/'ingestion/scripts/ingest_ssm_archive.py'),'--archive',str(r/'SSM_smplx_g.tar.bz2'),
 '--public',str(r/'ingestion/cache/humanml3d_public_20261018'),'--output',str(r/'extracted'),
 '--receipt',str(r/'ssm_verification.json')]
process=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90)
(r/'ingestion.stdout.log').write_text(process.stdout)
(r/'ingestion.stderr.log').write_text(process.stderr)
if process.returncode:raise RuntimeError(process.stderr)
record=json.loads((r/'ssm_verification.json').read_text())
print(json.dumps(record))
''',timeout=110)
        record['remote_ssm_verification']=run
        if run['archive_sha256']!=record['archive_sha256'] or run['motion_sequences']!=30:
            raise RuntimeError('Unexpected remote SSM inventory')
        record.update(status='SSM_STAGED_AND_VERIFIED_ON_SERVER',full_amass_downloaded=False,
                      environment_fully_validated=False,
                      limitations=['Remaining AMASS subsets and SMPL-X model weights are absent.',
                                   'Full download needs a suitable persistent storage allocation.',
                                   'No GMR retargeting, new 262D data, or generator training executed.'])
    except BaseException as error:
        record.update(status='FAILED',error=type(error).__name__+': '+str(error))
        raise
    finally:
        record['elapsed_seconds']=time.monotonic()-started
        (args.output/'deployment.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:v for k,v in record.items() if k not in
                          ('remote_ssm_verification','deployed_source_hashes')},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',required=True)
    p.add_argument('--identity',required=True,type=Path)
    p.add_argument('--remote-root',required=True)
    p.add_argument('--remote-python',required=True)
    p.add_argument('--archive',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    stage(p.parse_args())

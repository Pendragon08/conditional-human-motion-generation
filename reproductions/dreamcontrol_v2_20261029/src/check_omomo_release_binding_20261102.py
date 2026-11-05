"""Run release-binding preflight on the actual unaccepted geometry preview."""
import hashlib,json
from pathlib import Path
from audit_omniretarget_release_binding import audit_release_binding
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PREVIEW=ROOT/'omomo_geometry_schema_preview_20261102'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    manifest=PREVIEW/'manifest.json';before=sha(manifest);report=json.loads((PREVIEW/'report.json').read_text())
    if before!=report['manifest_sha256']:raise ValueError('manifest mismatch')
    records=json.loads(manifest.read_text())['records']
    result=audit_release_binding(records,root=ROOT,state_manifest=PREVIEW/'state_manifest.json',
        archive=ROOT/'data/OmniRetarget_Dataset/robot-object.zip',
        card=ROOT/'omomo_release_evidence_work_20261102/omniretarget_README.md')
    if sha(manifest)!=before:raise RuntimeError('manifest changed')
    cases=result.pop('cases');result.update({'count':len(cases),'manifest_sha256':before,
         'script_sha256':sha(Path(__file__)),'training_ready':False})
    out=Path(__file__).parent/'real_preflight.json'
    with out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()

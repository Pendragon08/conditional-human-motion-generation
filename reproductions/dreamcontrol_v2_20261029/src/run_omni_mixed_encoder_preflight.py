"""Exercise actual state preflight on legacy and quarantined sign-fix data."""
import hashlib,json
from pathlib import Path
from train_robot_prior import state_preflight

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
SOURCE=ROOT/'omni_variants_gpu_20261101';SNAP=ROOT/'omni_variants_snapshot_20261101'
REPAIR=ROOT/'omni_variants_signfix_snapshot_20261101';INDEPENDENT=ROOT/'omni_variants_signfix_audit_20261101'
HERE=Path(__file__).resolve().parent
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    rep=json.loads((REPAIR/'report.json').read_text());ind=json.loads((INDEPENDENT/'report.json').read_text())
    snap=json.loads((SNAP/'report.json').read_text());sa=json.loads((SNAP/'audit_v2.json').read_text())
    if sha(REPAIR/'records.jsonl')!=rep['rows_sha256'] or sha(REPAIR/'report.json')!=ind['repair_report_sha256'] or ind['failed'] or ind['checked']!=rep['technical_previews']:
        raise ValueError('repair evidence mismatch')
    if sha(SNAP/'records.jsonl')!=snap['records_sha256'] or sha(SNAP/'report.json')!=sa['report_sha256'] or sha(SNAP/'audit_rows_v2.jsonl')!=sa['audit_rows_sha256']:
        raise ValueError('snapshot evidence mismatch')
    source={r['sample_id']:r for r in map(json.loads,(SNAP/'records.jsonl').read_text().splitlines())}
    verified={r['sample_id'] for r in map(json.loads,(INDEPENDENT/'records.jsonl').read_text().splitlines()) if r['status']=='checked'}
    repairs=[r for r in map(json.loads,(REPAIR/'records.jsonl').read_text().splitlines()) if r['status']=='technical_preview']
    if verified!={r['sample_id'] for r in repairs}:raise ValueError('independent audit coverage mismatch')
    legacy=next(r for r in map(json.loads,(SNAP/'audit_rows_v2.jsonl').read_text().splitlines()) if r['status']=='technical_checks_passed')
    rows=[];states=[]
    for item in [legacy]+repairs:
        sid=item['sample_id'];src=source[sid];is_sign=sid!=legacy['sample_id']
        feature=(REPAIR/item['feature']) if is_sign else SOURCE/src['feature']
        feature_hash=item['feature_sha256'] if is_sign else src['feature_sha256']
        state=SOURCE/src['state20']
        if sha(feature)!=feature_hash or sha(state)!=src['state20_sha256']:
            raise ValueError('feature or state changed')
        rows.append({'sample_id':sid,'feature_path':str(feature.relative_to(ROOT)),'feature_sha256':feature_hash})
        states.append({'sample_id':sid,'feature_sha256':feature_hash,'state_path':str(state.relative_to(ROOT)),
                       'state_sha256':src['state20_sha256'],'foot_velocity_squared_threshold':.002,
                       'encoding_variant':'sign_continuous_v1' if is_sign else 'legacy'})
    def write(path,value):
        with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    manifest=HERE/'state_manifest.json';write(manifest,{'records':states,'training_ready':False})
    cfg={'state_root':str(ROOT),'state_manifest':str(manifest),'state_urdf':str(URDF),'state_urdf_sha256':sha(URDF)}
    result=state_preflight(ROOT,rows,cfg)
    if not result['state_technical_checks_passed'] or result['training_ready'] is not False:
        raise ValueError('mixed preflight failed')
    if [r['encoding_variant'] for r in result['cases']].count('sign_continuous_v1')!=len(repairs):
        raise ValueError('encoding labels did not propagate')
    states[1]['encoding_variant']='legacy';bad=HERE/'wrong_encoder_manifest.json';write(bad,{'records':states})
    try:state_preflight(ROOT,rows,{**cfg,'state_manifest':str(bad)})
    except ValueError as e:
        if 'full feature re-encoding failed' not in str(e):raise
        rejected=str(e)
    else:raise ValueError('wrong encoding label was not rejected')
    out={'state_manifest_sha256':sha(manifest),'repair_report_sha256':sha(REPAIR/'report.json'),
         'independent_audit_sha256':sha(INDEPENDENT/'report.json'),'script_sha256':sha(Path(__file__)),
         'trainer_sha256':sha(HERE/'train_robot_prior.py'),'state_audit_sha256':sha(HERE/'audit_robot_states.py'),
         'legacy_cases':1,'sign_continuous_cases':len(repairs),'state_technical_checks_passed':True,
         'wrong_label_rejected':rejected,'training_ready':False,
         'limitations':['Sign-continuous encoding is a project approximation; legacy and corrected re-encoding share project dependencies.',
                        'No caption semantics, license, source identity, true timing or physical acceptance.']}
    write(HERE/'report.json',out);print(json.dumps(out))


if __name__=='__main__':main()

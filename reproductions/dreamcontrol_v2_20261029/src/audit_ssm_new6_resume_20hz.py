"""Recompute state/features and provenance for new SSM 20 Hz previews."""
import hashlib,json,traceback
from pathlib import Path
from audit_robot_states import audit_robot_states
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent;RUN=ROOT/'ssm_new6_anchor_resume_20261101';HUMAN=ROOT/'ssm_new6_human_20261101'
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    report=json.loads((HERE/'report.json').read_text());assert sha(HERE/'records.jsonl')==report['records_sha256']
    assert sha(RUN/'batch_audit.json')==report['source_batch_audit_sha256']
    records=[json.loads(x) for x in (HERE/'records.jsonl').read_text().splitlines()]
    assert len(records)==report['cases']==6 and len({x['id'] for x in records})==6
    results=[]
    with (HERE/'audit_rows.jsonl').open('x',buffering=1) as stream:
        for r in records:
            try:
                if r['status']!='technical_preview':raise ValueError('export failed: '+r.get('error',''))
                sid=r['id'];artifact=HERE/r['artifact'];assert sha(artifact)==r['artifact_sha256']
                assert sha(HUMAN/sid/'human_report.json')==r['human_report_sha256']
                assert sha(RUN/sid/'report.json')==r['solver_report_sha256']
                assert sha(RUN/sid/'audit_report.json')==r['solver_audit_sha256']
                evidence={'sample_id':sid,'feature_path':artifact.name,'feature_sha256':r['artifact_sha256'],
                    'state_path':artifact.name,'state_sha256':r['artifact_sha256'],'foot_velocity_squared_threshold':.002}
                audit=audit_robot_states([evidence],feature_root=HERE,state_root=HERE,urdf_path=URDF,
                    tolerance_rad=1e-6,fk_decode_tolerance_m=1e-4,reencoding_tolerance=1e-5)
                if not audit['state_technical_checks_passed']:raise ValueError('URDF joint-limit check failed')
                out={'id':sid,'status':'technical_checks_passed','state_check':audit['cases'][0],'training_ready':False}
            except Exception as e:
                out={'id':r['id'],'status':'failed','error':repr(e),'traceback':traceback.format_exc(),'training_ready':False}
            results.append(out);stream.write(json.dumps(out,allow_nan=False)+'\n');stream.flush()
    out={'source_report_sha256':sha(HERE/'report.json'),'audit_rows_sha256':sha(HERE/'audit_rows.jsonl'),
        'cases':len(results),'technical_passed':sum(x['status']=='technical_checks_passed' for x in results),
        'failed':[x['id'] for x in results if x['status']=='failed'],'training_ready':False,
        'limitations':['Same FK and encoder; filter and optimizer not independently reexecuted.',
            'No semantic, subject, clock, license or physical acceptance.']}
    with (HERE/'audit.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()

"""Isolated sign-continuity correction for audited encoding-only failures.

Joint-limit violations remain quarantined; no paper quality threshold or
source, semantic, physics or training acceptance is inferred.
"""
import hashlib,json,traceback
from pathlib import Path
import numpy as np
from feature263 import encode
from feature263_signfix import encode_signfixed
from g1_fk import URDFFK
from audit_robot_states import audit_robot_states
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
SOURCE=ROOT/'omni_variants_gpu_20261101';CLASSIFIED=ROOT/'omni_remaining_failures_v2_20261101';HERE=Path(__file__).resolve().parent
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    report=json.loads((SOURCE/'report.json').read_text());audit=json.loads((SOURCE/'audit_v2.json').read_text())
    classified=json.loads((CLASSIFIED/'report.json').read_text())
    if (sha(SOURCE/'report.json')!=audit['report_sha256'] or sha(SOURCE/'records.jsonl')!=report['records_sha256'] or
        sha(SOURCE/'audit_rows_v2.jsonl')!=audit['audit_rows_sha256'] or classified['full_audit_sha256']!=sha(SOURCE/'audit_v2.json') or
        sha(CLASSIFIED/'records.jsonl')!=classified['records_sha256']):
        raise ValueError('source audit or classification changed')
    source={r['sample_id']:r for r in map(json.loads,(SOURCE/'records.jsonl').read_text().splitlines())}
    failed=[r for r in map(json.loads,(CLASSIFIED/'records.jsonl').read_text().splitlines())]
    candidates=[r for r in failed if r['category']=='encoding_fk_mismatch']
    robot=URDFFK(URDF);results=[]
    with (HERE/'records.jsonl').open('x',buffering=1) as log:
        for x in candidates:
            sid=x['sample_id'];r=source[sid]
            try:
                path=SOURCE/r['state20'];feat=SOURCE/r['feature']
                if sha(path)!=r['state20_sha256'] or sha(feat)!=r['feature_sha256']:raise ValueError('input changed')
                with np.load(path,allow_pickle=False) as d:
                    roots=d['root_wxyz_xyz'];q=d['joint_pos'];names=list(d['joint_names'])
                with np.load(feat,allow_pickle=False) as d:prior=d['features']
                points=robot.forward(roots[:,4:],roots[:,[1,2,3,0]],q,names)
                raw=encode(points,foot_velocity_squared_threshold=.002)
                if not np.array_equal(raw,prior):raise ValueError('source encoder changed')
                corrected=encode_signfixed(points,foot_velocity_squared_threshold=.002)
                changed=np.flatnonzero(prior[:,0]!=corrected[:,0])
                if not len(changed) or not np.array_equal(prior[:,1:],corrected[:,1:]):
                    raise ValueError('failure is not explained by root sign correction')
                out=HERE/(sid+'_features.npz')
                with out.open('xb') as f:np.savez_compressed(f,features=corrected)
                e={'sample_id':sid,'state_path':path.name,'state_sha256':r['state20_sha256'],
                   'feature_path':out.name,'feature_sha256':sha(out),'foot_velocity_squared_threshold':.002}
                check=audit_robot_states([e],feature_root=HERE,state_root=SOURCE,urdf_path=URDF,
                    tolerance_rad=1e-6,fk_decode_tolerance_m=1e-4,reencoding_tolerance=1e-5,reencode=encode_signfixed)
                if not check['state_technical_checks_passed']:raise ValueError('joint limits failed after sign correction')
                row={'sample_id':sid,'status':'technical_preview','parent_state_sha256':sha(path),
                    'parent_feature_sha256':sha(feat),'feature':out.name,'feature_sha256':sha(out),
                    'changed_angular_rows':changed.tolist(),'state_check':check['cases'][0],
                    'caption_accepted':False,'training_ready':False}
            except Exception as e:
                row={'sample_id':sid,'status':'failed','error':repr(e),'traceback':traceback.format_exc(),'training_ready':False}
            results.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
    result={'source_report_sha256':sha(SOURCE/'report.json'),'source_audit_sha256':sha(SOURCE/'audit_v2.json'),
        'classification_sha256':sha(CLASSIFIED/'report.json'),
        'script_sha256':sha(Path(__file__)),'module_sha256':sha(HERE/'feature263_signfix.py'),'rows_sha256':sha(HERE/'records.jsonl'),
        'encoding_failure_candidates':len(candidates),'technical_previews':sum(x['status']=='technical_preview' for x in results),
        'failed':[x['sample_id'] for x in results if x['status']=='failed'],
        'other_joint_limit_failures_left_quarantined':len(failed)-len(candidates),'training_ready':False,
        'limitations':['Alternative root angular encoding is a project approximation, not author v2.',
        'FK, feature extraction and decoding share project dependencies.',
        'No semantic, license, source clock or physical quality acceptance.']}
    with (HERE/'report.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()

"""Audit BMLmovi robot preview bindings, nominal time and technical states."""
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
from audit_robot_states import audit_robot_states

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PREVIEW=ROOT/'bmlmovi_full_robot_263_preview_20261102'
HUMAN=ROOT/'bmlmovi_full_human_20261102'
WORK=ROOT/'bmlmovi_full_robot_work_20261102'
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
TEXTS=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D/texts.zip')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    report_path=PREVIEW/'report.json';report=json.loads(report_path.read_text())
    audit_path=WORK/'robot_windows_audit_20261102.json'
    if sha(audit_path)!=report['window_audit_sha256']:raise ValueError('window audit changed')
    human_report=json.loads((HUMAN/'report.json').read_text())
    human_rows={r['id']:r for r in map(json.loads,(HUMAN/'records.jsonl').read_text().splitlines())}
    if sha(HUMAN/'records.jsonl')!=human_report['records_sha256'] or sha(TEXTS)!=human_report['caption_archive_sha256']:
        raise ValueError('human records or caption archive changed')
    results=[]
    for row in report['cases']:
        sid=row['id']
        if row['status']!='technical_preview':
            results.append({'id':sid,'status':'quarantined','reason':row.get('reason')});continue
        source=HUMAN/(sid+'.npz');output=PREVIEW/row['artifact']
        if sha(source)!=row['human_sha256'] or sha(output)!=row['output_sha256']:
            raise ValueError('source or output hash mismatch')
        original=human_rows[sid]
        if (row['caption_member']!=original['caption_member'] or row['caption_sha256']!=original['caption_sha256']
            or row['nominal_index_interval']!=original['nominal_index_interval'] or row['auxiliary_split']!=original['auxiliary_split']
            or row['official_split']!=original['official_split'] or row['caption_accepted'] is not False):
            raise ValueError('caption or split binding mismatch')
        with zipfile.ZipFile(TEXTS) as captions:
            if hashlib.sha256(captions.read(row['caption_member'])).hexdigest()!=row['caption_sha256']:
                raise ValueError('caption archive member hash mismatch')
        with np.load(source,allow_pickle=False) as s, np.load(output,allow_pickle=False) as r:
            source_frames=s['source_frames'].copy();fps=float(s['declared_fps'])
            indices=r['source_frames'].copy();times=r['sample_times'].copy()
            if (fps!=120. or not np.array_equal(source_frames,np.arange(len(source_frames)))
                or not np.array_equal(indices,np.arange(1,len(source_frames),6))
                or not np.array_equal(times,indices.astype(np.float64)/fps)):
                raise ValueError('nominal time or object source-frame mismatch')
        evidence={'sample_id':sid,'feature_path':row['artifact'],'state_path':row['artifact'],
                  'feature_sha256':row['output_sha256'],'state_sha256':row['output_sha256'],
                  'foot_velocity_squared_threshold':.002}
        try:
            check=audit_robot_states([evidence],feature_root=PREVIEW,state_root=PREVIEW,urdf_path=URDF,
                        tolerance_rad=1e-6,fk_decode_tolerance_m=1e-4,reencoding_tolerance=1e-5)
            result={'id':sid,'status':'technical_checked','passed':check['state_technical_checks_passed'],
                    'details':check['cases'][0],'source_sha256':sha(source),'output_sha256':sha(output),
                    'nominal_source_indices_checked':True}
        except (ValueError,RuntimeError) as exc:
            result={'id':sid,'status':'technical_failed','passed':False,'error':str(exc)}
        results.append(result)
    out={'cases':results,'source_report_sha256':sha(report_path),'window_audit_sha256':sha(audit_path),
         'script_sha256':sha(Path(__file__)),'training_ready':False,
         'limitations':['Nominal source frame alignment does not prove actual clock, source identity or contact.',
             'FK, decoder and feature reencoding reuse experimental components.',
             'Quaternion FIR/resampling, physical dynamics and source caption semantics are not independently verified.']}
    with (PREVIEW/'audit.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({'cases':len(results),'technical_passed':sum(r.get('passed') is True for r in results),
                      'training_ready':False}))


if __name__=='__main__':main()

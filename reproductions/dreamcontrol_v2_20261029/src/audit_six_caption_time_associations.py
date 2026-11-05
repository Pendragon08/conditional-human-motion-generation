"""Compute explicit time associations for six unaccepted preview records."""
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
from caption_interval_alignment import associate_caption

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HUMANML=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D')
PREVIEW=ROOT/'six_candidate_technical_preview_20261031'
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    manifest=PREVIEW/'experimental_manifest_not_for_training.json'
    prior=json.loads((PREVIEW/'independent_audit.json').read_text())
    if sha(manifest)!=prior['manifest_sha256']:raise ValueError('preview changed')
    view=json.loads((ROOT/'source_grouping_v3_20261031/six_source_preview.json').read_text())
    lookup={r['sample_id']:r for r in view['records']};out=[]
    with zipfile.ZipFile(HUMANML/'texts.zip') as archive:
        for rec in json.loads(manifest.read_text())['records']:
            name=rec['sample_id'];ref=lookup[name];feature_file=PREVIEW/rec['feature_path']
            if sha(feature_file)!=rec['feature_sha256']:raise ValueError('feature changed')
            with np.load(feature_file,allow_pickle=False) as data:
                if name.startswith('BML_'):feature=data['source_times'].copy()
                else:feature=data['time_grid'].copy()
                n=len(data['features'])
            if len(feature)!=n+1:raise ValueError('feature endpoint mismatch')
            if name.startswith('BML_'):
                annotation=ref['candidate_clip_annotations'][0];clipid=annotation['clip_id']
                start,stop=annotation['index_slice'];clip=np.arange(start,stop,dtype=float)/20.
                basis='declared 120Hz plus conditional official index mapping'
            else:
                clipid=name
                folder=(ROOT/'humanml_mirror_time_witness_v2_20261030' if name=='000008'
                        else ROOT/'ssm_three_human_candidates_20261031')
                with np.load(folder/(name+'_joint_witness.npz'),allow_pickle=False) as d:clip=d['source_times'].copy()
                basis='reference-derived SSM sample times; source model identity not established'
            data=archive.read('texts/'+clipid+'.txt');lines=[line.split('#') for line in data.decode('utf8').splitlines()]
            selected=[x for x in lines if len(x)==4 and x[0]==rec['caption']]
            if len(selected)!=1:raise ValueError('caption source ambiguous')
            x=selected[0]
            association=associate_caption(clip_sample_times=clip,feature_sample_times=feature,
                    start_tag=x[2],end_tag=x[3],timed_tag_convention='unknown')
            if association['caption_accepted'] is not False:raise ValueError('unexpected semantic acceptance')
            out.append({'sample_id':name,'caption_clip_id':clipid,'caption_member_sha256':hashlib.sha256(data).hexdigest(),
                        'raw_tags':x[2:],'clip_sample_count':len(clip),'feature_rows':n,
                        'clip_first_last_s':[float(clip[0]),float(clip[-1])],
                        'feature_first_last_s':[float(feature[0]),float(feature[-1])],
                        'time_basis':basis,'association':association})
    result={'manifest_sha256':sha(manifest),'preview_audit_sha256':sha(PREVIEW/'independent_audit.json'),
            'module_sha256':sha(HERE/'caption_interval_alignment.py'),'cases':out,'training_ready':False,
            'limitations':['Time association is conditional; source FPS and SMPL-X identity are unproven.',
                           '0/0 is a full-clip convention, not independent semantic coverage.',
                           'Timed tags require a separately established interpretation.']}
    with (HERE/'report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'cases':len(out),'full_annotation_coverage':sum(x['association'].get('feature_covers_full_annotation_sample_extent',False) for x in out),
                      'training_ready':False}),flush=True)


if __name__=='__main__':main()

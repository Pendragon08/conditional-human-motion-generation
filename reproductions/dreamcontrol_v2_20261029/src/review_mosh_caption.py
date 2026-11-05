"""Record bounded visual support for an original MoSh caption on the cropped motion."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'six_family_disjoint_preview_v2_20261101'
ASSOC=ROOT/'eight_caption_alignment_20261031'
REVIEW=ROOT/'mosh_all_frames_review_20261101'
ZIP=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D/texts.zip')
HERE=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    view=json.loads((REVIEW/'report.json').read_text()); movement=json.loads((REVIEW/'wrist_motion.json').read_text())
    assert movement['review_sha256']==sha(REVIEW/'report.json') and movement['candidate_sha256']==view['candidate_sha256']
    assert all(sha(REVIEW/x['path'])==x['sha256'] for x in view['pages'])
    covered=[]
    for x in view['pages']:covered.extend(range(x['first_frame'],x['last_frame']+1))
    assert covered==list(range(view['frames']))==list(range(149))
    assoc=json.loads((ASSOC/'caption_association.json').read_text())
    assert sha(ASSOC/'caption_association.json')==json.loads((ASSOC/'independent_audit.json').read_text())['association_sha256']
    item=next(x for x in assoc['cases'] if x['sample_id']=='MOSH_000798_crop')
    records=json.loads((BASE/'manifest.json').read_text())['records'];states=json.loads((BASE/'state_manifest.json').read_text())['records']
    rec=next(x for x in records if x['sample_id']=='MOSH_000798_crop');state=next(x for x in states if x['sample_id']==rec['sample_id'])
    assert sha(BASE/rec['feature_path'])==rec['feature_sha256']==view['candidate_sha256']
    assert sha(BASE/state['state_path'])==state['state_sha256']
    with zipfile.ZipFile(ZIP) as z:raw=z.read('texts/000798.txt')
    assert hashlib.sha256(raw).hexdigest()==item['caption_member_sha256']
    parts=raw.decode('utf8').splitlines()[0].split('#'); caption=item['captions'][0]
    assert parts[0]==caption['caption'] and parts[2:4]==caption['raw_tags']==['0.0','0.0']
    association=caption['association'];assert association['status']=='associated'
    with np.load(BASE/state['state_path'],allow_pickle=False) as f:t=f['sample_times'].copy()
    with np.load(BASE/rec['feature_path'],allow_pickle=False) as f:features=f['features']
    assert len(t)==len(features)+1==149 and association['matching_feature_intervals']==list(range(len(features)))
    src=[association['annotation_start_time'],association['annotation_end_time']];sample=[float(t[0]),float(t[-1])]
    assert src[0]<=sample[0]<sample[1]<=src[1]
    out={'sample_id':rec['sample_id'],'caption':parts[0],'caption_member_sha256':hashlib.sha256(raw).hexdigest(),
         'caption_archive_sha256':sha(ZIP),'selected_line':0,'tags':parts[2:4],
         'candidate_sha256':rec['feature_sha256'],'state_sha256':state['state_sha256'],
         'source_video_sha256':view['video_sha256'],'all_frame_review_sha256':sha(REVIEW/'report.json'),
         'wrist_measurements_sha256':sha(REVIEW/'wrist_motion.json'),'sample_range_s':sample,'source_range_s':src,
         'excluded_head_tail_s':[sample[0]-src[0],src[1]-sample[1]],
         'manual_semantic_assessment':{'status':'supported_for_this_exported_motion',
            'reviewer':'Codex assistant; every decoded comparison frame in two orthographic views',
            'reason':'Both fitted human and retargeted robot move both arms and wrists in a dance-like sequence. Dance is a subjective visual interpretation; the 22-point skeleton does not verify fingers.',
            'semantic_accepted_for_exported_samples':True},'training_ready':False,
         'script_sha256':sha(Path(__file__)),
         'limitations':['First 1.05 nominal seconds were removed; this review applies only to the retained crop.',
            'Dancing is a visual interpretation, not objectively verified intent or motion-class truth.',
            'Only wrist points are available; hand and finger movement are not independently verified.',
            'Source identity, real FPS, licensing chain, body-model equivalence, ground and physical quality remain unaccepted.']}
    with (HERE/'semantic_review.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({'semantic_supported':True,'sample_id':rec['sample_id'],'training_ready':False}))


if __name__=='__main__':main()

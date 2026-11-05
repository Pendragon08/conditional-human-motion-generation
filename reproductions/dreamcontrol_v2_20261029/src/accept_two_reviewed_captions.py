"""Record two bounded manual caption reviews; never mark datasets train-ready."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'six_family_disjoint_preview_v2_20261101'
ASSOC=ROOT/'eight_caption_alignment_20261031'
SHEETS=ROOT/'five_all_frames_review_20261101'
VIDEOS=ROOT/'five_caption_review_videos_20261031'
ZIP=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D/texts.zip')
HERE=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    sheet=json.loads((SHEETS/'report.json').read_text())
    video=json.loads((VIDEOS/'report.json').read_text())
    assert sha(VIDEOS/'report.json')==sheet['video_report_sha256']
    association=json.loads((ASSOC/'caption_association.json').read_text())
    assert sha(ASSOC/'caption_association.json')==json.loads((ASSOC/'independent_audit.json').read_text())['association_sha256']
    records={r['sample_id']:r for r in json.loads((BASE/'manifest.json').read_text())['records']}
    states={r['sample_id']:r for r in json.loads((BASE/'state_manifest.json').read_text())['records']}
    all_reviewed=[]
    for sid,clip,why in [
        ('000008','000008','Both fitted human and robot bend their knees, raise both arms, and then rise; this statement does not require a speed or contact claim.'),
        ('ACCAD_004347','004347','Both fitted human and robot hold their arms extended sideways during the nearly static calibration pose.')]:
        evidence=next(x for x in sheet['cases'] if x['sample_id']==sid)
        prior=next(x for x in video['cases'] if x['sample_id']==sid)
        matching=next(x for x in association['cases'] if x['sample_id']==sid)
        r=records[sid]; s=states[sid]; feature=BASE/r['feature_path'];state=BASE/s['state_path']
        assert sha(feature)==r['feature_sha256']==evidence['feature_sha256'] and sha(state)==s['state_sha256']
        assert sha(VIDEOS/(sid+'.mp4'))==prior['video_sha256']==evidence['video_sha256']
        assert all(sha(SHEETS/p['path'])==p['sha256'] for p in evidence['pages'])
        coverage=[]
        for p in evidence['pages']:coverage.extend(range(p['first_frame'],p['last_frame']+1))
        assert coverage==list(range(evidence['frames']))
        with zipfile.ZipFile(ZIP) as z:raw=z.read('texts/'+clip+'.txt')
        assert hashlib.sha256(raw).hexdigest()==matching['caption_member_sha256']==prior['caption_member_sha256']
        parts=raw.decode('utf8').splitlines()[1].split('#')
        caption=matching['captions'][1]
        assert parts[0]==caption['caption'] and parts[2:4]==caption['raw_tags']==['0.0','0.0']
        assoc=caption['association'];assert assoc['status']=='associated'
        with np.load(state,allow_pickle=False) as f:times=f['sample_times'].copy()
        with np.load(feature,allow_pickle=False) as f:features=f['features']
        assert len(times)==len(features)+1 and assoc['matching_feature_intervals']==list(range(len(features)))
        src=[assoc['annotation_start_time'],assoc['annotation_end_time']];sample=[float(times[0]),float(times[-1])]
        assert src[0]<=sample[0]<sample[-1]<=src[1]+1e-8
        receipt={'sample_id':sid,'caption':parts[0],'caption_member_sha256':hashlib.sha256(raw).hexdigest(),
            'caption_archive_sha256':sha(ZIP),'selected_line':1,'tags':parts[2:4],
            'candidate_sha256':sha(feature),'state_sha256':sha(state),'source_video_sha256':sha(VIDEOS/(sid+'.mp4')),
            'all_frame_sheet_report_sha256':sha(SHEETS/'report.json'),'prior_video_report_sha256':sha(VIDEOS/'report.json'),
            'sample_range_s':sample,'source_range_s':src,'excluded_head_tail_s':[sample[0]-src[0],src[1]-sample[1]],
            'manual_semantic_assessment':{'status':'supported_for_this_exported_motion',
                'reviewer':'Codex assistant; every decoded comparison frame in two views',
                'reason':why,'semantic_accepted_for_exported_samples':True},'training_ready':False,
            'script_sha256':sha(Path(__file__)),
            'limitations':['Visual judgment of approximate fitted human and retargeted robot, not independently labelled truth.',
                'Full-clip annotation is restricted to the exported time range.',
                'No source identity, FPS, model equivalence, license, ground, contact or physical acceptance.',
                'Near-static calibration pose is not sufficient validation of motion generalization.']}
        path=HERE/(sid+'_review.json')
        with path.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
        all_reviewed.append({'sample_id':sid,'receipt_sha256':sha(path),'caption':parts[0],'frames':len(times)})
    with (HERE/'report.json').open('x') as f:json.dump({'cases':all_reviewed,'training_ready':False},f,indent=2);f.write('\n')
    print(json.dumps({'reviewed':len(all_reviewed),'training_ready':False}))


if __name__=='__main__':main()

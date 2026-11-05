"""Record one manual motion-only annotation, not original GRAB text."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
REVIEW=ROOT/'grab_s3_weak_label_review_20261101';MEASURE=ROOT/'grab_s3_arm_caption_candidate_20261101'
BRIDGE=ROOT/'grab_robot_feature_bridge_20261031';HERE=Path(__file__).resolve().parent
FK=ROOT/'resume_state_audit_20261031/g1_fk.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    review=json.loads((REVIEW/'report.json').read_text());audit=json.loads((REVIEW/'audit.json').read_text())
    measure=json.loads((MEASURE/'report.json').read_text());feat=BRIDGE/'s3_features.npz'
    assert sha(REVIEW/'report.json')==audit['report_sha256']==measure['source_review_sha256']
    assert sha(REVIEW/'audit.json')==measure['source_audit_sha256'] and sha(feat)==review['bridge_sha256']==measure['robot_sha256']
    for fig in review['figures']:assert sha(REVIEW/fig['path'])==fig['sha256']
    with np.load(feat,allow_pickle=False) as f:frames=f['source_frames'].copy();features=f['features'].copy()
    with np.load(MEASURE/'series.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(frames,f['source_frames'])
        human=f['human_right_wrist_above_shoulder_m'].copy();robot=f['robot_right_wrist_above_shoulder_m'].copy()
    assert sha(MEASURE/'series.npz')==measure['series_sha256'] and len(frames)==20 and features.shape==(19,263)
    tree=ast.parse(FK.read_text())
    assignment=next(x for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(y,ast.Name) and y.id=='PAPER_22_LINKS' for y in x.targets))
    links=ast.literal_eval(assignment.value)
    assert links[17]=='right_shoulder_pitch_link' and links[21]=='right_wrist_yaw_link'
    assert human[-1]<human[0] and robot[-1]<robot[0]
    # This wording was chosen after viewing both full-frame comparison figures.
    # The recorded numerical displacement supports that limited visual claim.
    out={'sample_id':'GRAB_S3_apple_eat_window','caption':'a person lowers their right arm.',
         'caption_origin':'manual visual annotation by Codex; not provided by GRAB or Motion-X',
         'reviewer':'Codex assistant; all 20 source-sampled frames in two projections',
         'semantic_supported_for_exported_window':True,'source_member':review['source_member'],
         'source_archive_sha256':review['archive_sha256'],'feature_sha256':sha(feat),'human_sha256':review['human_sha256'],
         'review_sha256':sha(REVIEW/'report.json'),'review_audit_sha256':sha(REVIEW/'audit.json'),
         'measure_sha256':sha(MEASURE/'report.json'),'measure_series_sha256':sha(MEASURE/'series.npz'),
         'comparison_fk_sha256':sha(FK),'robot_right_shoulder_link':links[17],'robot_right_wrist_link':links[21],
         'script_sha256':sha(Path(__file__)),'source_frames':frames.tolist(),'feature_rows':len(features),
         'nominal_time_s':[float(frames[0]/120),float(frames[-1]/120)],
         'right_wrist_relative_to_shoulder_height_change_m':{'human':float(human[-1]-human[0]),'robot':float(robot[-1]-robot[0])},
         'training_ready':False,
         'limitations':['Manual annotation for this window; not a source caption and not a claim of eating.',
                        'The right wrist and shoulder link convention is the experiment mapping; skeletons are approximate.',
                        'No proof of real source timing, grasp, ground, dynamics, data independence or training eligibility.',
                        'Only 19 feature rows from one sequence; no training dataset or final quality acceptance.']}
    with (HERE/'annotation.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))
if __name__=='__main__':main()

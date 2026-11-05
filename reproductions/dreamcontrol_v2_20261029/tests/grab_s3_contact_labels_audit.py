"""Count left/right GRAB object-label evidence in the audited S3 time span."""
import hashlib,json,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
sys.path.insert(0,str(ROOT/'grab_s3_human_source_20261030'))
from grab_restricted_reader import read_grab_field

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    folder=ROOT/'grab_s3_object_wrist_audit_v2_20261030';report=json.loads((folder/'report.json').read_text())
    frames_path=folder/'frames.jsonl'
    if sha(frames_path)!=report['frames_sha256']:raise ValueError('frames hash mismatch')
    rows=[json.loads(x) for x in frames_path.read_text().splitlines()];frames=np.array([x['frame'] for x in rows])
    archive=ROOT/'data/GRAB/grab__s3.zip';rec=json.loads((ROOT/'grab_s3_human_source_20261030/report.json').read_text())
    if sha(archive)!=rec['archive_sha256'] or sha(archive)!=report['archive_sha256']:raise ValueError('archive mismatch')
    with zipfile.ZipFile(archive) as z:contact=read_grab_field(z,rec['source_member'],'contact')
    labels=np.asarray(contact['object'])[frames]
    if labels.ndim!=2 or labels.dtype.kind not in 'ui' or labels.min()<0 or labels.max()>55:raise ValueError('bad contact labels')
    left=(labels==21)|((labels>=26)&(labels<=40));right=(labels==22)|((labels>=41)&(labels<=55))
    per_frame=[]
    for i,row in enumerate(rows):
        if int((labels[i]!=0).sum())!=row['object_labeled_vertices']:raise ValueError('contact count mismatch')
        per_frame.append({'frame':int(frames[i]),'left_hand_labeled_vertices':int(left[i].sum()),
                          'right_hand_labeled_vertices':int(right[i].sum())})
    counts={'left':int(np.any(left,axis=1).sum()),'right':int(np.any(right,axis=1).sum()),
            'both':int((np.any(left,axis=1)&np.any(right,axis=1)).sum())}
    metrics={}
    for side,mask,idx in [('left',left,0),('right',right,1)]:
        selected=np.any(mask,axis=1);metrics[side]={}
        for name in ('human','baseline_robot','anchored_robot'):
            values=np.array([r[name+'_wrist_to_object_vertex_m'][idx] for r in rows])
            metrics[side][name]=float(np.median(values[selected])) if selected.any() else None
    out=Path(__file__).with_name('frames.jsonl')
    with out.open('x',encoding='utf8') as f:
        for row in per_frame:f.write(json.dumps(row)+'\n')
    doc={'source_report_sha256':sha(folder/'report.json'),'archive_sha256':sha(archive),'script_sha256':sha(__file__),
         'frame_labels_sha256':sha(out),'mapping_reference':'https://github.com/otaheri/GRAB/blob/4dab3211fae4fc5b8eb6ab86246ccc3a42d8f611/tools/utils.py',
         'left_hand_label_ids':[21]+list(range(26,41)),'right_hand_label_ids':[22]+list(range(41,56)),
         'frames':len(rows),'frames_with_hand_labels':counts,'median_wrist_to_any_object_vertex_m_in_side_labeled_frames':metrics,
         'warning':'Body-part labels derived from proximity and other criteria, not measured force; wrist-to-object-vertex distances do not measure hand surfaces. Dataset-specific indexing and SMPL-X version require full validation. No robot physics or training acceptance.'}
    with Path(__file__).with_name('report.json').open('x',encoding='utf8') as f:json.dump(doc,f,indent=2,allow_nan=False)
    print(json.dumps(doc),flush=True)
if __name__=='__main__':main()

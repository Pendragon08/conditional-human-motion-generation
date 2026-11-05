"""Compare open/closed geometric hand proxy on identical object/robot frames."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    a=ROOT/'grab_s3_mujoco_object_20261030';b=ROOT/'grab_s3_mujoco_object_closed_20261030'
    reports=[json.loads((p/'report.json').read_text()) for p in (a,b)]
    for key in ('source_sha256','archive_sha256','mesh_sha256','scene_sha256','mujoco_version','right_hand_labeled_frames','robot_hand_collision_geoms'):
        if reports[0][key]!=reports[1][key]:raise ValueError('mismatch '+key)
    rows=[]
    for p,report in zip((a,b),reports):
        if sha(p/'frames.jsonl')!=report['frames_sha256']:raise ValueError('frame hash mismatch')
        values=[json.loads(x) for x in (p/'frames.jsonl').read_text().splitlines()]
        rows.append({(x['frame'],x['method']):x for x in values})
    if set(rows[0])!=set(rows[1]) or len(rows[0])!=236:raise ValueError('frame coverage mismatch')
    summary={}
    for method in ('baseline','anchored'):
        keys=sorted(k for k in rows[0] if k[1]==method)
        for k in keys:
            for key in ('right_hand_labeled_vertices','left_hand_labeled_vertices'):
                if rows[0][k][key]!=rows[1][k][key]:raise ValueError('label mismatch')
        keys=[k for k in keys if rows[0][k]['right_hand_labeled_vertices']>0]
        x=np.array([rows[0][k]['right_min_signed_distance_m'] for k in keys]);y=np.array([rows[1][k]['right_min_signed_distance_m'] for k in keys])
        summary[method]={'frames':len(keys),'open_median_m':float(np.median(x)),
            'closed_median_m':float(np.median(y)),'median_closed_minus_open_m':float(np.median(y-x)),
            'closed_closer_frame_count':int((y<x).sum()),'open_closer_frame_count':int((x<y).sum()),
            'open_nonpositive_count':int((x<=0).sum()),'closed_nonpositive_count':int((y<=0).sum())}
    doc={'open_report_sha256':sha(a/'report.json'),'closed_report_sha256':sha(b/'report.json'),
         'script_sha256':sha(__file__),'summary':summary,
         'warning':'Static geometry query on one subject/action; closed finger targets from DreamControl v1 Pick. Convex apple hull and assumed robot/object alignment; no force, dynamics, hand-object control or Isaac Lab. Changes in distance do not prove grasp or data acceptance.'}
    with (HERE/'comparison.json').open('x',encoding='utf8') as f:json.dump(doc,f,indent=2,allow_nan=False)
    print(json.dumps(doc),flush=True)
if __name__=='__main__':main()

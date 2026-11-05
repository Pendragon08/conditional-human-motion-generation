"""Independent provenance and complete frame verification of existing S5 run."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
RUN=ROOT/'pyroki_grab_s5_cup_windowed_20261030';SOURCE=ROOT/'grab_s5_cup_human_source_20261030';BASE=ROOT/'pyroki_grab_s1_pour_windowed_20261030'
OUT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    manifest=json.loads((RUN/'manifest.json').read_text());base=json.loads((BASE/'manifest.json').read_text());source=json.loads((SOURCE/'report.json').read_text());r=json.loads((RUN/'report.json').read_text());a=json.loads((RUN/'audit_report.json').read_text())
    assert sha(SOURCE/'report.json')==manifest['source_report_sha256'] and sha(SOURCE/'human_source.npz')==manifest['source_artifact_sha256']==source['artifact_sha256']==r['source_sha256']==a['source_sha256']
    assert sha(BASE/'manifest.json')==manifest['base_manifest_sha256'] and sha(RUN/'report.json')==a['report_sha256'] and sha(RUN/'windows.jsonl')==r['rows_sha256']==a['rows_sha256']
    old="ROOT/'grab_s1_pour_human_source_20261030/human_source.npz'";new="ROOT/'grab_s5_cup_human_source_20261030/human_source.npz'"
    for file,key in [('pyroki_grab_windowed_pilot.py','s1_driver_sha256'),('audit_pyroki_windowed_pilot.py','s1_audit_sha256')]:
        before=(BASE/file).read_text();after=(RUN/file).read_text();assert sha(BASE/file)==base[key] and before.count(old)==1 and before.replace(old,new)==after
        assert sha(RUN/file)==manifest['adapted_files'][file]
    rows=[json.loads(x) for x in (RUN/'windows.jsonl').read_text().splitlines()];assert len(rows)==29 and r['window_count']==a['windows_verified']==29
    seen=set();joint_names=None
    for i,row in enumerate(rows):
        start=min(309+4*i,420)
        assert row['window']==i and row['frames']==list(range(start,start+7))
        file=RUN/row['artifact'];assert file.parent==RUN and file.name==f'window_{i:02d}.npz' and sha(file)==row['artifact_sha256']
        with np.load(file,allow_pickle=False) as d:
            assert d['frames'].tolist()==row['frames'] and float(d['fps'])==120 and d['root_wxyz_xyz'].shape==(7,7) and d['joint_pos'].shape==(7,27)
            assert np.isfinite(d['root_wxyz_xyz']).all() and np.isfinite(d['joint_pos']).all()
            names=d['joint_names'].tolist();joint_names=names if joint_names is None else joint_names;assert names==joint_names
        seen.update(row['frames'])
    assert seen==set(range(309,427)) and len(seen)==a['unique_frames']==118 and a['max_joint_limit_violation_rad']==0
    out={'windows':29,'unique_frames':len(seen),'first':min(seen),'last':max(seen),'source_report_sha256':sha(SOURCE/'report.json'),'window_report_sha256':sha(RUN/'report.json'),'audit_report_sha256':sha(RUN/'audit_report.json'),'manifest_sha256':sha(RUN/'manifest.json'),'rows_sha256':sha(RUN/'windows.jsonl'),'script_sha256':sha(Path(__file__)),
         'limitations':'Checks provenance, frame IDs, finiteness and hash references; relies on prior auditor for kinematic/limit metrics, does not establish true source FPS, floor, contacts, dynamics, or v2 training acceptance.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2)
    print(json.dumps(out))
if __name__=='__main__':main()

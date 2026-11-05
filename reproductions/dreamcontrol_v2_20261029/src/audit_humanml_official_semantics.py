"""Verify public HumanML source semantics and trace a bounded index witness."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PUBLIC = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public')
OUT = Path(__file__).resolve().parent
COMMIT = '9176e8fb446b71c7d2a725eb5cf6fec1ae3b3c23'
PINS = {'index.csv':'80226910d7259655ab5f411fd807b7651037341b4806fde025946a74d62b1065',
        'all.txt':'49fa084e75a7896dde9f6af735979c9430d19feb84bed2d36a39024baf096663',
        'train.txt':'e3e6975099b19964ee085db3e1d1160365a27e275f4f2b3d37c8ef24cccf4ce9',
        'val.txt':'effa9e866f7c2bd3e5bb518600f95a5fd780a2dcf72c3cdb1e0caf5c93051e17',
        'test.txt':'b89bc7df558df6246840702d0976a5d37ecbc4f055642b5c99e509a2d85ac12e'}
NOTEBOOK_HASH = 'fc1ffb4ed2f0e808d94fd434a2e75a02099aceed531b35dd4fae7c9e068ffe6f'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    notebook=OUT/'raw_pose_processing.ipynb'
    if sha(notebook)!=NOTEBOOK_HASH:raise ValueError('notebook changed')
    sources = PUBLIC/'index.csv'
    for name,expected in PINS.items():
        path = sources if name=='index.csv' else PUBLIC/'HumanML3D'/name
        if sha(path)!=expected:raise ValueError('public file differs: '+name)
    cells=json.loads(notebook.read_text(encoding='utf8'))['cells']
    code='\n'.join(''.join(c.get('source',[])) for c in cells if c.get('cell_type')=='code')
    checks=['down_sample = int(fps / ex_fps)', "bdata_poses = bdata['poses'][::down_sample,...]",
            'data = data[start_frame:end_frame]', 'data_m = swap_left_right(data)',
            "np.save(pjoin(save_dir, 'M'+new_name), data_m)", 'data = data[3*fps:]',
            'data = data[1*fps:]', 'data = data[int(0.5*fps):]', 'data[..., 0] *= -1']
    if not all(s in code for s in checks):raise ValueError('expected source operations missing')
    with sources.open(newline='',encoding='utf8') as f:
        index = {r['new_name'].removesuffix('.npy'): r for r in csv.DictReader(f)}
    a,b=index['007089'],index['012633']
    if a['source_path']!=b['source_path'] or (a['start_frame'],a['end_frame'],b['start_frame'],b['end_frame'])!=('7','207','179','379'):
        raise ValueError('index witness changed')
    splits={s:set((PUBLIC/'HumanML3D'/f'{s}.txt').read_text().splitlines()) for s in ('train','val','test')}
    if '007089' not in splits['test'] or '012633' not in splits['val'] or 'M007089' not in splits['test'] or 'M012633' not in splits['val']:
        raise ValueError('witness official split changed')
    mapping=ROOT/'humanml_raw_mapping_v2_20261030/mapping.jsonl'
    mapping_hash='358875ac6aca9e976086778fe317f27f64134e1fafaff51581817c09f22da778'
    if sha(mapping)!=mapping_hash:raise ValueError('mapping changed')
    records=[json.loads(x) for x in mapping.read_text().splitlines()]
    witness=[r for r in records if r['motion_id'] in ('007089','012633')]
    if len(witness)!=2 or len({r['candidate_local_path'] for r in witness})!=1:
        raise ValueError('candidate mapping mismatch')
    candidate=Path(witness[0]['candidate_local_path'])
    # Candidate hash only: a file path/header match does not establish SMPL+H identity or physical FPS.
    if not candidate.is_file():raise ValueError('candidate no longer available')
    start=max(int(a['start_frame']),int(b['start_frame']))
    stop=min(int(a['end_frame']),int(b['end_frame']))
    if (start,stop)!=(179,207):raise ValueError('incorrect slice arithmetic')
    report={'official_repository':'https://github.com/EricGuo5513/HumanML3D', 'commit':COMMIT,
            'notebook_url':f'https://github.com/EricGuo5513/HumanML3D/blob/{COMMIT}/raw_pose_processing.ipynb',
            'notebook_sha256':NOTEBOOK_HASH,'index_and_splits_sha256':PINS,'mapping_sha256':mapping_hash,
            'script_sha256':sha(Path(__file__)),
            'verified_operations':{'source_downsampling':'integer floor fps/20 followed by stride slicing',
              'dataset_prefix_trims_nominal_samples':{'Eyes_Japan_Dataset':60,'MPI_HDM05':60,'TotalCapture':20,'MPI_Limits':20,'Transitions_mocap':10},
              'segmentation':'Python half-open [start_frame:end_frame] after prefix trim for non-HumanAct12',
              'mirroring':'reflection of coordinate 0 and swap left/right joints; output name prefixed M',
              'humanact12':'the shown notebook bypasses the AMASS-specific slice and initial coordinate flip'},
            'witness':{'dataset':'ACCAD','source_path':a['source_path'],'ids':{'007089':'test','012633':'val'},
              'source_index_slices':{'007089':[7,207],'012633':[179,379]},'shared_pose_data_sample_indices':[start,stop],
              'shared_sample_count':stop-start,'candidate_local_path':str(candidate),'candidate_sha256':sha(candidate),
              'candidate_source_identity_and_reference_fps_verified':False},
            'limitations':'Official public notebook and index/splits fixed by bytes; data files may have been generated by another version. Shared indexed pose_data samples follow from the notebook only if source length covers them; actual files, reference SMPL+H, mirrored pose values and physical time remain unverified. Candidate SMPL-X header is not proof of reference FPS. No training split changed.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,sort_keys=True)
    print(json.dumps(report,sort_keys=True),flush=True)

if __name__=='__main__':main()

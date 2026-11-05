"""Verify and copy only an intact contiguous PyRoki prefix to a new run.

No optimization runs here. Original files are never modified. A failed or
ambiguous prefix aborts rather than being silently skipped or overwritten.
"""
import hashlib,json,shutil,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
OLD=ROOT/'ssm_new6_anchor_gpu_v2_20261101';HUMAN=ROOT/'ssm_new6_human_20261101';HERE=Path(__file__).resolve().parent
URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
IDS=('000162','004028','006749','007129','008894','009466')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def copy_exact(src,dst):
    with src.open('rb') as i,dst.open('xb') as o:shutil.copyfileobj(i,o,1024*1024)
    if sha(src)!=sha(dst):raise RuntimeError('copy hash mismatch')
def main():
    if (OLD/'queue_status.txt').read_text().strip()!='solver_failed' or 'resource guard stopped:' not in (OLD/'solver.log').read_text()[-4096:]:
        raise ValueError('expected prior resource-guard stop')
    progress=[json.loads(x) for x in (OLD/'progress.jsonl').read_text().splitlines()]
    completed={x['id']:x for x in progress}
    if len(completed)!=len(progress) or not set(completed).issubset(IDS):raise ValueError('invalid completed sample list')
    limits={}
    for j in ET.parse(URDF).getroot().findall('joint'):
        if j.get('type')=='continuous':limits[j.get('name')]=(-np.inf,np.inf)
        elif j.get('type')=='revolute':
            x=j.find('limit');limits[j.get('name')]=(float(x.get('lower')),float(x.get('upper')))
    checked=[];partial_count=0
    for sid in IDS:
        src=HUMAN/sid;hr=json.loads((src/'human_report.json').read_text());ha=json.loads((src/'human_audit.json').read_text())
        if sha(src/'human_report.json')!=ha['human_report_sha256'] or sha(src/'human_rawfps.npz')!=ha['artifact_sha256']:
            raise ValueError('human audit mismatch')
        out=OLD/sid
        if not out.exists():
            if sid in completed:raise ValueError('completed directory missing')
            checked.append({'id':sid,'status':'not_started','window_count':0,'source_sha256':ha['artifact_sha256']});continue
        rows=[json.loads(x) for x in (out/'windows.jsonl').read_text().splitlines()]
        with np.load(src/'human_rawfps.npz',allow_pickle=False) as f:frames=f['source_frames'];times=f['source_times']
        starts=list(range(0,len(frames)-7,4))
        if starts[-1]!=len(frames)-8:starts.append(len(frames)-8)
        if not rows or len(rows)>len(starts):raise ValueError('invalid window count')
        if {x.name for x in out.glob('window_*.npz')}!={x['artifact'] for x in rows}:raise ValueError('unlogged/missing window')
        previous=None;maximum={'root_m':0.,'rotation_rad':0.,'joint_rad':0.};maxlimit=0.
        for i,(row,start) in enumerate(zip(rows,starts)):
            p=out/row['artifact'];idx=np.arange(start+1,start+8)
            if row['window']!=i or row['sample_indices']!=idx.tolist() or sha(p)!=row['artifact_sha256']:
                raise ValueError('window prefix or hash mismatch')
            with np.load(p,allow_pickle=False) as d:
                q=d['joint_pos'];root=d['root_wxyz_xyz'];names=[str(x) for x in d['joint_names']]
                if not np.array_equal(d['sample_indices'],idx) or not np.array_equal(d['source_frames'],frames[idx]) or not np.array_equal(d['source_times'],times[idx]):
                    raise ValueError('window frame identity mismatch')
            if q.shape!=(7,27) or root.shape!=(7,7) or set(names)!=set(limits) or not np.isfinite(q).all() or not np.isfinite(root).all() or not np.allclose(np.linalg.norm(root[:,:4],axis=1),1.,atol=1e-4,rtol=0):
                raise ValueError('invalid state')
            low=np.array([limits[x][0] for x in names]);high=np.array([limits[x][1] for x in names])
            maxlimit=max(maxlimit,float(np.maximum(np.maximum(low-q,q-high),0).max()))
            if previous is not None:
                common=np.intersect1d(previous['indices'],idx);a=np.searchsorted(previous['indices'],common);b=np.searchsorted(idx,common)
                ra=Rotation.from_quat(previous['root'][a][:,[1,2,3,0]]);rb=Rotation.from_quat(root[b][:,[1,2,3,0]])
                actual={'root_m':float(np.linalg.norm(previous['root'][a,4:]-root[b,4:],axis=1).max()),
                        'rotation_rad':float((ra.inv()*rb).magnitude().max()),'joint_rad':float(np.abs(previous['q'][a]-q[b]).max())}
                if row['overlap']['sample_indices']!=common.tolist() or any(abs(actual[k]-row['overlap'][k])>1e-6 for k in actual):
                    raise ValueError('overlap metadata differs')
                for k in maximum:maximum[k]=max(maximum[k],actual[k])
            previous={'indices':idx,'root':root,'q':q}
        if sid in completed:
            if len(rows)!=len(starts):raise ValueError('completed window coverage mismatch')
            report=json.loads((out/'report.json').read_text())
            if (sha(out/'report.json')!=completed[sid]['report_sha256'] or report['rows_sha256']!=sha(out/'windows.jsonl') or
                report['source_sha256']!=ha['artifact_sha256'] or report['urdf_sha256']!=sha(URDF) or
                report['script_sha256']!=sha(OLD/'pyroki_ssm_new6_anchor_gpu.py')):
                raise ValueError('completed provenance mismatch')
            state='complete'
        else:
            partial_count+=1;state='partial'
            if len(rows)>=len(starts) or (out/'report.json').exists():raise ValueError('invalid incomplete sample')
        checked.append({'id':sid,'status':state,'window_count':len(rows),'expected_windows':len(starts),
            'source_sha256':ha['artifact_sha256'],'rows_sha256':sha(out/'windows.jsonl'),
            'max_overlap':maximum,'max_joint_limit_excess_rad':maxlimit})
    if partial_count!=1:raise ValueError('expected exactly one partial sample')
    if shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('disk headroom unavailable')
    for item in checked:
        if item['status']=='not_started':continue
        sid=item['id'];src=OLD/sid;dst=HERE/sid;dst.mkdir(exist_ok=False)
        rows=[json.loads(x) for x in (src/'windows.jsonl').read_text().splitlines()]
        for name in ['windows.jsonl']+[r['artifact'] for r in rows]+(['report.json'] if item['status']=='complete' else []):
            copy_exact(src/name,dst/name)
    with (HERE/'progress.jsonl').open('x',buffering=1) as f:
        for row in progress:f.write(json.dumps(row)+'\n')
    report={'old_run':str(OLD),'old_script_sha256':sha(OLD/'pyroki_ssm_new6_anchor_gpu.py'),
        'old_status_sha256':sha(OLD/'queue_status.txt'),'old_progress_sha256':sha(OLD/'progress.jsonl'),
        'urdf_sha256':sha(URDF),'cases':checked,'training_ready':False,
        'limitations':['Only saved numerical prefix, hashes, source indexing, limits and overlaps checked.',
          'Same optimizer, model and reference FPS; no text semantic or physical acceptance.']}
    with (HERE/'resume_precheck.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'complete':len(completed),'partial':partial_count,'windows_verified':sum(x['window_count'] for x in checked),'training_ready':False}))
if __name__=='__main__':main()

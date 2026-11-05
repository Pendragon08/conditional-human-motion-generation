"""Read-only, per-subject GRAB frame/mesh/contact structure audit.

Only trusted verified GRAB archives are permitted. Restricted NumPy pickle
handling reduces but cannot eliminate hostile-pickle risk. A matching number of
vertices does not prove vertex order, contact force, physics or train readiness.
"""
import argparse,hashlib,io,json,pickle,zipfile
from collections import Counter
from pathlib import Path,PurePosixPath
import numpy as np
from grab_restricted_reader import read_object_npy

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
STAGE=ROOT/'grab_mesh_stage_20261030'
LIMIT=256*1024**2

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def ply_vertices(path):
    with path.open('rb') as f:
        if f.readline().strip()!=b'ply':raise ValueError('not PLY')
        count=None
        for _ in range(500):
            line=f.readline()
            if not line or len(line)>8192:raise ValueError('invalid PLY header')
            words=line.strip().split()
            if words[:2]==[b'element',b'vertex']:
                if count is not None or len(words)!=3:raise ValueError('invalid vertex count')
                count=int(words[2])
            if words==[b'end_header']:
                if count is None or count<1:raise ValueError('missing PLY vertices')
                return count
    raise ValueError('oversized PLY header')

def main():
    p=argparse.ArgumentParser();p.add_argument('--subject',required=True);p.add_argument('--archive-sha256',required=True);p.add_argument('--expected-count',type=int,required=True);p.add_argument('--start-index',type=int,default=0);p.add_argument('--limit',type=int);args=p.parse_args()
    if args.subject not in {f's{i}' for i in range(1,11)} or len(args.archive_sha256)!=64 or args.expected_count<1:raise ValueError('invalid inputs')
    if args.limit is not None and args.limit<1:raise ValueError('invalid limit')
    if args.start_index<0 or args.start_index>=args.expected_count:raise ValueError('invalid start index')
    archive=ROOT/'data/GRAB'/f'grab__{args.subject}.zip'
    if sha(archive)!=args.archive_sha256:raise ValueError('archive hash mismatch')
    stage_manifest=STAGE/'manifest.json';manifest_hash=sha(stage_manifest)
    asset_rows=json.loads(stage_manifest.read_text())['files'];assets={r['path']:r for r in asset_rows}
    if len(assets)!=len(asset_rows):raise ValueError('duplicate staged asset manifest entries')
    cache={}
    def verify_asset(ref,kind):
        path=PurePosixPath(ref)
        if path.is_absolute() or '..' in path.parts or path.parts[:2]!=('tools',kind):raise ValueError('unsafe asset reference')
        key=str(path)
        if key not in cache:
            expected=assets[key];real=(STAGE/key).resolve(strict=True)
            if not real.is_relative_to(STAGE.resolve()) or sha(real)!=expected['sha256'] or real.stat().st_size!=expected['bytes']:raise ValueError('staged asset hash mismatch')
            cache[key]=(ply_vertices(real),expected['sha256'])
        return cache[key]
    rows_path=Path(__file__).with_name('rows.jsonl');counts=Counter();meshes=set();subject_frames=0
    with rows_path.open('x',encoding='utf8') as out,zipfile.ZipFile(archive) as z:
        members=sorted(n for n in z.namelist() if n.endswith('.npz'))
        if len(members)!=args.expected_count or len(set(members))!=len(members):raise ValueError('unexpected archive sequence count')
        members=members[args.start_index:]
        if args.limit is not None:members=members[:args.limit]
        for member in members:
            try:
                info=z.getinfo(member)
                if info.file_size>LIMIT:raise ValueError('outer sequence exceeds 256MiB cap')
                raw=z.read(info)
                with np.load(io.BytesIO(raw),allow_pickle=False) as f:
                    frames=int(f['n_frames'].item());fps=float(f['framerate'].item());subject=str(f['sbj_id'].item());gender=str(f['gender'].item());pca=int(f['n_comps'].item())
                if frames<2 or not np.isfinite(fps) or fps<=0 or subject!=args.subject or gender not in ('male','female') or pca!=24:raise ValueError('source metadata mismatch')
                with zipfile.ZipFile(io.BytesIO(raw)) as inner:
                    fields={}
                    for name in ('body','object','contact'):
                        entry=inner.getinfo(name+'.npy')
                        if entry.file_size>LIMIT:raise ValueError(name+' exceeds 256MiB cap')
                        fields[name]=read_object_npy(inner.read(entry))
                body,obj,contact=fields['body'],fields['object'],fields['contact']
                template=body['vtemp'];mesh=obj['object_mesh']
                if template!=f'tools/subject_meshes/{gender}/{args.subject}.ply':raise ValueError('subject template identity mismatch')
                nbody,body_hash=verify_asset(template,'subject_meshes');nobject,object_hash=verify_asset(mesh,'object_meshes')
                bp=body['params'];op=obj['params']
                for key,dim in {'body_pose':63,'fullpose':165,'global_orient':3,'left_hand_pose':24,'right_hand_pose':24,'transl':3,'jaw_pose':3,'leye_pose':3,'reye_pose':3,'expression':10}.items():
                    if np.asarray(bp[key]).shape!=(frames,dim):raise ValueError('body '+key+' shape mismatch')
                for key in ('global_orient','transl'):
                    if np.asarray(op[key]).shape!=(frames,3):raise ValueError('object '+key+' shape mismatch')
                human=np.asarray(contact['body']);object_contact=np.asarray(contact['object'])
                if human.shape!=(frames,nbody) or object_contact.shape!=(frames,nobject) or human.dtype.kind not in 'biu' or object_contact.dtype.kind not in 'biu':raise ValueError('contact shape/type mismatch')
                ranges={'body':[int(human.min()),int(human.max())],'object':[int(object_contact.min()),int(object_contact.max())]}
                if min(ranges['body'][0],ranges['object'][0])<0 or max(ranges['body'][1],ranges['object'][1])>55:raise ValueError('contact label outside known range')
                rec={'member':member,'status':'ok','frames':frames,'fps':fps,'gender':gender,'template':template,'template_sha256':body_hash,'body_vertices':nbody,'mesh':mesh,'mesh_sha256':object_hash,'object_vertices':nobject,'contact_label_ranges':ranges}
                subject_frames+=frames;meshes.add(mesh);counts['ok']+=1
            except (ValueError,KeyError,TypeError,IndexError,OSError,EOFError,pickle.UnpicklingError,zipfile.BadZipFile) as exc:
                rec={'member':member,'status':'unresolved','error_type':type(exc).__name__,'error':str(exc)};counts['unresolved']+=1
            out.write(json.dumps(rec,allow_nan=False)+'\n');out.flush()
            print(json.dumps({'processed':counts['ok']+counts['unresolved'],'status':rec['status'],'member':member}),flush=True)
    if sha(archive)!=args.archive_sha256 or sha(stage_manifest)!=manifest_hash:raise RuntimeError('inputs changed during audit')
    report={'subject':args.subject,'archive_sha256':args.archive_sha256,'stage_manifest_sha256':manifest_hash,'script_sha256':sha(Path(__file__)),
            'reader_sha256':sha(Path(__file__).with_name('grab_restricted_reader.py')),'records_sha256':sha(rows_path),
            'expected_sequences':args.expected_count,'start_index':args.start_index,'scanned_sequences':sum(counts.values()),'counts':dict(counts),'unique_object_meshes_verified':len(meshes),'validated_frames':subject_frames,
            'limitations':'Verified trusted inputs, bounded reader and shape checks only. Oversize fields are unresolved, not corrupt. Matching counts do not prove vertex order, source framerate truth, actual contact, physics, task feasibility or training readiness.'}
    with Path(__file__).with_name('report.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()

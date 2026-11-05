"""Structure-audit all GRAB s2 sequences using the restricted object reader.

No template, object mesh or motion output is generated. Verified archive only;
this does not independently validate safety of hostile pickle, body modeling,
contact meaning, source FPS, legal rights, scene geometry or robot retargeting.
"""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import pickle
import zipfile
import numpy as np
from grab_restricted_reader import read_grab_field


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def asset_reference(value):
    if not isinstance(value,str) or not value: raise ValueError('empty asset reference')
    p=PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts: raise ValueError('unsafe relative asset path')
    return value


def main():
    path=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/data/GRAB/grab__s2.zip')
    expected='d77eb7d13cfb3e3d0b81f08ccf9d8d79320fa6541352e6e65173c56d4b01b9b6'
    if digest(path)!=expected: raise ValueError('archive hash mismatch')
    rows=[]; errors=[]; fps_counts=Counter(); templates=Counter(); objects=Counter()
    with zipfile.ZipFile(path) as archive:
        members=['s2/cubesmall_lift.npz','s2/piggybank_lift.npz','s2/stanfordbunny_lift.npz']
        for member in members:
            try:
                info=archive.getinfo(member)
                if info.file_size>128*1024**2: raise ValueError('outer sequence exceeds reader limit')
                with np.load(io.BytesIO(archive.read(info)),allow_pickle=False) as raw:
                    frames=int(raw['n_frames'].item()); fps=float(raw['framerate'].item())
                    subject=str(raw['sbj_id'].item()); gender=str(raw['gender'].item()); pca=int(raw['n_comps'].item())
                if frames<2 or fps!=120. or subject!='s2' or gender!='male' or pca!=24:
                    raise ValueError('unexpected frame rate, subject, gender or hand PCA')
                body=read_grab_field(archive,member,'body'); obj=read_grab_field(archive,member,'object')
                contact=read_grab_field(archive,member,'contact')
                body_params=body['params']; obj_params=obj['params']
                shapes={'body_pose':63,'fullpose':165,'global_orient':3,'left_hand_pose':24,
                        'right_hand_pose':24,'transl':3,'jaw_pose':3,'leye_pose':3,'reye_pose':3,'expression':10}
                for name,dim in shapes.items():
                    if not isinstance(body_params[name],np.ndarray) or body_params[name].shape!=(frames,dim):
                        raise ValueError(f'body {name} shape mismatch')
                for name in ('global_orient','transl'):
                    if not isinstance(obj_params[name],np.ndarray) or obj_params[name].shape!=(frames,3):
                        raise ValueError(f'object {name} shape mismatch')
                human_contact,object_contact=contact['body'],contact['object']
                if (not isinstance(human_contact,np.ndarray) or human_contact.shape!=(frames,10475) or
                    not isinstance(object_contact,np.ndarray) or object_contact.ndim!=2 or object_contact.shape[0]!=frames or
                    human_contact.dtype.kind not in 'biu' or object_contact.dtype.kind not in 'biu'):
                    raise ValueError('contact array structure mismatch')
                template=asset_reference(body['vtemp']); mesh=asset_reference(obj['object_mesh'])
                fps_counts[str(fps)]+=1; templates[template]+=1; objects[mesh]+=1
                rows.append({'member':member,'frames':frames,'fps':fps,'gender':gender,'hand_pca_components':pca,
                             'subject_template':template,'object_mesh':mesh,
                             'body_contact_shape':list(human_contact.shape),'object_contact_shape':list(object_contact.shape),
                             'body_contact_range':[int(human_contact.min()),int(human_contact.max())],
                             'object_contact_range':[int(object_contact.min()),int(object_contact.max())]})
            except (ValueError,KeyError,TypeError,IndexError,OSError,EOFError,pickle.UnpicklingError,zipfile.BadZipFile) as exc:
                errors.append({'member':member,'error_type':type(exc).__name__,'error':str(exc)})
    if digest(path)!=expected: raise RuntimeError('archive changed during audit')
    print(json.dumps({'archive_sha256':expected,'sequence_count':len(members),'structurally_read':len(rows),
                      'error_count':len(errors),'errors':errors,'fps_counts':dict(fps_counts),
                      'subject_template_counts':dict(templates),'object_mesh_counts':dict(objects),'sequences':rows,
                      'warning':'Restricted read of verified archive; structure only, not hostile-pickle safety, actual scene geometry, subject-template availability, contact correctness, licensing, retargeting or training acceptance.'},allow_nan=False))


if __name__=='__main__': main()

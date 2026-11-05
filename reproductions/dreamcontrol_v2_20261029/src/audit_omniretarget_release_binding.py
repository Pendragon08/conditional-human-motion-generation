"""Recompute source-release byte bindings and nominal time support.

The fixed publisher dataset card declares MIT for its release. This check
cannot establish all upstream rights, physical truth, or source clocks and
must not be used alone to authorize training.
"""
import hashlib,io,json,zipfile
from pathlib import Path
import numpy as np

ARCHIVE_SHA='93001f298aec587cd2b0a79aa5553d264cf2148e5397d4aa4bb767a9d58041b3'
CARD_SHA='cd662e88d89f583d9007f659c32a50e7d4ddb2d0be394d45e144126911b09024'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def audit_release_binding(records, *, root=None, state_manifest=None, archive=None, card=None):
    if not isinstance(records,list) or not records:raise ValueError('nonempty records required')
    if all(x is None for x in (root,state_manifest,archive,card)):
        return {'status':'missing_release_evidence','release_bytes_verified':False,'training_ready':False}
    if any(x is None for x in (root,state_manifest,archive,card)):raise ValueError('incomplete release evidence')
    root=Path(root).resolve(strict=True);sm=Path(state_manifest).resolve(strict=True)
    archive=Path(archive).resolve(strict=True);card=Path(card).resolve(strict=True);state_hash=sha(sm)
    if sha(archive)!=ARCHIVE_SHA or sha(card)!=CARD_SHA:raise ValueError('pinned release archive or card mismatch')
    text=card.read_text(encoding='utf8')
    if '\nlicense: mit\n' not in text or 'robot-object' not in text:raise ValueError('release card mismatch')
    states=json.loads(sm.read_text())['records'];index={r['sample_id']:r for r in states}
    if len(index)!=len(states) or set(index)!={r['sample_id'] for r in records} or len(records)!=len(index):
        raise ValueError('state evidence coverage mismatch')
    cases=[];seen=set();tail_count=0
    with zipfile.ZipFile(archive) as z:
      names=z.namelist()
      if len(names)!=len(set(names)):raise ValueError('duplicate archive member')
      for row in records:
        sid=row['sample_id'];member=row.get('source_member');state=index[sid]
        if not isinstance(member,str) or not member.startswith('robot-object/') or not member.endswith('.npz') or member in seen:
            raise ValueError('invalid or repeated release member')
        seen.add(member);info=z.getinfo(member)
        if info.file_size>64*1024**2:raise ValueError('release member too large')
        payload=z.read(info)
        if hashlib.sha256(payload).hexdigest()!=row.get('source_sha256'):raise ValueError('release member hash mismatch')
        with np.load(io.BytesIO(payload),allow_pickle=False) as data:
            source=data['qpos'];fps=float(data['fps'])
            if source.ndim!=2 or len(source)<2 or not np.isfinite(source).all() or fps!=30.:
                raise ValueError('invalid source motion or declared fps')
        rel=Path(state['state_path']);fre=Path(row['feature_path'])
        if rel.is_absolute() or fre.is_absolute():raise ValueError('paths must be relative')
        sp=(root/rel).resolve(strict=True);fp=(root/fre).resolve(strict=True)
        if (not sp.is_relative_to(root) or not fp.is_relative_to(root) or
            sha(sp)!=state['state_sha256'] or sha(fp)!=row['feature_sha256'] or
            state['feature_sha256']!=row['feature_sha256']):raise ValueError('state/feature hash mismatch')
        with np.load(sp,allow_pickle=False) as data:times=np.asarray(data['sample_times'],dtype=float)
        with np.load(fp,allow_pickle=False) as data:n=len(data['features'])
        if (times.shape!=(n+1,) or not np.isfinite(times).all() or
            not np.allclose(times,np.arange(len(times))/20,rtol=0,atol=1e-9) or
            times[-1]>(len(source)-1)/fps+1e-9):raise ValueError('nominal timing outside source support')
        tail=(len(source)-1)/fps-times[-1];tail_count+=int(tail>2/30+1e-9)
        if sha(sp)!=state['state_sha256'] or sha(fp)!=row['feature_sha256']:raise RuntimeError('state changed while checking')
        cases.append({'sample_id':sid,'source_sha256':row['source_sha256'],'declared_source_fps':fps,
                      'source_frames':len(source),'nominal_output_range_s':[float(times[0]),float(times[-1])]})
    if sha(archive)!=ARCHIVE_SHA or sha(card)!=CARD_SHA or sha(sm)!=state_hash:raise RuntimeError('release evidence changed')
    return {'status':'release_bytes_and_nominal_support_checked','release_bytes_verified':True,
            'archive_sha256':ARCHIVE_SHA,'card_sha256':CARD_SHA,'state_manifest_sha256':state_hash,
            'publisher_license_field':'mit','tail_over_two_source_frames':tail_count,'cases':cases,'training_ready':False,
            'limitations':['Released bytes and self-declared fps do not establish raw capture time or recording identity.',
                           'MIT is the publisher dataset-card field, not a determination of every upstream right.',
                           'Contact, physical quality, text semantics and overall training readiness are not verified.']}

"""Render complete nominal robot/object tracks for bounded caption review.

The object is a center marker with orientation axes, NOT a box or mesh. Stick
figures come from processed FK. This cannot verify contact or physical truth.
"""
import collections
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PREVIEW = ROOT / 'omomo_processed_full_preview_20261101'
HERE = Path(__file__).resolve().parent
CHAINS = ((0, 2, 5, 8, 11), (0, 1, 4, 7, 10), (0, 3, 6, 9, 12, 15),
          (9, 14, 17, 19, 21), (9, 13, 16, 18, 20))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def main():
    manifest=PREVIEW/'manifest.json'; report=json.loads((PREVIEW/'report.json').read_text())
    audit=json.loads((PREVIEW/'audit.json').read_text()); manifest_hash=sha(manifest)
    if manifest_hash!=report['manifest_sha256'] or audit['manifest_sha256']!=manifest_hash or audit['preview_report_sha256']!=sha(PREVIEW/'report.json'):
        raise ValueError('preview provenance mismatch')
    groups=collections.defaultdict(list)
    for row in json.loads(manifest.read_text())['records']:
        # Original non-cropped recordings only; variants and long15 are excluded.
        if row['source_member'].endswith('_original.npz') and row['processing_evidence_folder']!='omni_long15_gpu_20261101' and row['feature_frames']>=40:
            groups[row['caption']].append(row)
    if len(groups)!=7: raise ValueError('expected seven distinct nominal text templates')
    picked=[]
    for caption,rows in sorted(groups.items()):
        rows.sort(key=lambda r:(r['feature_frames'],r['sample_id']))
        picked.append(rows[len(rows)//2])
    cases=[]
    for row in picked:
        if shutil.disk_usage(HERE).free < 8*1024**3: raise RuntimeError('disk guard')
        path=(ROOT/row['state_path']).resolve(strict=True)
        if not path.is_relative_to(ROOT.resolve()) or sha(path)!=row['state_sha256']: raise ValueError('state hash mismatch')
        with np.load(path,allow_pickle=False) as f:
            pts=f['points_z_up'].astype(float);obj=f['object_wxyz_xyz'].astype(float);times=f['sample_times'].astype(float)
        n=len(times)
        if (pts.shape!=(n,22,3) or obj.shape!=(n,7) or n<2 or n!=row['feature_frames']+1 or
            not np.isfinite(pts).all() or not np.isfinite(obj).all() or
            not np.allclose(np.diff(times),.05,atol=1e-8,rtol=0) or
            not np.allclose(np.linalg.norm(obj[:,:4],axis=1),1.,atol=1e-5,rtol=0)):
            raise ValueError('state or nominal timeline invalid')
        # Object axes are visual markers of pose only, with arbitrary length.
        axes=Rotation.from_quat(obj[:,[1,2,3,0]]).as_matrix()*.15
        coords=np.concatenate([pts.reshape(-1,3),obj[:,4:]],axis=0)
        center=(coords.min(0)+coords.max(0))/2
        radius=max(float(np.ptp(coords,axis=0).max()*.58),.65)
        fig,panels=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
        plots=[]
        for ax,(a,b,label) in zip(panels,((0,2,'x / z'),(1,2,'y / z'),(0,1,'x / y'))):
            ax.set(xlim=(center[a]-radius,center[a]+radius),ylim=(center[b]-radius,center[b]+radius),aspect='equal',title=label,xlabel='world '+ 'xyz'[a] +' (m)',ylabel='world '+ 'xyz'[b] +' (m)')
            ax.grid(alpha=.15)
            chains=[ax.plot([],[],'-o',lw=1.5,markersize=2)[0] for _ in CHAINS]
            marker=ax.plot([],[],'ks',ms=5)[0]
            triad=[ax.plot([],[],color=c,lw=1.3)[0] for c in ('r','g','b')]
            trace=ax.plot([],[],color='black',alpha=.3,lw=.6)[0]
            plots.append((a,b,chains,marker,triad,trace))
        fig.suptitle(row['sample_id']+' | candidate text: '+row['caption']+'\nObject marker and 0.15 m pose axes; no object mesh/contact',fontsize=8)
        stamp=fig.text(.5,.01,'',ha='center',fontsize=8)
        movie=HERE/(row['sample_id']+'.mp4')
        if movie.exists():raise FileExistsError(movie)
        writer=FFMpegWriter(fps=20,codec='libx264',extra_args=['-pix_fmt','yuv420p','-crf','26'])
        with writer.saving(fig,str(movie),dpi=90):
            for i in range(n):
                for a,b,chains,marker,triad,trace in plots:
                    for line,chain in zip(chains,CHAINS):line.set_data(pts[i,list(chain),a],pts[i,list(chain),b])
                    center_obj=obj[i,4:];marker.set_data([center_obj[a]],[center_obj[b]])
                    for j,line in enumerate(triad):
                        end=center_obj+axes[i,:,j];line.set_data([center_obj[a],end[a]],[center_obj[b],end[b]])
                    trace.set_data(obj[:i+1,4+a],obj[:i+1,4+b])
                stamp.set_text(f'Nominal {times[i]:.2f} s | frame {i+1}/{n} | not a physical simulation')
                writer.grab_frame()
        plt.close(fig)
        if sha(path)!=row['state_sha256']:raise RuntimeError('state changed during rendering')
        cases.append({'sample_id':row['sample_id'],'caption':row['caption'],'state_sha256':row['state_sha256'],
                      'source_sha256':row['source_sha256'],'split':row['auxiliary_split'],'frames':n,
                      'video':movie.name,'video_sha256':sha(movie),'caption_accepted':False})
    if sha(manifest)!=manifest_hash:raise RuntimeError('manifest changed')
    out={'preview_manifest_sha256':manifest_hash,'script_sha256':sha(__file__),'cases':cases,
         'selection':'median duration, original non-long-crop, one per caption; not random',
         'training_ready':False,
         'limitations':['FK stick figure and object center/axes only; actual object dimensions and collision geometry absent.',
                        'Source identity, license, physical time and all object/human contact remain unverified.',
                        'Review videos alone do not accept caption semantics or training quality.']}
    with (HERE/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(out,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()

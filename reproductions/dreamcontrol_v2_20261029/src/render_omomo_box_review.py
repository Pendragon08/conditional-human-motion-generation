"""Render original tracks with the official original largebox convex silhouette.

The projected silhouette is of the mesh's convex hull, not collision/contact
geometry. Processed FK and object motion have not passed physical acceptance.
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
from matplotlib.patches import Polygon
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PREVIEW = ROOT / 'omomo_processed_full_preview_20261101'
ASSET = ROOT / 'omniretarget_largebox_asset_20261101'
PRIOR = ROOT / 'omomo_motion_review_20261101'
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
    if (sha(ASSET/'largebox.obj')!='dbcc11281f62e9226f49165252375080d2e490e5a3f0ab6ba917acbd8f7abc1c' or
        sha(ASSET/'largebox.urdf')!='627b7cf5b46500c89ca69afdd3ebdcaf64db6c6810dae43b94120eacafbab37e' or
        sha(ASSET/'visualize.py')!='5d00bd5f21b16aebbc3c0e9fed069a7c06e63128e25af54d699a4b5237a4f574'):
        raise ValueError('official object asset hash mismatch')
    prior=json.loads((PRIOR/'report.json').read_text())
    if sha(PRIOR/'report.json')!='1be78e7e7f28f2a0f83b5fe53dd40dd60f61f92420f601ca834803eff880de9c' or len(prior['cases'])!=7:
        raise ValueError('prior visual selection mismatch')
    candidates={r['sample_id']:r for r in json.loads(manifest.read_text())['records']}
    picked=[]
    for case in prior['cases']:
        row=candidates[case['sample_id']]
        if (row['state_sha256']!=case['state_sha256'] or row['caption']!=case['caption'] or
            not row['source_member'].endswith('_original.npz') or row['processing_evidence_folder']=='omni_long15_gpu_20261101'):
            raise ValueError('selected original binding mismatch')
        picked.append(row)
    mesh=trimesh.load(ASSET/'largebox.obj',force='mesh',process=False)
    vertices=np.asarray(mesh.vertices,dtype=float)
    if vertices.shape!=(13551,3) or not np.isfinite(vertices).all():raise ValueError('unexpected mesh')
    hull_vertices=vertices[ConvexHull(vertices).vertices]
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
        rotations=Rotation.from_quat(obj[:,[1,2,3,0]]).as_matrix()
        world_box=np.einsum('tij,kj->tki',rotations,hull_vertices)+obj[:,None,4:]
        coords=np.concatenate([pts.reshape(-1,3),world_box.reshape(-1,3)],axis=0)
        center=(coords.min(0)+coords.max(0))/2
        radius=max(float(np.ptp(coords,axis=0).max()*.58),.65)
        fig,panels=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
        plots=[]
        for ax,(a,b,label) in zip(panels,((0,2,'x / z'),(1,2,'y / z'),(0,1,'x / y'))):
            ax.set(xlim=(center[a]-radius,center[a]+radius),ylim=(center[b]-radius,center[b]+radius),aspect='equal',title=label,xlabel='world '+ 'xyz'[a] +' (m)',ylabel='world '+ 'xyz'[b] +' (m)')
            ax.grid(alpha=.15)
            chains=[ax.plot([],[],'-o',lw=1.5,markersize=2)[0] for _ in CHAINS]
            marker=ax.plot([],[],'ks',ms=5)[0]
            box=Polygon(np.zeros((3,2)),closed=True,edgecolor='black',facecolor='deepskyblue',alpha=.22,lw=1.3)
            ax.add_patch(box)
            trace=ax.plot([],[],color='black',alpha=.3,lw=.6)[0]
            plots.append((a,b,chains,marker,box,trace))
        fig.suptitle(row['sample_id']+' | candidate text: '+row['caption']+'\nOfficial original box convex silhouette; no object contact or scene',fontsize=8)
        stamp=fig.text(.5,.01,'',ha='center',fontsize=8)
        movie=HERE/(row['sample_id']+'.mp4')
        if movie.exists():raise FileExistsError(movie)
        writer=FFMpegWriter(fps=20,codec='libx264',extra_args=['-pix_fmt','yuv420p','-crf','26'])
        with writer.saving(fig,str(movie),dpi=90):
            for i in range(n):
                for a,b,chains,marker,box,trace in plots:
                    for line,chain in zip(chains,CHAINS):line.set_data(pts[i,list(chain),a],pts[i,list(chain),b])
                    center_obj=obj[i,4:];marker.set_data([center_obj[a]],[center_obj[b]])
                    projected=world_box[i,:,[a,b]].T
                    box.set_xy(projected[ConvexHull(projected).vertices])
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
         'mesh_sha256':sha(ASSET/'largebox.obj'),'urdf_sha256':sha(ASSET/'largebox.urdf'),
         'upstream_visualizer_sha256':sha(ASSET/'visualize.py'),'prior_render_sha256':sha(PRIOR/'report.json'),
         'selection':'same seven prior original non-long-crop cases, one per caption; not random',
         'training_ready':False,
         'limitations':['FK stick figure and projected convex hull of non-watertight original mesh; not actual surface or collision geometry.',
                        'Source identity, license, physical time and all object/human contact remain unverified.',
                        'Review videos alone do not accept caption semantics or training quality.']}
    with (HERE/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(out,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()

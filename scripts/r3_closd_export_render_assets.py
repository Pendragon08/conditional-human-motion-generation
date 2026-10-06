"""Private offline MeshCat scene assets for recorded Isaac Gym transforms.

Do not add exported asset HTML / raw transforms to the GitHub report archive.
Only derived videos and compact provenance/metrics belong to that archive.
"""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from meshcat import transformations as tf
from sim_web_visualizer.base_visualizer_client import MeshCatVisualizerBase

p=argparse.ArgumentParser();p.add_argument('directory');args=p.parse_args()
out=Path(args.directory)
with gzip.open(str(out/'actual_poses.json.gz'),'rt') as f:data=json.load(f)
viewer=MeshCatVisualizerBase()
for meta,env in zip(data['metadata'],data['frames'][0]['environments']):
    viewer.delete_all()
    for i,asset in enumerate(meta['assets']):
        resource=viewer.dry_load_asset(asset,False)
        viewer.load_asset_resources(resource,'Sim/env:0/actor:{}'.format(i))
    index=0
    for actor,names in enumerate(meta['body_names']):
        for name in names:
            pose=env['poses_xyzw'][index];index+=1
            mat=tf.quaternion_matrix([pose[6],pose[3],pose[4],pose[5]])
            # Isaac Gym rigid-body tensors are already environment-local.
            mat[:3,3]=np.asarray(pose[:3])
            viewer.viz['/Sim/env:0/actor:{}/{}'.format(actor,name)].set_transform(mat)
    (out/(meta['task'].lower()+'_private_scene.html')).write_text(viewer.viz.static_html())
    print('CLOSD_PRIVATE_RENDER_ASSET',meta['task'],flush=True)

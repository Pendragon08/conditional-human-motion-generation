"""Independent MeshCat renderer of actual CLoSD body and scene transforms."""
import json
from pathlib import Path
import socket
import time
import numpy as np
from meshcat import transformations as tf
from sim_web_visualizer.base_visualizer_client import MeshCatVisualizerBase

ROOT=Path(__file__).resolve().parents[1]; MON=ROOT/'artifacts/monitoring'
sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
sock.bind(('127.0.0.1',18446)); sock.settimeout(1)
visual=None; current=None; last=None
while True:
    try:
        data,_=sock.recvfrom(65535); packet=json.loads(data)
        while True:
            sock.setblocking(False)
            try: data,_=sock.recvfrom(65535); packet=json.loads(data)
            except BlockingIOError: break
            finally: sock.settimeout(1)
        meta=json.loads((MON/'r3_closd_actual_scene_meta.json').read_text())
        if current!=meta:
            visual=MeshCatVisualizerBase(port='6000',host='127.0.0.1')
            for i,asset in enumerate(meta['assets']):
                if asset and Path(asset).is_file():
                    resource=visual.dry_load_asset(asset,False)
                    visual.load_asset_resources(resource,f'Sim/env:0/actor:{i}')
            current=meta
        origin=np.asarray(meta['origin']); index=0
        for actor,names in enumerate(meta['body_names']):
            for name in names:
                pose=packet['poses_xyzw'][index]; index+=1
                matrix=tf.quaternion_matrix([pose[6],pose[3],pose[4],pose[5]])
                matrix[:3,3]=np.asarray(pose[:3])-origin
                visual.viz[f'/Sim/env:0/actor:{actor}/{name}'].set_transform(matrix)
        last=packet
        tmp=MON/'r3_closd_actual_scene_frame.tmp'
        tmp.write_text(json.dumps(packet)); tmp.replace(MON/'r3_closd_actual_scene_frame.json')
        (MON/'r3_closd_actual_scene_receiver.state').write_text(
            f'LIVE source=actual_isaacgym_simulation step={packet["step"]} experiment={packet["experiment"]}\n')
    except socket.timeout:
        if last and time.time()-last['ts']>10:
            (MON/'r3_closd_actual_scene_receiver.state').write_text('STALE source=actual_isaacgym_simulation\n')
    except Exception as error:
        print('MONITOR_ONLY_ERROR',repr(error),flush=True)
        visual=None; current=None; time.sleep(2)

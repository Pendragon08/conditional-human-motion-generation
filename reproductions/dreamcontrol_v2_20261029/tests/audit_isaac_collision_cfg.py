"""Read existing DreamControl v1 Isaac Lab saved environment configuration.

Use BaseLoader, which does not instantiate Python objects embedded in YAML.
These are saved configuration values, not a runtime USD/PhysX or v2 audit.
"""
import hashlib
import json
from pathlib import Path
import yaml


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def inspect(path):
    with path.open(encoding='utf-8') as f: data=yaml.load(f,Loader=yaml.BaseLoader)
    scene=data['scene']; robot=scene['robot']; spawn=robot['spawn']; props=spawn['articulation_props']; terrain=scene['terrain']
    return {'config':str(path),'sha256':sha256(path),'num_envs':scene.get('num_envs'),
            'robot_usd_path':spawn.get('usd_path'),'enabled_self_collisions':props.get('enabled_self_collisions'),
            'fix_root_link':props.get('fix_root_link'),'contact_sensors':spawn.get('activate_contact_sensors'),
            'terrain_type':terrain.get('terrain_type'),'terrain_prim_path':terrain.get('prim_path'),
            'terrain_material':terrain.get('physics_material'),'sim_gravity':data.get('sim',{}).get('gravity'),
            'sim_dt':data.get('sim',{}).get('dt'),'scene_assets':sorted(scene.keys())}


def main():
    root=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/logs/rsl_rl/g1')
    tasks=('2026-09-27_07-24-06-Punch-a','2026-10-01_19-59-36-Pick-a','2026-10-03_02-26-12-Open-Drawer-a')
    rows=[inspect(root/name/'params/env.yaml') for name in tasks]
    print(json.dumps({'tasks':rows,'warning':'Read-only v1 saved YAML only; runtime USD collision filtering, physical behavior and DreamControl-v2 setup are not verified.'},allow_nan=False))


if __name__=='__main__': main()

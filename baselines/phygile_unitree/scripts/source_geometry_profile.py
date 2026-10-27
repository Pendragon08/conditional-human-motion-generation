"""Deterministic G1 source-URDF collision conversion; no installed asset edits."""
from pathlib import Path
import hashlib,xml.etree.ElementTree as ET
import mujoco,numpy as np
from scipy.spatial.transform import Rotation

def source_collision_factory(original,urdf):
    data=Path(urdf).read_bytes();digest=hashlib.sha256(data).hexdigest();root=ET.fromstring(data)
    definitions=[]
    for link in root.findall('link'):
        for index,collision in enumerate(link.findall('collision')):
            origin=collision.find('origin');geometry=collision.find('geometry')
            xyz=np.fromstring(origin.get('xyz','0 0 0') if origin is not None else '0 0 0',sep=' ')
            rpy=np.fromstring(origin.get('rpy','0 0 0') if origin is not None else '0 0 0',sep=' ')
            if geometry.find('cylinder') is not None:
                g=geometry.find('cylinder');kind=int(mujoco.mjtGeom.mjGEOM_CAPSULE);size=[float(g.get('radius')),float(g.get('length'))/2,0]
            elif geometry.find('sphere') is not None:
                g=geometry.find('sphere');kind=int(mujoco.mjtGeom.mjGEOM_SPHERE);size=[float(g.get('radius')),0,0]
            elif geometry.find('box') is not None:
                g=geometry.find('box');kind=int(mujoco.mjtGeom.mjGEOM_BOX);size=(np.fromstring(g.get('size'),sep=' ')/2).tolist()
            else:raise ValueError('Unsupported source collision; no silent approximation')
            body=link.get('name');foot=body.endswith('_ankle_roll_link');side=body.split('_')[0]
            name=(side+'_foot'+str(index+1)+'_collision') if foot else body+'_source'+str(index)+'_collision'
            definitions.append({'body':body,'name':name,'type':kind,'size':size,'pos':xyz.tolist(),
                                'quat':Rotation.from_euler('xyz',rpy).as_quat(scalar_first=True).tolist(),'foot':foot})
    assert len(definitions)==29,'Unexpected source collision schema'
    def factory():
        spec=original()
        for geom in list(spec.geoms):
            if geom.name.endswith('_collision'):spec.delete(geom)
        for d in definitions:
            spec.body(d['body']).add_geom(name=d['name'],type=d['type'],size=d['size'],pos=d['pos'],quat=d['quat'],
              contype=1,conaffinity=1,condim=3 if d['foot'] else 1,priority=1 if d['foot'] else 0,
              friction=[1.,.005,.0001],group=3,rgba=[.2,.6,.2,.3])
        return spec
    return factory,{'source_urdf_sha256':digest,'source_collision_count':29,'source_definitions':definitions,
       'scope':'Collision primitives from measured source URDF; cylinders become capsules as in Isaac import. Preserve official articulated kinematics, inertia, actuator settings, solver and self-contact condim; sliding friction1.0. No claim of complete engine equivalence.'}

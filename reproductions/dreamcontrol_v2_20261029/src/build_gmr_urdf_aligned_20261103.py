"""Create a separate GMR kinematic model with URDF waist offsets; never modify vendor files."""
import hashlib, json, xml.etree.ElementTree as ET
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
src=R/'gmr_locked27_model_20261103/g1_gmr_locked27.xml'
out=R/'gmr_urdf_aligned27_20261103'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
root=ET.parse(src).getroot()
roll=root.find(".//body[@name='waist_roll_link']")
torso=roll.find("body[@name='torso_link']")
head=torso.find("body[@name='head_link']")
assert roll.get('pos')=='-0.0039635 0 0.044' and torso.get('pos') is None and head.get('pos')=='0.0039635 0 -0.044'
roll.set('pos','-0.0039635 0 0.035')
torso.set('pos','0 0 0.019')
head.set('pos','0.0039635 0 -0.054')
for e in torso.findall('geom'):
 if e.get('pos')=='0.0039635 0 -0.044': e.set('pos','0.0039635 0 -0.054')
out.mkdir(exist_ok=False)
target=out/'g1_gmr_locked27_aligned.xml'
ET.ElementTree(root).write(target,encoding='utf-8',xml_declaration=True)
with (out/'provenance.json').open('x') as f: json.dump({'source_sha256':sha(src),'output_sha256':sha(target),'changes':'Waist fixed-link and head offsets set to target URDF; copied GMR limits, inertias and meshes are not validated for dynamics.','training_ready':False},f,indent=2)
print(sha(target))

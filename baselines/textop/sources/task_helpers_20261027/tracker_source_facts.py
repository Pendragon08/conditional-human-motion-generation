import pathlib,json,hashlib,xml.etree.ElementTree as ET,subprocess
base=pathlib.Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party')
p=base/'whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf'
tree=ET.parse(p).getroot()
lab=base/'IsaacLab';version=(lab/'VERSION').read_text().strip()
query=subprocess.run(['git','-C',str(lab),'rev-parse','HEAD'],text=True,capture_output=True);head=query.stdout.strip() if query.returncode==0 else None
out={'urdf_path':str(p),'urdf_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'robot_links':len(tree.findall('link')),
     'actuated_joint_names':[j.get('name') for j in tree.findall('joint') if j.get('type') not in ['fixed','floating']],
     'IsaacLab_version':version,'IsaacLab_commit':head}
print(json.dumps(out))

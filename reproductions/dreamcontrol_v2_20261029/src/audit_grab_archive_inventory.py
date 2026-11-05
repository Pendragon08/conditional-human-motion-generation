"""Inventory GRAB archives and staged assets, without deserializing motion fields."""
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT=Path(r'D:\具身智能\DATASET')
REPORT=Path(__file__).resolve().parents[1]/'reports/grab_s2_all_structure_20261030.json'
EXTRA=Path(__file__).resolve().parents[1]/'reports/grab_s2_large_contact_structure_20261030.json'

def main():
    genders={}
    assets={}
    for gender in ('male','female'):
        with zipfile.ZipFile(ROOT/('tools__subject_meshes__'+gender+'.zip')) as z:
            names=set(z.namelist())
            for n in range(1,11):
                if f'{gender}/s{n}.ply' in names:
                    genders.setdefault(f's{n}',[]).append(gender)
            assets[gender]=len([n for n in names if n.endswith('.ply')])
    with zipfile.ZipFile(ROOT/'tools__object_meshes__contact_meshes.zip') as z:
        object_members={PurePosixPath(n).name for n in z.namelist() if n.endswith('.ply')}
    refs=set(json.loads(REPORT.read_text(encoding='utf8'))['object_mesh_counts'])
    refs.update(row['object_mesh'] for row in json.loads(EXTRA.read_text(encoding='utf8'))['sequences'])
    rows=[]
    for n in range(1,11):
        with zipfile.ZipFile(ROOT/f'grab__s{n}.zip') as z:
            names=[x.filename for x in z.infolist() if x.filename.endswith('.npz')]
            rows.append({'subject':n,'archive':f'grab__s{n}.zip','sequence_count':len(names),
                         'all_paths_match_subject':all(PurePosixPath(x).parts[0]==f's{n}' for x in names),
                         'duplicate_paths':len(names)-len(set(names)), 'subject_template_gender':genders.get(f's{n}',[])})
    print(json.dumps({'subjects':rows,'sequence_total':sum(r['sequence_count'] for r in rows),
       'object_ply_count':len(object_members),'template_ply_counts':assets,
       's2_referenced_object_count':len(refs),
       's2_missing_object_names':[x for x in sorted(refs) if PurePosixPath(x).name not in object_members],
       'limitations':['Archive directory inventory, not full content or motion-quality validation.',
                      'Subject template existence does not establish SMPL-X model compatibility.',
                      'Matching PLY filenames does not prove mesh topology or vertex-contact alignment.',
                      'Other subjects object references and license details have not been individually audited.']},indent=2))

if __name__=='__main__':main()

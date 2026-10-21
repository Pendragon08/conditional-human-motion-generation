"""Inventory only HumanML3D-required recordings; no downloads/deletions."""
import argparse
import collections
import csv
import json
from pathlib import Path, PurePosixPath
import re


def normalized(name):
    return re.sub('[^a-z0-9/]', '', name.lower())


def main(a):
    base=a.base.resolve();a.output.mkdir(parents=True,exist_ok=False)
    records={};server={}
    for folder in ['acquisition_v4','acquisition_v5']:
        root=base/'results/amass_remaining_20261021'/folder
        for p in root.glob('*.json'):
            data=json.loads(p.read_text())
            if data.get('status')=='LOCAL_ARCHIVE_VERIFIED':records[data['subset']]=data
        for p in (root/'server').glob('*.json'):
            data=json.loads(p.read_text());server[data['subset']]=data
    # SSM predates the 15-subset batch. The actual server extraction is pinned.
    ssm=json.loads((base/'results/paired_data_20261021/SSM_verified_acquisition.json').read_text())
    ssm_rows=list(csv.DictReader((base/'cache/humanml_amass_plan_20261021_verified/source_requirements.csv').open()))
    frames={}
    for subset,row in records.items():
        frames[subset]={m['path'].split('/',1)[1]:m['path'] for m in row['members'] if '/' in m['path'] and m['path'].endswith('.npz')}
    reference=json.loads((base/'results/humanml_g1_pipeline_20261021/ssm_reference_server.json').read_text())
    refmap={m['path'].split('/',1)[1]:m for m in reference['files'] if m['path'].endswith('_poses.npz')}
    frames['SSM']={k[:-len('_poses.npz')]+'_stageii.npz':'SSM/'+k[:-len('_poses.npz')]+'_stageii.npz' for k in refmap}
    rows=[];counts=collections.defaultdict(collections.Counter);jobs=[]
    plan=[json.loads(line) for line in (base/'cache/humanml_amass_plan_20261021_verified/alignment_plan.jsonl').read_text().splitlines()]
    ids=collections.defaultdict(list)
    for item in plan:ids[item['source_group']].append(item['motion_id'])
    for item in ssm_rows:
        subset=item['amass_download_name'];name=item['expected_original_smplh_file'].split('/',1)[1]
        target=name[:-len('_poses.npz')]+'_stageii.npz' if name.endswith('_poses.npz') else name
        members=frames.get(subset,{})
        if subset=='humanact12':status='SEPARATE_HUMANACT12_ADAPTER_REQUIRED';match=None
        elif subset in records and records[subset]['export']!='SMPL-X G':status='SMPLH_BODY_MODEL_ADAPTER_REQUIRED';match=None
        else:
            match=target if target in members else None
            if match is None:
                candidates=[n for n in members if normalized(n)==normalized(target)]
                if len(candidates)==1:match=candidates[0]
            status='LOCAL_MATCH_NEEDS_REFERENCE_TIMELINE' if match else 'REQUIRED_RECORDING_NOT_AVAILABLE'
        server_path=None
        if match and subset in server:
            server_path=server[subset]['server_directory']+'/extracted/'+members[match]
        if match and subset=='SSM':
            server_path='/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/ssm_20261021/extracted/'+members[match]
            if name not in refmap:raise ValueError('Missing SSM reference')
            jobs.append({'source_group':item['humanml_source_path'],'motion_ids':sorted(ids[item['humanml_source_path']]),
                'smplx_file':server_path,'smplh_reference_file':reference['server_directory']+'/extracted/SSM_synced/'+name})
            status='READY_FOR_MEASURED_MARKER_TIMEBASE_VALIDATION'
        counts[subset][status]+=1
        rows.append({**item,'matched_smplx_member':members.get(match),'server_smplx_path':server_path,'status':status})
    with (a.output/'indexed_asset_coverage.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={'required_recordings':len(rows),'ssm_sources_ready_for_validation':len(jobs),
        'ssm_motion_ids':sum(len(j['motion_ids']) for j in jobs),
        'subsets':{k:dict(v) for k,v in sorted(counts.items())},
        'whole_amass_download_stopped':True,'no_inputs_deleted':True,
        'coverage_is_not_timeline_or_motion_quality_validation':True}
    for name,value in [('summary.json',result),('ssm_jobs.json',{'jobs':sorted(jobs,key=lambda x:x['source_group'])})]:
        (a.output/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())

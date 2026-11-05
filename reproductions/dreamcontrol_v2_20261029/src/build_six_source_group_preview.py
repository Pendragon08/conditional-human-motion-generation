"""Build a non-training, directory-grouped six-source candidate view.

Captions remain raw annotations for complete HumanML clips. Temporal coverage
here is conditional on the candidate source/index/fps assumptions and never
asserts that an annotation describes an excerpt.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile
from provenance_groups import audit_provenance

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HUMANML = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D')
HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    preview = ROOT/'ssm_four_preview_20261031'
    original = json.loads((preview/'experimental_manifest_not_for_training.json').read_text())
    before = json.loads((ROOT/'bmlhandball_overlap_witness_v2_20261030/report.json').read_text())
    prepared = json.loads((ROOT/'bml_global_candidates_20261031/manifest.json').read_text())
    archive = HUMANML/'texts.zip'
    archive_hash = '58f4f19b5429fe137d7a2ea216099f4745f2be51fff001f2f90d6747eaf053d2'
    if sha(archive) != archive_hash: raise ValueError('caption archive changed')
    records=[]
    for rec in original['records']:
        source=rec['source_id']; parent=str(PurePosixPath(source).parent)
        assigned = 'test' if parent.endswith('20161014_50033') else 'train'
        records.append({'sample_id':rec['sample_id'],'split':assigned,'official_split':rec['official_split'],
                        'source_id':source,'provenance_keys':{'source_path':[source], 'source_directory':[parent]},
                        'excerpt_text_accepted':False, 'kind':'existing_ssm_preview'})
    with zipfile.ZipFile(archive) as z:
        for index, assigned in enumerate(('val','test')):
            src=before['witnesses'][index]; item=prepared['cases'][index]
            if src['candidate_sha256'] != item['candidate_source_sha256']: raise ValueError('candidate mismatch')
            source=src['source_path']; parent=str(PurePosixPath(source).parent)
            start,end=item['source_frames_first_last']
            if start%6 or end%6 or end-start!=360: raise ValueError('unexpected 20 Hz grid')
            annotations=[]
            for clip_id, bounds in src['index_slices'].items():
                # Solver output begins at input point 1; no extension beyond either clip.
                first=start//6+1; last=end//6
                if first<bounds[0] or last>=bounds[1]: raise ValueError('excerpt outside clip')
                content=z.read('texts/'+clip_id+'.txt')
                lines=[]
                for line in content.decode('utf8').splitlines():
                    fields=line.split('#')
                    if len(fields)!=4: raise ValueError('invalid caption line')
                    lines.append({'caption':fields[0], 'tokens':fields[1], 'start_raw':fields[2], 'end_raw':fields[3]})
                annotations.append({'clip_id':clip_id,'official_split':src['ids'][clip_id],
                                    'index_slice':bounds,'excerpt_first_last_seconds_relative_to_clip':
                                      [(first-bounds[0])/20,(last-bounds[0])/20],
                                    'caption_member_sha256':hashlib.sha256(content).hexdigest(),'annotations':lines})
            records.append({'sample_id':'BML_'+item['label'],'split':assigned,'source_id':source,
                            'provenance_keys':{'source_path':[source],'source_directory':[parent],
                                               'raw_candidate_sha256':[item['candidate_source_sha256']]},
                            'source_frames_input_first_last':item['source_frames_first_last'],
                            'candidate_clip_annotations':annotations,'excerpt_text_accepted':False,
                            'kind':'bml_pending_robot_output'})
    grouping=audit_provenance(records)
    if not grouping['source_groups_disjoint'] or grouping['component_count'] != 4:
        raise ValueError('unexpected proposed source grouping')
    result={'records':records,'group_audit':grouping,'source_preview_sha256':sha(preview/'experimental_manifest_not_for_training.json'),
            'bml_source_report_sha256':sha(ROOT/'bmlhandball_overlap_witness_v2_20261030/report.json'),
            'bml_prepare_manifest_sha256':sha(ROOT/'bml_global_candidates_20261031/manifest.json'),
            'caption_archive_sha256':archive_hash,'training_ready':False,
            'limitations':['Auxiliary split proposal, official splits unchanged; BML validation source also occurs in official test, so this is not an official benchmark split.',
                           'Directory grouping is not proof of subject independence or complete aliases.',
                           'BML timing relies on declared FPS and candidate index identity; semantic correspondence is unreviewed.',
                           'BML dataset terms and training quality not accepted; robot results pending.',
                           'Cannot calculate training statistics until actual eligible training data is established.']}
    with (HERE/'six_source_preview.json').open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps({'samples':len(records),'groups':grouping['component_count'],
                      'crossings':len(grouping['cross_split_components']),'training_ready':False}))


if __name__=='__main__':main()

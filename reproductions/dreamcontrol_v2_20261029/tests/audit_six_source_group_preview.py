"""Independently inspect six-source candidate view against source records."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath
import zipfile

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HUMANML=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D')
HERE=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    viewpath=HERE/'six_source_preview.json'; view=json.loads(viewpath.read_text())
    ssm=ROOT/'ssm_four_preview_20261031/experimental_manifest_not_for_training.json'
    bml=ROOT/'bmlhandball_overlap_witness_v2_20261030/report.json'
    prepared=ROOT/'bml_global_candidates_20261031/manifest.json'
    captions=HUMANML/'texts.zip'
    assert sha(ssm)==view['source_preview_sha256']
    assert sha(bml)==view['bml_source_report_sha256']
    assert sha(prepared)==view['bml_prepare_manifest_sha256']
    assert sha(captions)==view['caption_archive_sha256']
    original={r['sample_id']:r for r in json.loads(ssm.read_text())['records']}
    candidates=json.loads(bml.read_text())['witnesses']
    source_rows={r['source_path']:r for r in candidates}
    items=json.loads(prepared.read_text())['cases']
    batch={r['label']:r for r in items}
    groups=defaultdict(set); checked=[]; copied_captions=0
    with zipfile.ZipFile(captions) as z:
        for rec in view['records']:
            source=rec['source_id']; parent=str(PurePosixPath(source).parent)
            assert rec['provenance_keys']['source_directory']==[parent]
            assert rec['provenance_keys']['source_path']==[source]
            groups[parent].add(rec['split'])
            assert rec['excerpt_text_accepted'] is False
            if rec['kind']=='existing_ssm_preview':
                old=original[rec['sample_id']]
                assert source==old['source_id'] and rec['official_split']==old['official_split']
                if old['official_split']=='test': assert rec['split']!='train'
            else:
                assert rec['sample_id'].startswith('BML_')
                src=source_rows[source]; item=batch[rec['sample_id'][4:]]
                assert src['candidate_sha256']==item['candidate_source_sha256']
                assert sha(Path(src['candidate_path']))==src['candidate_sha256']
                assert rec['provenance_keys']['raw_candidate_sha256']==[src['candidate_sha256']]
                start,end=item['source_frames_first_last']
                for ann in rec['candidate_clip_annotations']:
                    ident=ann['clip_id']; lo,hi=src['index_slices'][ident]
                    assert ann['index_slice']==[lo,hi] and ann['official_split']==src['ids'][ident]
                    assert lo<=start//6+1<=end//6<hi
                    assert ann['excerpt_first_last_seconds_relative_to_clip']==[(start//6+1-lo)/20,(end//6-lo)/20]
                    data=z.read('texts/'+ident+'.txt')
                    assert hashlib.sha256(data).hexdigest()==ann['caption_member_sha256']
                    entries=[line.split('#') for line in data.decode('utf8').splitlines()]
                    saved=ann['annotations']; assert len(entries)==len(saved)
                    for fields, obj in zip(entries,saved):
                        assert fields==[obj['caption'],obj['tokens'],obj['start_raw'],obj['end_raw']]
                        copied_captions+=1
            checked.append(rec['sample_id'])
    assert len(checked)==6 and len(set(checked))==6
    assert len(groups)==4 and all(len(x)==1 for x in groups.values())
    assert view['training_ready'] is False
    result={'preview_sha256':sha(viewpath),'records_checked':len(checked),
            'source_directory_groups':len(groups),'cross_split_directories':0,
            'raw_caption_lines_checked':copied_captions,'status':'passed',
            'limitations':['Does not verify semantic fit, candidate identity, group completeness, license or physical quality.',
                           'Shares underlying source records with builder; not an independent source-identity check.']}
    with (HERE/'independent_audit.json').open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

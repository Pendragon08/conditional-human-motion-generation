"""Verify and stage an authorised SSM SMPL-X archive without unpickling objects.

This does not retarget, certify cross-export timing, or create a training set.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ingest(archive_path, public, output, receipt):
    archive_path, public, output, receipt = map(Path, (archive_path, public, output, receipt))
    if os.name == 'nt':
        prefix = chr(92)*2 + '?' + chr(92)
        output = Path(prefix + str(output.resolve()))
        receipt = Path(prefix + str(receipt.resolve()))
    if output.exists() or receipt.exists():
        raise FileExistsError('Preserve the existing staged data and receipt')
    with archive_path.open('rb') as f:
        if f.read(3) != b'BZh':
            raise ValueError('Not a bzip2 archive')
    output.mkdir(parents=True)
    raw, companions, licenses = {}, [], []
    with tarfile.open(archive_path, 'r:bz2') as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or '..' in name.parts:
                raise ValueError('Unsafe archive entry: ' + member.name)
            if not name.parts or name.parts[0] != 'SSM':
                raise ValueError('Unexpected SSM entry: ' + member.name)
            payload = archive.extractfile(member).read()
            if len(payload) != member.size:
                raise ValueError('Incomplete archive member')
            if name.name == 'LICENSE.txt':
                target = output.joinpath(*name.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                licenses.append({'file':str(name),'sha256':hashlib.sha256(payload).hexdigest(),
                                 'bytes':len(payload)})
                continue
            if name.suffix != '.npz':
                raise ValueError('Unexpected SSM entry: ' + member.name)
            with np.load(io.BytesIO(payload), allow_pickle=False) as z:
                model = str(z['surface_model_type'].item())
                gender = str(z['gender'].item())
                if model != 'smplx' or gender not in ('female', 'male', 'neutral'):
                    raise ValueError('Unexpected body model metadata')
                beta = z['betas']
                if beta.shape != (16,) or not np.isfinite(beta).all():
                    raise ValueError('Expected finite 16-beta SMPL-X parameters')
                info = {'file': str(name), 'sha256':hashlib.sha256(payload).hexdigest(),
                        'bytes':len(payload), 'gender':gender, 'surface_model_type':model}
                if name.name.endswith('_stageii.npz'):
                    fps = float(z['mocap_frame_rate'])
                    n = len(z['trans'])
                    if not np.isfinite(fps) or fps < 20 or n < 2:
                        raise ValueError('Invalid motion timeline')
                    for key, shape in [('trans',(n,3)), ('root_orient',(n,3)),
                                       ('pose_body',(n,63)), ('poses',(n,165))]:
                        values = z[key]
                        if values.shape != shape or not np.isfinite(values).all():
                            raise ValueError('Invalid SMPL-X array: '+key)
                    if not np.allclose(z['poses'][:,:3], z['root_orient']) or not np.allclose(
                            z['poses'][:,3:66], z['pose_body']):
                        raise ValueError('Redundant pose fields disagree')
                    info.update(frames=n, fps=fps, timestamp_span_seconds=(n-1)/fps)
                    raw[str(name)] = info
                elif name.name.endswith('_stagei.npz'):
                    companions.append(info)
                else:
                    raise ValueError('Unexpected NPZ role')
            target = output.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            if digest(target) != info['sha256']:
                raise ValueError('Staged member checksum mismatch')
    official_ids = set((public/'HumanML3D/all.txt').read_text().splitlines())
    matches = []
    for row in csv.DictReader((public/'index.csv').open(encoding='utf-8',newline='')):
        source = PurePosixPath(row['source_path'])
        if len(source.parts)<4 or source.parts[1]!='SSM_synced':
            continue
        tail = PurePosixPath(*source.parts[2:])
        if not tail.name.endswith('_poses.npy'):
            raise ValueError('Unexpected HumanML SSM recording name')
        target_name = str(PurePosixPath('SSM')/tail.parent/(tail.name[:-len('_poses.npy')]+'_stageii.npz'))
        mid = Path(row['new_name']).stem
        matched = raw.get(target_name)
        matches.append({'canonical_motion_id':mid,
                        'official_text_ids':[x for x in (mid,'M'+mid) if x in official_ids],
                        'humanml_source':row['source_path'], 'smplx_export':target_name,
                        'export_found': matched is not None,
                        'index_start_20':int(row['start_frame']), 'index_end_20':int(row['end_frame']),
                        'timeline_status':'recording_name_matched; original_smplh_clock_not_verified',
                        'g1_motion_generated':False})
    result={'date':'2026-10-21','status':'SSM_RAW_ARCHIVE_VERIFIED_NOT_G1_TRAINING_DATA',
            'archive_filename':archive_path.name,'archive_bytes':archive_path.stat().st_size,
            'archive_sha256':digest(archive_path),
            'motion_sequences':len(raw),'companion_parameter_files':len(companions),
            'companion_files_are_body_model_weights':False,
            'required_model_genders':sorted({v['gender'] for v in raw.values()}),
            'required_body_model':'SMPL-X body model weights, separate download',
            'raw_motion_records':list(raw.values()), 'companion_records':companions,
            'license_records':licenses,'numpy_version':np.__version__,
            'humanml_source_clips':len(matches),
            'matched_source_clips':sum(x['export_found'] for x in matches),
            'candidate_text_motion_ids':sum(len(x['official_text_ids']) for x in matches if x['export_found']),
            'source_matches':matches,
            'cross_export_timing_verified':False,'retargeting_executed':False,
            'g1_paired_sequences':0,'new_262d_sequences':0,'generator_training_started':False,
            'object_fields_loaded':False,'redistribution_performed':False,
            'license_url':'https://download.is.tue.mpg.de/amass/licenses2024/SSM/ssm_license_updated.html',
            'code_sha256':digest(__file__)}
    receipt.parent.mkdir(parents=True,exist_ok=True)
    receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in
                      ('raw_motion_records','companion_records','source_matches')},ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','public','output','receipt'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    ingest(args.archive,args.public,args.output,args.receipt)

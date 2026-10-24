"""CPU-only model installation witness; produces no paired training motion."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import smplx
import torch


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4194304), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-root', type=Path, required=True)
    p.add_argument('--ssm-file', type=Path, required=True)
    p.add_argument('--server-model-receipt', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    installation = json.loads(a.server_model_receipt.read_text())
    assert installation['archive_sha256'] == '88d35123fc97151bc258dc6f28f3967ae4c16117e4b452445f20598c43c43099'
    file_records = {Path(x['path']).name: x for x in installation['files']}
    records = []
    with np.load(a.ssm_file, allow_pickle=False) as data:
        betas = np.array(data['betas'], copy=True).reshape(1, -1)
        body_pose = np.array(data['pose_body'][:1], copy=True)
        root = np.array(data['root_orient'][:1], copy=True)
        trans = np.array(data['trans'][:1], copy=True)
        gender = str(data['gender'])
    assert gender == 'female' and betas.shape == (1, 16)
    for model_gender in ('female', 'male', 'neutral'):
        model_file = a.model_root / 'smplx' / ('SMPLX_' + model_gender.upper() + '.npz')
        digest = sha(model_file)
        assert digest == file_records[model_file.name]['sha256']
        model = smplx.create(str(a.model_root), 'smplx', gender=model_gender,
                             ext='npz', num_betas=16, use_pca=False).cpu()
        kwargs = {}
        if model_gender == 'female':
            kwargs = {'betas': torch.from_numpy(betas).float(),
                      'body_pose': torch.from_numpy(body_pose).float(),
                      'global_orient': torch.from_numpy(root).float(),
                      'transl': torch.from_numpy(trans).float()}
        with torch.no_grad():
            out = model(**kwargs)
        assert out.vertices.shape == (1, 10475, 3)
        assert torch.isfinite(out.vertices).all() and torch.isfinite(out.joints).all()
        records.append({'gender': model_gender, 'model_file_sha256': digest,
                        'vertices_shape': list(out.vertices.shape), 'joints_shape': list(out.joints.shape),
                        'finite': True, 'actual_ssm_body_frame_used': model_gender == 'female'})
    result = {'status': 'SMPLX_MODEL_CPU_FORWARD_PASSED', 'seed': 42, 'device': 'cpu',
              'model_archive_sha256': installation['archive_sha256'], 'model_genders': records,
              'ssm_input_sha256': sha(a.ssm_file), 'num_betas': 16,
              'shape_parameter_note': 'Explicitly use all 16 AMASS betas; this is an installation witness, not the unmodified GMR loader.',
              'hands_and_face_note': 'Default model hand and face parameters; no full motion retargeting is claimed.',
              'versions': {x: importlib.metadata.version(x) for x in ('numpy', 'torch', 'smplx')},
              'human_model_forward_pass_executed': True, 'gmr_retargeting_executed': False,
              'g1_pairs_generated': 0, 'new_262d_sequences': 0, 'generator_training_started': False,
              'source_sha256': sha(Path(__file__))}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

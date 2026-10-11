"""Record small reproducibility artifacts without redistributing motion arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''): h.update(part)
    return h.hexdigest()


def main(out):
    files=list((ROOT/'scripts').glob('physics_guided*'))+[ROOT/'scripts/r3_closd_isolated_entry.py',
        ROOT/'configs/physics_guided_20261011.json',ROOT/'configs/env-closd-baseline-20261010.json',
        ROOT/'docs/physics_guided_20261011.md',
        out/'source_manifest.json',out/'humanml_source_index.csv']
    record=dict(python=sys.version,files={str(p.relative_to(ROOT)):digest(p) for p in files if p.is_file()},
                upstream_commit=subprocess.check_output(['git','-C',str(ROOT/'third_party/CLoSD'),
                    'rev-parse','HEAD'],text=True).strip())
    weights=ROOT/'third_party/CLoSD/closd/diffusion_planner/save/DiP_multi-target_10steps_context20_predict40/model000300000.pt'
    record['generator_weights_sha256']=digest(weights)
    tracker=ROOT/'third_party/CLoSD/output/CLoSD/CLoSD_multitask_finetune/Humanoid.pth'
    record['controller_weights_sha256']=digest(tracker)
    snapshot=Path(json.loads((out/'source_manifest.json').read_text())['snapshot'])
    evaluator=snapshot/'evaluators/t2m/text_mot_match/model/finest.tar'
    record['text_motion_evaluator_weights_sha256']=digest(evaluator) if evaluator.exists() else 'not_available'
    manifest=json.loads((out/'source_manifest.json').read_text())
    inputs={manifest['mean'],manifest['std']}|{r['cache_file'] for r in manifest['rows']}
    record['external_motion_inputs']={p:digest(Path(p)) for p in sorted(inputs)}
    semantic_inputs=[ROOT/'third_party/CLoSD/closd/diffusion_planner/dataset/t2m_mean.npy',
                     ROOT/'third_party/CLoSD/closd/diffusion_planner/dataset/t2m_std.npy']
    semantic_inputs+=sorted(p for p in (snapshot/'glove').rglob('*') if p.is_file())
    record['external_semantic_inputs']={str(p):digest(p) for p in semantic_inputs}
    record['assets_revision']=snapshot.name
    (out/'versions.json').write_text(json.dumps(record,indent=2))
    protocol=out/'protocol_snapshot.md'
    source=(ROOT/'docs/physics_guided_20261011.md').read_text()
    if protocol.exists(): assert protocol.read_text()==source, 'Do not overwrite preserved protocol snapshot'
    else: protocol.write_text(source)
    manifest=json.loads((out/'source_manifest.json').read_text())
    public=[dict(clip_id=r['clip_id'],family=r['family'],split=r['split'],
            source_sha256=hashlib.sha256(r['source_id'].encode()).hexdigest()) for r in manifest['rows']]
    (out/'source_split_public.json').write_text(json.dumps(public,indent=2))
    print('PHYSICS_GUIDED_VERSIONS_SAVED',len(public))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True)
    main(Path(p.parse_args().output))

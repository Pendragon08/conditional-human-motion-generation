"""Bounded, separately recorded measurement repair after the historical run."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'results/physics_guided_20261011'
OUT = ROOT / 'results/reset_alignment_v3_20261011'
FAILED = ROOT / 'results/reset_alignment_20261011'
FAILED_V2 = ROOT / 'results/reset_alignment_v2_20261011'
RUNTIME = Path('/data/code/ouyanghao/oyh/humanoid-combat-motion')
PYTHON = RUNTIME / '.venv-pbhc-training/bin/python'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def main():
    assert json.loads((OLD / 'queue_state.json').read_text())['status'] == 'COMPLETE_GATE_NOT_MET', 'Do not overlap old GPU jobs'
    assert not OUT.exists(), 'Keep all attempts; choose a new repair version if this one fails'
    OUT.mkdir()
    source = json.loads((OLD / 'source_manifest.json').read_text())
    rows = []
    for family in ('jump', 'dance'):
        eligible = [r for r in source['rows'] if r['split'] == 'val' and r['family'] == family]
        assert len(eligible) == 12
        rows.extend(eligible[:2])  # Preselected by manifest order, not execution outcome.
    source['rows'] = rows
    manifest = OUT / 'measurement_manifest.json'
    save(manifest, source)
    files = [ROOT/'scripts/reset_aligned_collect_v3.py', ROOT/'scripts/analyze_reset_alignment_v3.py', Path(__file__), ROOT/'scripts/physics_guided_collect.py',
             ROOT/'scripts/physics_guided_core.py', ROOT/'scripts/physics_guided_lineage.py', ROOT/'scripts/r3_closd_isolated_entry.py',
             ROOT/'configs/physics_guided_20261011.json', ROOT/'configs/env-closd-baseline-20261010.json',
             OLD/'versions.json', OLD/'frozen_thresholds.json', OLD/'source_manifest.json', manifest]
    original_versions=json.loads((OLD/'versions.json').read_text())
    generator_hash=sha(ROOT/'third_party/CLoSD/closd/diffusion_planner/save/DiP_multi-target_10steps_context20_predict40/model000300000.pt')
    controller_hash=sha(ROOT/'third_party/CLoSD/output/CLoSD/CLoSD_multitask_finetune/Humanoid.pth')
    assert generator_hash==original_versions['generator_weights_sha256']
    assert controller_hash==original_versions['controller_weights_sha256']
    motion_input_hashes={p:sha(Path(p)) for p in original_versions['external_motion_inputs']}
    assert motion_input_hashes==original_versions['external_motion_inputs']
    upstream_commit=subprocess.check_output(['git','-C',str(ROOT/'third_party/CLoSD'),'rev-parse','HEAD'],text=True).strip()
    assert upstream_commit==original_versions['upstream_commit']
    save(OUT/'versions_before_execution.json', dict(time=time.time(), hashes={str(p):sha(p) for p in files},
         generator_weights_sha256=generator_hash,controller_weights_sha256=controller_hash,
         external_motion_inputs=motion_input_hashes,upstream_commit=upstream_commit,
         upstream_worktree_status=subprocess.check_output(['git','-C',str(ROOT/'third_party/CLoSD'),'status','--porcelain'],text=True),
         study='independent measurement repair; never pooled with historical test results',
         selections='first two validation sources in each family; all three offsets',
         coordinate_fixture_cases_per_batch=12, coordinate_fixture_batches=6,
         full_validation_cases_per_batch=72, full_validation_batches=2, total_expected_attempts=216,
         top_level_environment_seed=11, generation_seeds=[11,12,13],
         initialization='bootstrap reset skips only observation generation before conditions exist; restore before formal player reset; original physics reset unchanged',
         preceding_failed_attempt_budget_sha256=sha(FAILED/'budget_ledger.json'), preceding_failed_attempt_state_sha256=sha(FAILED/'state.json'),
         preceding_failed_v2_budget_sha256=sha(FAILED_V2/'budget_ledger.json'),
         reference_anchor='uniform reference XY translation before immutable targets are constructed; actual state and reference relative poses unchanged'))
    env = os.environ.copy()
    env.update(PYTHONPATH=str(RUNTIME/'envs/closd-r3/site-packages')+':'+str(ROOT/'third_party/CLoSD')+':'+str(ROOT/'third_party/CLoSD/closd'),
               HF_HOME=str(RUNTIME/'cache/closd_hf'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               TORCH_EXTENSIONS_DIR=str(RUNTIME/'cache/closd-r3-torch-extensions'), WANDB_MODE='disabled', WANDB_SILENT='true')
    env['PATH'] = str(RUNTIME/'envs/closd-r3/site-packages/ninja/data/bin')+':'+env['PATH']
    for key in ('R3_CLOSD_SCENE_UDP','R3_CLOSD_WEB_FULL_SCENE','R3_CLOSD_TELEMETRY_UDP'):
        env.pop(key, None)
    ledger=[]
    jobs=[(mode+'_s'+str(seed),mode,seed,manifest) for seed in (11,12,13) for mode in ('planned','native')]
    jobs.extend([('full_val_'+mode+'_s11',mode,11,OLD/'source_manifest.json') for mode in ('planned','native')])
    for label,mode,seed,task_manifest in jobs:
            prior=json.loads((OLD/'budget_ledger.json').read_text())+json.loads((FAILED/'budget_ledger.json').read_text())+json.loads((FAILED_V2/'budget_ledger.json').read_text())
            total=sum(r['seconds'] for r in prior+ledger)
            assert total < 72*3600
            assert sum(r['seconds'] for r in ledger)+sum(r['seconds'] for r in prior if r['stage']=='measurement') < 4*3600
            assert shutil.disk_usage('/data').free >= 20*1024**3
            size=sum(p.stat().st_size for directory in (OLD,FAILED,FAILED_V2,OUT) for p in directory.rglob('*') if p.is_file())
            assert size < 15*1024**3
            gpu=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
            assert int(gpu[0])<2048 and int(gpu[1])<=5, 'No displacement of another workload'
            command=[str(PYTHON),'-u',str(ROOT/'scripts/reset_aligned_collect_v3.py'),'--manifest',str(manifest),
                     '--output',str(OUT/label),'--split','val','--mode',mode,'--seed',str(seed),
                     '--thresholds',str(OLD/'frozen_thresholds.json')]
            save(OUT/'state.json',dict(status='RUNNING',label=label,time=time.time()))
            command[command.index('--manifest')+1]=str(task_manifest)
            request_path=OUT/(label+'_execution_request.json')
            command.extend(['--request-file',str(request_path)])
            save(request_path,dict(command=command,version_hash=sha(OUT/'versions_before_execution.json'),
                 mode=mode,seed=seed,round=0,manifest=sha(task_manifest),
                 config=sha(ROOT/'configs/physics_guided_20261011.json'),
                 thresholds=sha(OLD/'frozen_thresholds.json'),time=time.time()))
            start=time.monotonic()
            with (OUT/(label+'.log')).open('w') as stream:
                try:
                    result=subprocess.run(command,cwd=ROOT/'third_party/CLoSD',env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
                    exit_code=result.returncode
                except subprocess.TimeoutExpired:
                    exit_code=124
            ledger.append(dict(stage='measurement',label=label,seconds=time.monotonic()-start,exit_code=exit_code))
            save(OUT/'budget_ledger.json',ledger)
            print('RESET_ALIGNMENT_BATCH',label,exit_code,flush=True)
            if exit_code:
                save(OUT/'state.json',dict(status='MEASUREMENT_FAILED_RETAINED',label=label,time=time.time()))
                raise SystemExit(exit_code)
    save(OUT/'state.json',dict(status='COLLECTION_COMPLETE_REQUIRES_INDEPENDENT_CHECK',time=time.time()))


if __name__ == '__main__':
    main()

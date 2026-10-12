"""Read-only artifact examination after one observed documented check."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np

ROOT = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
OUT = ROOT/'results/controller_state_diagnosis_v2_20261012'
TRACE = ROOT/'.aris/traces/controller_state_diagnosis_v2_20261012/document_witness'
sys.path.insert(0,str(ROOT/'scripts'))
from verify_execution_lineage_v39_20261012 import guard_batch
from child_input_guard_20261012 import loaded_motion_hash

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

exit_code=int(sys.argv[1])
terminal=(TRACE/'terminal_full.txt').read_text()
checks={}
record=dict(status='FAIL',review_independence='same-family',acceptance_status='provisional',
    runtime_witness_sha256=sha(OUT/'runtime_witness.json') if (OUT/'runtime_witness.json').exists() else None,
    versions_sha256=sha(OUT/'versions.json'),
    actual_command='cd /data/code/ouyanghao/oyh/conditional-human-motion-generation\nPYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python scripts/controller_state_diagnosis_v2_20261012.py --check',
    terminal_exit_code=exit_code,check_invocations=1,prepare_invocations_by_witness=0,run_invocations_by_witness=0,
    terminal_sentinel_present='CONTROLLER_STATE_DOCUMENT_CHECK_PASS' in terminal,
    tool_evidence=dict(exec_session_id=32917,launch_chunk_id='f992fc',terminal_chunk_id='155149',terminal_tool_exit_code=0,terminal_transcript=str(TRACE/'terminal_full.txt'),terminal_transcript_sha256=sha(TRACE/'terminal_full.txt'),launch_wrapper=str(TRACE/'documented_check_once.sh'),launch_wrapper_sha256=sha(TRACE/'documented_check_once.sh')),
    artifact_verifier_correction=dict(prior_verifier_attempt_sha256=sha(TRACE/'artifact_verifier_attempt1_DOCUMENT_WITNESS.json'),prior_verifier_attempt=str(TRACE/'artifact_verifier_attempt1_DOCUMENT_WITNESS.json'),reason='Witness verifier initially assumed case axis0; collector source physics_guided_collect.py lines141-142 and metric_v23 line104 document time/case order. Corrected artifact verifier uses axis1. Original terminal check passed; no documented command rerun and no runtime artifact changed.'),
    scope='Fresh document-following same-family witness of internal validation fixed replay using released multitask CLoSD training state. Simulated human proxy measurements, not candidate efficacy, innovation, G1, downstream learning, independent final test, or hidden PhysX identity.',
    candidate_efficacy_test=False,innovation_confirmed=False,hidden_physx_identity_established=False,
    checks=checks)
try:
    assert exit_code==0
    assert record['terminal_sentinel_present']
    assert 'DOCUMENT_WITNESS_TERMINAL_EXIT=0' in terminal
    checks['actual_terminal_exit_and_sentinel']=True
    v=read(OUT/'versions.json');p=read(OUT/'preparation.json');path=OUT/'raw/multitask'
    wanted=read(OUT/'requests/multitask.json')
    assert wanted['versions']==sha(OUT/'versions.json')
    assert wanted['manifest']==sha(OUT/'source_manifest.json')
    assert wanted['thresholds']==sha(OUT/'frozen_thresholds.json')
    assert wanted['config']==sha(ROOT/'configs/physics_guided_20261011.json')
    assert (wanted['split'],wanted['mode'],wanted['seed'],wanted['round'],wanted['noise_round'],wanted['controller_state'])==('val','replay',11,0,0,'multitask')
    binding=guard_batch(path,wanted)
    checks['execution_request_lineage_receipt']=binding
    assert all(sha(ROOT/name)==digest for name,digest in v['files'].items())
    checks['project_version_bound_source_files']=len(v['files'])
    loaded=read(path/'loaded_input_binding.json');weights=read(path/'loaded_checkpoint_binding.json')
    assert loaded['versions_sha256']==weights['versions_sha256']==sha(OUT/'versions.json')
    assert loaded['verified_source_files']==len(v['files'])
    with np.load(OUT/'references/fixed.npz',allow_pickle=False) as z:
        reference_hash=loaded_motion_hash(z);reference=z['reference'].copy()
    assert reference_hash==loaded['loaded_replay_content_sha256']==wanted['replay_motion']
    assert sha(OUT/'references/fixed.npz')==p['reference_sha256']
    assert sha(OUT/'references/fixed.npz')==sha(ROOT/'results/support_reference_probe_v63_20261012/references/baseline.npz')
    checks['actual_loaded_reference']=dict(content_sha256=reference_hash,file_sha256=p['reference_sha256'],unchanged_parent_reference=True)
    state=v['controller_states']['multitask']
    expected={str((ROOT/state['path']).resolve()):state['sha256'],str((ROOT/v['generator_path']).resolve()):v['generator_weights_sha256']}
    assert weights['actual_loaded_weight_bytes']==expected
    assert all(sha(name)==digest for name,digest in expected.items())
    assert weights['controller_state']=='multitask' and weights['exp_name']==state['exp_name']
    assert weights['request_sha256']==sha(OUT/'requests/multitask.json')
    commit=subprocess.check_output(['git','-C',str(ROOT/'third_party/CLoSD'),'rev-parse','HEAD'],text=True).strip()
    assert commit==weights['actual_upstream_commit']==v['upstream_commit']=='d8950bdb78312fcd2844f336b3fc079e7368c87f'
    assert subprocess.check_output(['git','-C',str(ROOT/'third_party/CLoSD'),'diff','--name-only','HEAD','--','closd'],text=True)==''
    checks['actual_loaded_weight_bytes']=expected
    checks['upstream_commit_and_clean_closd']=commit
    meta=read(path/'metadata.json');rows=read(path/'metrics.json');receipt=read(path/'receipt.json')
    assert len(rows)==len(meta['cases'])==receipt['cases']==72
    assert [r['case_id'] for r in rows]==p['cases']==[r['case_id'] for r in meta['cases']]
    assert len({r['source_id'] for r in rows})==24
    assert set(r['offset_m'] for r in meta['cases'])=={-.15,0.,.15}
    assert meta['original_goal_sha256']==p['original_goal_sha256']
    assert np.array_equal(np.asarray(meta['proposed_goal']),np.asarray(meta['original_goal']))
    assert np.array_equal(np.asarray(meta['proposed_heading']),np.asarray(meta['original_heading']))
    checks['denominator']=dict(rows=72,sources=24,offsets=[-.15,0.,.15],all_ordered_cases_retained=True)
    assert meta['generation_batch_calls']==0
    checks['generation_batch_calls']=0
    assert meta['initial_reset_gate_pass']
    assert all(x==0 for x in meta['initial_rigid_cache_consistency'].values())
    assert all(x==0 for x in meta['manual_reset_vs_first_action_state_max_difference'].values())
    assert sha(path/'initial_states.npz')==meta['initial_state_archive_sha256']
    fields={'_humanoid_root_states','_dof_pos','_dof_vel','_rigid_body_pos','_rigid_body_rot','_rigid_body_vel','_rigid_body_ang_vel'}
    with np.load(path/'initial_states.npz',allow_pickle=False) as z:
        assert all(np.array_equal(z['bootstrap_'+key],z['first_action_'+key]) for key in fields)
    checks['reset_gate']=dict(initial_reset_gate_pass=True,rigid_cache_consistency=meta['initial_rigid_cache_consistency'],bootstrap_to_first_action_seven_tensors_exact=True)
    with np.load(path/'trajectories.npz',allow_pickle=False) as z:
        assert z['reference'].shape==(120,72,24,3) and z['valid'].shape==(120,72)
        checks['actual_array_axes']=dict(reference_shape=list(z['reference'].shape),valid_shape=list(z['valid'].shape),axes=['time','cases','bodies','xyz'])
        canonical=(reference-reference[:,:,0:1,:])+reference[:,:,0:1,:]
        max_error=float(np.max(np.abs(z['reference'][z['valid']]-canonical[z['valid']])))
        assert np.allclose(z['reference'][z['valid']],canonical[z['valid']],atol=1e-12,rtol=0)
    checks['applied_reference_canonical_error_m']=max_error
    runtime=read(OUT/'runtime_witness.json')
    assert runtime['status']=='PASS' and runtime['versions_sha256']==sha(OUT/'versions.json') and runtime['report']['cases']==72
    assert runtime['report']['binding']==binding
    assert runtime['report']['loaded_checkpoint_binding_sha256']==sha(path/'loaded_checkpoint_binding.json')
    assert runtime['report']['loaded_input_binding_sha256']==sha(path/'loaded_input_binding.json')
    assert read(OUT/'state.json')['status']=='CHECK_COMPLETE'
    ret=read(OUT/'multitask_parent_return.json')
    assert ret['child_returncode']==0
    ledger=read(OUT/'budget_ledger.json')
    assert len(ledger)==1 and ledger[0]['label']=='multitask' and ledger[0]['exit_code']==0
    checks['budget_ledger']=ledger
    checks['child_parent_return']=ret
    record['status']='PASS'
except Exception:
    record['failure']=traceback.format_exc()
for name in ('runtime_witness.json','versions.json','preparation.json','state.json','multitask_parent_return.json','budget_ledger.json'):
    path=OUT/name
    if path.exists():record.setdefault('artifact_hashes',{})[name]=sha(path)
if (OUT/'raw/multitask').exists():
    for name in ('metadata.json','metrics.json','lineage.json','receipt.json','loaded_input_binding.json','loaded_checkpoint_binding.json','initial_states.npz','trajectories.npz'):
        path=OUT/'raw/multitask'/name
        if path.exists():record.setdefault('artifact_hashes',{})['raw/multitask/'+name]=sha(path)
with (OUT/'DOCUMENT_WITNESS.json').open('w') as f:json.dump(record,f,indent=2,ensure_ascii=False)
summary='# Controller state diagnosis v2 document witness\n\nStatus: '+record['status']+'\n\nActual documented --check invoked once. Terminal exit: '+str(exit_code)+'. Successful documented sentinel: '+str(record['terminal_sentinel_present'])+'.\n\nReview independence: same-family. Acceptance: provisional.\n\nRuntime witness SHA256: '+str(record['runtime_witness_sha256'])+'\n\nVersions SHA256: '+record['versions_sha256']+'\n\n'+record['scope']+'\n\n```bash\n'+record['actual_command']+'\n```\n\nDetailed runtime bindings, denominator, reset gates, command/tool evidence and hashes are recorded in DOCUMENT_WITNESS.json. Full trace is under .aris/traces/controller_state_diagnosis_v2_20261012/document_witness/.\n'
summary+='\nArtifact verifier correction: the initial verifier used an incorrect extra axis0 case-count assumption. It is preserved in trace as artifact_verifier_attempt1_DOCUMENT_WITNESS.json/.md with its source and transcript. Collector and frozen evaluator use (time,cases,bodies,xyz), observed (120,72,24,3). The corrected read-only examination uses axis1. The documented check was invoked once and passed; runtime files and source were unchanged.\n'
if record.get('failure'):summary+='\nFailure retained; no rerun or repair:\n\n```\n'+record['failure']+'```\n'
with (OUT/'DOCUMENT_WITNESS.md').open('w') as f:f.write(summary)
print(json.dumps(record,indent=2,ensure_ascii=False))
sys.exit(0 if record['status']=='PASS' else 1)

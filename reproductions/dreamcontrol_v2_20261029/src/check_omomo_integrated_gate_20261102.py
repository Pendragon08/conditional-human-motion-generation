"""Exercise release + geometry preflight with a temporary test-only manifest.

Declaring captions accepted in this isolated test copy only verifies wiring;
it does not approve original candidates, use GPU, or create checkpoints.
"""
import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PREVIEW=ROOT/'omomo_geometry_schema_preview_20261102'
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    original=PREVIEW/'manifest.json';pinned=sha(original);data=json.loads(original.read_text())
    if any(r['caption_accepted'] is not False for r in data['records']):raise ValueError('original must be unaccepted')
    for r in data['records']:r['caption_accepted']=True
    data['training_ready']=False;data['test_only_do_not_train']=True
    test=HERE/'test_only_do_not_train';test.mkdir(exist_ok=False);run=test/'empty_run';run.mkdir()
    manifest=test/'synthetic_declarations_manifest.json'
    with manifest.open('x',encoding='utf8') as f:json.dump(data,f,indent=2)
    cfg={'dataset_root':str(ROOT),'manifest':str(manifest),'stats':str(PREVIEW/'stats.npz'),
      'initial_checkpoint':'/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen/save/omnicontrol_ckpt/model_humanml3d.pt',
      'run_dir':str(run),'min_std':.001,'batch_size':2,'seed':1,'max_steps':1,'checkpoint_every':1,
      'guidance_steps':1,'guidance_scale':.5,'learning_rate':1e-5,'weight_decay':0.,'device':'cuda:0','checkpoint_size_estimate':1024**3,
      'geometry_caption_binding':{'evidence_manifest':str(PREVIEW/'geometry_receipts.jsonl'),
          'evidence_sha256':sha(PREVIEW/'geometry_receipts.jsonl'),'state_root':str(ROOT)},
      'omniretarget_release_binding':{'root':str(ROOT),'state_manifest':str(PREVIEW/'state_manifest.json'),
          'archive':str(ROOT/'data/OmniRetarget_Dataset/robot-object.zip'),
          'card':str(ROOT/'omomo_release_evidence_work_20261102/omniretarget_README.md')}}
    config=test/'config.json'
    with config.open('x',encoding='utf8') as f:json.dump(cfg,f,indent=2)
    env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    proc=subprocess.run([sys.executable,str(HERE/'train_robot_prior.py'),'--config',str(config),'--run'],
                        cwd=HERE,env=env,capture_output=True,text=True,timeout=120)
    (test/'stdout.log').write_text(proc.stdout);(test/'stderr.log').write_text(proc.stderr)
    expected='training blocked: source and applicable quality acceptance not yet implemented'
    if proc.returncode==0 or expected not in proc.stderr or list(run.iterdir()) or sha(original)!=pinned:
        raise RuntimeError('integrated negative gate failed; inspect preserved test logs')
    lines=[json.loads(x) for x in proc.stdout.splitlines() if x.startswith('{')]
    if len(lines)!=1:raise ValueError('preflight report missing')
    pre=lines[0];geo=pre['geometry_caption_binding'];release=pre['omniretarget_release_binding']
    if not geo['geometry_checks_passed'] or not release['release_bytes_verified'] or len(geo['cases'])!=1194 or len(release['cases'])!=1194:
        raise ValueError('integrated binding coverage mismatch')
    result={'original_manifest_sha256':pinned,'test_manifest_sha256':sha(manifest),'config_sha256':sha(config),
      'entry_sha256':sha(HERE/'train_robot_prior.py'),'script_sha256':sha(Path(__file__)),
      'stdout_sha256':sha(test/'stdout.log'),'stderr_sha256':sha(test/'stderr.log'),
      'caption_declarations':'test_only_copy_not_accepted','geometry_cases':len(geo['cases']),'release_cases':len(release['cases']),
      'expected_gate_block':True,'run_directory_empty':True,'exit_code':proc.returncode,'training_ready':False,
      'limitations':['This is a negative wiring test; the test manifest must never be treated as an accepted dataset.',
          'Release bytes and nominal timing do not establish source clock, all rights, semantics, or physics.',
          'No GPU, training, or checkpoint was used.']}
    with (test/'report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()

"""Frozen offline CPU verification of exactly uploaded public text encoders."""
import argparse
import json
import os
import pathlib
import subprocess
import time
import traceback
from ldm100k_verified_input import ROOT, read, sha, owned

p = argparse.ArgumentParser()
p.add_argument('--request', required=True)
a = p.parse_args()
sp = owned(a.request)
s = read(sp)
assert os.getuid() == 1013 and sha(__file__) == s['executor_sha256']
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and set(os.sched_getaffinity(0)) == {0, 1, 2, 3}
rp = owned(s['import_receipt']['path'])
assert sha(rp) == s['import_receipt']['sha256']
imported = read(rp)
assert imported['status'] == 'PUBLIC_TMR_TEXT_FILES_SERVER_BYTES_VERIFIED' and imported['output_writers_closed']
assert imported['total_bytes'] == 707323471 and len(imported['files']) == 11
bootpath = owned(s['runtime_receipt']['path'])
assert sha(bootpath) == s['runtime_receipt']['sha256']
boot = read(bootpath)
assert boot['status'] == 'TMR_ISOLATED_RUNTIME_BOOTSTRAP_FINISHED' and boot['output_writers_closed']
target = owned(boot['dependency_dir'])
parent = owned(ROOT / 'evaluator_tmr_20261029')
out = parent / s['run']
out.mkdir(mode=0o700, exist_ok=False)
started = time.monotonic()
receipt = dict(status='STARTING_IMPORTED_TMR_TEXT_MODELS_CPU_VERIFICATION', run=s['run'], source_sha256=sha(__file__),
    request_sha256=sha(sp), import_receipt_sha256=sha(rp), output_writers_closed=False,
    robot_motion_or_babel_text_reads=0, learned_tmr_weight_reads=0, training_optimizer_updates=0,
    evaluator_training_authorized=False, formal_tmr_metrics_available=False)


def save():
    receipt['elapsed_seconds'] = time.monotonic() - started
    temp = out / 'receipt.partial'
    with temp.open('w') as f:
        json.dump(receipt, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(out / 'receipt.json')


def verify_files():
    for item in imported['files']:
        p = owned(item['path'])
        assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256']
        st = p.stat()
        assert dict(device=st.st_dev, inode=st.st_ino, size=st.st_size, mtime_ns=st.st_mtime_ns, uid=st.st_uid) == item['fingerprint']


save()
try:
    verify_files()
    probe = r'''
import sys,json,pathlib,torch
sys.path.insert(0,PRIVATE_TARGET_LITERAL)
from transformers import AutoModel,AutoTokenizer
assert not torch.cuda.is_initialized() and not torch.cuda.is_available()
torch.set_num_threads(4);torch.manual_seed(20261029)
rows=[]
for n in range(2):
 path=str(pathlib.Path(IMPORTED_PARENT_LITERAL)/('text_model_'+str(n)))
 tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=True)
 model=AutoModel.from_pretrained(path,local_files_only=True,use_safetensors=True).eval().requires_grad_(False)
 assert all(torch.isfinite(x).all() for x in model.state_dict().values())
 inputs=tokenizer(['A person walks forward.','A person bends.'],padding=True,truncation=True,return_tensors='pt')
 with torch.inference_mode():
  states=model(**inputs).last_hidden_state
  assert states.shape[:2]==inputs['attention_mask'].shape and states.shape[2]==768 and torch.isfinite(states).all()
  mean=(states*inputs['attention_mask'].unsqueeze(-1)).sum(1)/inputs['attention_mask'].sum(1,keepdim=True)
  normalized=torch.nn.functional.normalize(mean,dim=-1)
 assert torch.allclose(torch.linalg.vector_norm(normalized,dim=-1),torch.ones(2),atol=1e-6)
 assert all(not p.requires_grad for p in model.parameters())
 rows.append(dict(model_type=model.config.model_type,parameters=sum(p.numel() for p in model.parameters()),
  tokens=list(states.shape),finite=True,frozen=True,all_model_state_tensors_finite=True,
  pooled_l2_norms=torch.linalg.vector_norm(normalized,dim=-1).tolist()))
 del model,tokenizer,states,mean,normalized,inputs
assert not torch.cuda.is_initialized()
print(json.dumps(dict(text_models=rows,cuda_initialized=False,actual_training_updates=0,
 private_motion_or_babel_text_reads=0,robot_semantic_encoder_trained=False,formal_tmr_metrics_available=False)))
'''.replace('PRIVATE_TARGET_LITERAL', repr(str(target))).replace('IMPORTED_PARENT_LITERAL', repr(str(rp.parent)))
    import ast
    ast.parse(probe, feature_version=(3, 8))
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
    proc = subprocess.run([str(ROOT / 'env/bin/python'), '-B', '-c', probe], capture_output=True, text=True, env=env, timeout=300)
    for name, content in [('probe_stdout.txt', proc.stdout), ('probe_stderr.txt', proc.stderr)]:
        with (out / name).open('x') as f:
            f.write(content)
    with (out / 'probe_exit.json').open('x') as f:
        json.dump(dict(actual_returncode=proc.returncode), f)
        f.flush()
        os.fsync(f.fileno())
    assert proc.returncode == 0, proc.stderr[-3000:]
    verify_files()
    receipt.update(status='IMPORTED_TMR_TEXT_MODELS_CPU_VERIFIED', output_writers_closed=True,
        capability_probe=json.loads(proc.stdout), total_verified_public_file_bytes=707323471,
        model_directories=[str(rp.parent / ('text_model_' + str(i))) for i in range(2)],
        runtime_dependency_dir=str(target), all_input_fingerprints_unchanged=True)
    save()
except BaseException as error:
    receipt.update(status='FAILED_IMPORTED_TMR_TEXT_MODELS_CPU_VERIFICATION', error=repr(error), traceback=traceback.format_exc())
    save()
    raise

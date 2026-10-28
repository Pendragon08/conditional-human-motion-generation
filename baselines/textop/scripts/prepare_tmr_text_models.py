"""Fetch pinned public text weights once and verify frozen offline CPU encoders.

No robot motion, BABEL text, learned TMR state or evaluator gradients are read.
Failed partial downloads remain in place. HTTP ranges never silently restart.
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from ldm100k_verified_input import ROOT, owned, read, sha

p = argparse.ArgumentParser()
p.add_argument('--request', required=True)
a = p.parse_args()
sp = owned(a.request)
s = read(sp)
assert os.getuid() == 1013 and sha(__file__) == s['executor_sha256']
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and set(os.sched_getaffinity(0)) == {0, 1, 2, 3}
runtime = owned(s['runtime_receipt']['path'])
assert sha(runtime) == s['runtime_receipt']['sha256']
boot = read(runtime)
assert boot['status'] == 'TMR_ISOLATED_RUNTIME_BOOTSTRAP_FINISHED' and boot['output_writers_closed']
target = owned(boot['dependency_dir'])
parent = owned(ROOT / 'evaluator_tmr_20261029')
out = parent / s['run']
out.mkdir(mode=0o700, exist_ok=False)
start = time.monotonic()
receipt = dict(status='STARTING_PINNED_TMR_TEXT_MODEL_PREPARATION', run=s['run'], source_sha256=sha(__file__),
    request_sha256=sha(sp), completed_files=[], pretrained_text_model_downloads_authorized=True,
    robot_motion_or_babel_text_reads=0, learned_tmr_weight_reads=0, training_optimizer_updates=0,
    evaluator_training_authorized=False, formal_tmr_metrics_available=False, output_writers_closed=False)


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


def save():
    receipt['elapsed_seconds'] = time.monotonic() - start
    temp = out / 'receipt.partial'
    with temp.open('w') as f:
        json.dump(receipt, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(out / 'receipt.json')


save()
try:
    total = sum(x['bytes'] for m in s['models'] for x in m['files'])
    assert total == 707323471
    assert shutil.disk_usage(out).free > 2 * 1024**3 + 128 * 1024**2 + total + 16 * 1024**2
    for mi, model in enumerate(s['models']):
        folder = out / ('text_model_' + str(mi))
        folder.mkdir(mode=0o700)
        for item in model['files']:
            assert pathlib.Path(item['name']).name == item['name']
            assert item['url'] == 'https://huggingface.co/' + model['repository'] + '/resolve/' + model['revision'] + '/' + item['name']
            dest = folder / item['name']
            partial = dest.with_name(dest.name + '.partial')
            write(folder / (item['name'] + '.intent.json'), item)
            with partial.open('xb'):
                pass
            for attempt in range(3):
                offset = partial.stat().st_size
                assert offset <= item['bytes']
                headers = {'User-Agent': 'TextOp-pinned-public-model-recovery/1.0', 'Accept-Encoding': 'identity'}
                if offset:
                    headers['Range'] = 'bytes={}-'.format(offset)
                log = dict(attempt=attempt, offset=offset, prefix_sha256=sha(partial), url=item['url'])
                try:
                    if offset < item['bytes']:
                        request = urllib.request.Request(item['url'], headers=headers)
                        with urllib.request.urlopen(request, timeout=90) as remote:
                            log['response_code'] = remote.status
                            if offset:
                                assert remote.status == 206 and remote.headers['Content-Range'].startswith('bytes {}-'.format(offset))
                                assert remote.headers['Content-Range'].endswith('/' + str(item['bytes']))
                            else:
                                assert remote.status == 200
                            with partial.open('ab') as local:
                                while True:
                                    block = remote.read(1024**2)
                                    if not block:
                                        break
                                    assert local.tell() + len(block) <= item['bytes']
                                    assert shutil.disk_usage(out).free > 2 * 1024**3 + 128 * 1024**2 + len(block)
                                    local.write(block)
                                local.flush()
                                os.fsync(local.fileno())
                    assert partial.stat().st_size == item['bytes']
                    if item['sha256']:
                        assert sha(partial) == item['sha256']
                    else:
                        raw = partial.read_bytes()
                        assert hashlib.sha1(('blob %d\0' % len(raw)).encode() + raw).hexdigest() == item['git_blob_sha']
                    partial.replace(dest)
                    log['completed'] = True
                    log['sha256'] = sha(dest)
                    write(folder / (item['name'] + '.attempt' + str(attempt) + '.json'), log)
                    receipt['completed_files'].append(dict(repository=model['repository'], revision=model['revision'],
                        path=str(dest), bytes=dest.stat().st_size, sha256=sha(dest), git_blob_sha=item['git_blob_sha']))
                    save()
                    break
                except (OSError, TimeoutError) as error:
                    log.update(completed=False, error=repr(error), partial_bytes=partial.stat().st_size,
                               partial_sha256=sha(partial))
                    write(folder / (item['name'] + '.attempt' + str(attempt) + '.json'), log)
                    save()
                    if attempt == 2:
                        raise
            assert dest.exists()
    probe = r'''
import sys,json,pathlib,torch
sys.path.insert(0,PRIVATE_TARGET_LITERAL)
from transformers import AutoModel,AutoTokenizer
assert not torch.cuda.is_initialized() and not torch.cuda.is_available()
torch.set_num_threads(4);torch.manual_seed(20261029)
rows=[]
for n in range(2):
 path=str(pathlib.Path(MODEL_PARENT_LITERAL)/('text_model_'+str(n)))
 tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=True)
 model=AutoModel.from_pretrained(path,local_files_only=True,use_safetensors=True).eval().requires_grad_(False)
 inputs=tokenizer(['A person walks forward.','A person bends.'],padding=True,truncation=True,return_tensors='pt')
 with torch.inference_mode():
  states=model(**inputs).last_hidden_state
  assert states.shape[:2]==inputs['attention_mask'].shape and states.shape[2]==768 and torch.isfinite(states).all()
  mean=(states*inputs['attention_mask'].unsqueeze(-1)).sum(1)/inputs['attention_mask'].sum(1,keepdim=True)
  normalized=torch.nn.functional.normalize(mean,dim=-1)
 assert torch.allclose(torch.linalg.vector_norm(normalized,dim=-1),torch.ones(2),atol=1e-6)
 assert all(not p.requires_grad for p in model.parameters())
 rows.append(dict(model_type=model.config.model_type,parameters=sum(p.numel() for p in model.parameters()),
  tokens=list(states.shape),finite=True,frozen=True,pooled_l2_norms=torch.linalg.vector_norm(normalized,dim=-1).tolist()))
 del model,tokenizer,states,mean,normalized,inputs
assert not torch.cuda.is_initialized()
print(json.dumps(dict(text_models=rows,cuda_initialized=False,actual_training_updates=0,
 private_motion_or_babel_text_reads=0,robot_semantic_encoder_trained=False,formal_tmr_metrics_available=False)))
'''.replace('PRIVATE_TARGET_LITERAL', repr(str(target))).replace('MODEL_PARENT_LITERAL', repr(str(out)))
    import ast
    ast.parse(probe, feature_version=(3, 8))
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
    proc = subprocess.run([str(ROOT / 'env/bin/python'), '-B', '-c', probe], capture_output=True, text=True, env=env, timeout=300)
    for name, value in [('probe_stdout.txt', proc.stdout), ('probe_stderr.txt', proc.stderr)]:
        with (out / name).open('x') as f:
            f.write(value)
    write(out / 'probe_exit.json', dict(actual_returncode=proc.returncode))
    assert proc.returncode == 0, proc.stderr[-3000:]
    receipt.update(status='PINNED_TMR_TEXT_MODELS_CPU_VERIFIED', output_writers_closed=True,
        capability_probe=json.loads(proc.stdout), total_verified_public_file_bytes=total,
        model_directories=[str(out / ('text_model_' + str(i))) for i in range(2)],
        data_free_bytes=shutil.disk_usage(ROOT).free)
    save()
except BaseException as error:
    receipt.update(status='FAILED_TMR_TEXT_MODEL_PREPARATION_ALL_PARTIALS_RETAINED', error=repr(error), traceback=traceback.format_exc())
    save()
    raise

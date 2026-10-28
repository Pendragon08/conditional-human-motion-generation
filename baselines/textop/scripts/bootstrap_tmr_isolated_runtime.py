"""Pinned wheels in a new private target plus a real CPU architecture probe.

Never installs into the original environment. No pretrained weights, motion
arrays, text data, CUDA model, gradients or formal evaluator training are used.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from ldm100k_verified_input import ROOT, owned, read, sha

p = argparse.ArgumentParser()
p.add_argument('--request', required=True)
a = p.parse_args()
sp = owned(a.request)
s = read(sp)
assert os.getuid() == 1013 and sha(__file__) == s['executor_sha256']
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and set(os.sched_getaffinity(0)) == {0,1,2,3}
parent = owned(ROOT / 'evaluator_tmr_20261029')
out = parent / s['run']
out.mkdir(mode=0o700, exist_ok=False)
start = time.monotonic()


def original_module_snapshot():
    result = {}
    for name in ['torch', 'transformers', 'tokenizers', 'safetensors', 'huggingface_hub', 'filelock', 'fsspec']:
        found = importlib.util.find_spec(name)
        result[name] = None if found is None else dict(path=found.origin, sha256=sha(found.origin))
    return result


original_modules = original_module_snapshot()
assert original_modules['torch'] is not None and all(original_modules[n] is None for n in original_modules if n != 'torch')
receipt = dict(status='STARTING_TMR_ISOLATED_RUNTIME_BOOTSTRAP', run=s['run'], source_sha256=sha(__file__),
    request_sha256=sha(sp), completed_wheels=[], pretrained_weights_downloaded=False,
    model_weight_payload_reads=0, motion_array_reads=0, raw_text_reads=0, formal_training_updates=0,
    evaluator_training_authorized=False, semantic_metrics_computed=False, output_writers_closed=False)


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
    wheels = out / 'wheels'
    wheels.mkdir(mode=0o700)
    for item in s['packages']:
        assert pathlib.Path(item['filename']).name == item['filename'] and item['filename'].endswith('.whl')
        assert item['url'].startswith('https://files.pythonhosted.org/')
        assert shutil.disk_usage(out).free > 2 * 1024**3 + 128 * 1024**2 + item['bytes']
        dest = wheels / item['filename']
        partial = dest.with_name(dest.name + '.partial')
        assert not dest.exists() and not partial.exists()
        write(wheels / (item['filename'] + '.intent.json'), item)
        count = 0
        with urllib.request.urlopen(item['url'], timeout=90) as remote, partial.open('xb') as local:
            while True:
                block = remote.read(1024**2)
                if not block:
                    break
                count += len(block)
                assert count <= item['bytes'] and shutil.disk_usage(out).free > 2 * 1024**3 + 128 * 1024**2
                local.write(block)
            local.flush()
            os.fsync(local.fileno())
        assert count == item['bytes'] and sha(partial) == item['sha256']
        partial.replace(dest)
        with zipfile.ZipFile(dest) as archive:
            assert archive.testzip() is None
            assert all(not pathlib.PurePosixPath(x.filename).is_absolute() and '..' not in pathlib.PurePosixPath(x.filename).parts for x in archive.infolist())
            uncompressed = sum(x.file_size for x in archive.infolist())
        write(wheels / (item['filename'] + '.outcome.json'), dict(actual_sha256=sha(dest), actual_bytes=count,
            crc_verified=True, uncompressed_bytes=uncompressed))
        receipt['completed_wheels'].append(dict(name=item['name'], version=item['version'], path=str(dest),
            bytes=count, sha256=sha(dest), uncompressed_bytes=uncompressed))
        save()
    assert sum(x['uncompressed_bytes'] for x in receipt['completed_wheels']) < 128 * 1024**2
    target = out / 'dependencies'
    assert not target.exists()
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_NO_CACHE_DIR='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    cmd = [str(ROOT / 'env/bin/python'), '-B', '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-compile',
           '--target', str(target)] + [x['path'] for x in receipt['completed_wheels']]
    with (out / 'pip.log').open('xb') as log:
        pip = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=300)
    write(out / 'pip_exit.json', dict(command=cmd, actual_returncode=pip.returncode))
    assert pip.returncode == 0
    assert target.resolve() == target and target.stat().st_uid == 1013
    installed_bytes = sum(x.stat().st_size for x in target.rglob('*') if x.is_file())
    assert installed_bytes < 128 * 1024**2
    actor = owned(s['actor_path'])
    assert sha(actor) == s['actor_sha256']
    probe = r'''
import importlib.util,importlib.metadata,sys,json,pathlib,torch
sys.path.insert(0,PRIVATE_TARGET_LITERAL)
import transformers,tokenizers,safetensors,huggingface_hub,filelock,fsspec
root=pathlib.Path(PRIVATE_TARGET_LITERAL)
for m in [transformers,tokenizers,safetensors,huggingface_hub,filelock,fsspec]:assert root in pathlib.Path(m.__file__).resolve().parents
assert not torch.cuda.is_initialized() and not torch.cuda.is_available()
spec=importlib.util.spec_from_file_location('tmr_pinned_actor',PINNED_ACTOR_PATH_LITERAL);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
torch.set_num_threads(4);torch.manual_seed(20261029)
motion=mod.ACTORStyleEncoder(69,True,256,1024,6,4,.1,'gelu').eval()
text=mod.ACTORStyleEncoder(768,True,256,1024,6,4,.1,'gelu').eval()
decoder=mod.ACTORStyleDecoder(69,256,1024,6,4,.1,'gelu').eval()
mask=torch.tensor([[1,1,1,0,0],[1,1,1,1,1]],dtype=torch.bool)
with torch.inference_mode():
 m=motion({'x':torch.zeros(2,5,69),'mask':mask})
 t=text({'x':torch.zeros(2,5,768),'mask':mask})
 result=decoder({'z':m[:,0],'mask':mask})
assert m.shape==t.shape==(2,2,256) and result.shape==(2,5,69)
assert torch.isfinite(m).all() and torch.isfinite(t).all() and torch.isfinite(result).all()
assert torch.equal(result[~mask],torch.zeros_like(result[~mask]))
count=sum(p.numel() for model in [motion,text,decoder] for p in model.parameters())
bytes_=sum(p.numel()*p.element_size() for model in [motion,text,decoder] for p in model.parameters())
assert not torch.cuda.is_initialized()
print(json.dumps(dict(actual_cpu_random_architecture_constructions=3,actual_parameter_count=count,parameter_bytes=bytes_,
 conservative_adam_fullstate_upper_bytes=((3*bytes_+8*1024**2+1024**2-1)//1024**2)*1024**2,
 variable_length_forward_finite=True,padded_decoder_output_exact_zero=True,torch_version=torch.__version__,
 transformers_version=transformers.__version__,cuda_initialized=False,pretrained_weight_reads=0,
 actual_motion_or_text_data_reads=0,optimizer_updates=0,formal_training_ready=False)))
'''.replace('PRIVATE_TARGET_LITERAL', repr(str(target))).replace('PINNED_ACTOR_PATH_LITERAL', repr(str(actor)))
    # The replacement tokens intentionally occur only in executable path literals.
    import ast
    ast.parse(probe, feature_version=(3, 8))
    proc = subprocess.run([str(ROOT / 'env/bin/python'), '-B', '-c', probe], capture_output=True,
                          text=True, env=env, timeout=120)
    with (out / 'probe_stdout.txt').open('x') as f:
        f.write(proc.stdout)
    with (out / 'probe_stderr.txt').open('x') as f:
        f.write(proc.stderr)
    write(out / 'probe_exit.json', dict(actual_returncode=proc.returncode))
    assert proc.returncode == 0, proc.stderr[-4000:]
    result = json.loads(proc.stdout)
    after_modules = original_module_snapshot()
    assert after_modules == original_modules
    receipt.update(status='TMR_ISOLATED_RUNTIME_BOOTSTRAP_FINISHED', output_writers_closed=True,
        dependency_dir=str(target), installed_package_bytes=installed_bytes, capability_probe=result,
        original_module_snapshot_before=original_modules, original_module_snapshot_after=after_modules,
        prior_training_environment_installed_into=False, original_runtime_sources_modified=False,
        data_free_bytes=shutil.disk_usage(ROOT).free, home_free_bytes=shutil.disk_usage('/home/ouyanghao').free)
    save()
except BaseException as error:
    receipt.update(status='FAILED_TMR_ISOLATED_RUNTIME_BOOTSTRAP_ALL_PARTIALS_RETAINED', error=repr(error), traceback=traceback.format_exc())
    save()
    raise

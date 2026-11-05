"""Check GPU and CUDA plugin availability without allocating GPU memory.

This is a point-in-time safety check. It cannot reserve the GPU and must be
repeated immediately before a separate GPU worker is started.
"""
import argparse
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path


def assess(index, gpu_rows, process_rows, plugin_available, minimum_free_mib, *, allow_shared=False):
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError('explicit nonnegative GPU index required')
    if isinstance(minimum_free_mib, bool) or not isinstance(minimum_free_mib, int) or minimum_free_mib <= 0:
        raise ValueError('positive free-memory requirement required')
    devices = {}
    for line in gpu_rows.splitlines():
        if not line.strip(): continue
        fields = [field.strip() for field in line.split(',')]
        if len(fields) != 3: raise ValueError('unexpected GPU inventory')
        i, uuid, free = int(fields[0]), fields[1], int(fields[2])
        if i in devices or not uuid: raise ValueError('invalid GPU inventory')
        devices[i] = (uuid, free)
    if index not in devices: raise ValueError('GPU index not found')
    uuid, free = devices[index]
    pids = []
    for line in process_rows.splitlines():
        if not line.strip(): continue
        fields = [field.strip() for field in line.split(',')]
        if len(fields) != 2: raise ValueError('unexpected compute process inventory')
        pid = int(fields[1])
        if fields[0] == uuid: pids.append(pid)
    reasons = []
    if pids and not allow_shared: reasons.append('existing_compute_process')
    if free < minimum_free_mib: reasons.append('insufficient_free_memory')
    if not plugin_available: reasons.append('jax_cuda_plugin_missing')
    return {'gpu_index': index, 'gpu_uuid': uuid, 'free_mib': free,
            'existing_compute_pids': pids, 'shared_use_explicitly_allowed': bool(allow_shared),
            'jax_cuda_plugin_available': bool(plugin_available),
            'minimum_free_mib': minimum_free_mib, 'gpu_retarget_may_start': not reasons,
            'reasons': reasons,
            'limitations': ['Read-only check cannot reserve a device; repeat immediately before start.',
                            'Plugin metadata does not establish CUDA runtime compatibility.',
                            'This does not validate retargeting results or authorize training.']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu-index', required=True, type=int)
    ap.add_argument('--minimum-free-mib', required=True, type=int)
    ap.add_argument('--allow-shared', action='store_true', help='only when user explicitly permits sharing a busy GPU')
    ap.add_argument('--output', required=True, type=Path)
    a = ap.parse_args()
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                          '--format=csv,noheader,nounits'], text=True)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                          '--format=csv,noheader,nounits'], text=True)
    # Does not import JAX or initialize CUDA. GPU plugin modules may exist but
    # still fail at runtime because of missing libraries or version mismatch.
    try:
        plugin_version = importlib.metadata.version('jax-cuda12-plugin')
        pjrt_version = importlib.metadata.version('jax-cuda12-pjrt')
        jax_version = importlib.metadata.version('jax')
        plugin = plugin_version == pjrt_version == jax_version
    except importlib.metadata.PackageNotFoundError:
        plugin = False
    result = assess(a.gpu_index, inventory, processes, plugin, a.minimum_free_mib, allow_shared=a.allow_shared)
    result['cuda_visible_devices_at_check'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    with a.output.open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()

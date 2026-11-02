"""Read resource usage without loading the simulator or altering any process."""
import os
import pathlib
import shutil
import subprocess


def sample_resources(root, child_group=None, minimum_ram_bytes=2 * 1024**3,
                     minimum_disk_bytes=2 * 1024**3):
    root = pathlib.Path(root)
    assert root.resolve() == root and root.stat().st_uid == os.getuid()
    memory = [int(x.split()[1]) * 1024 for x in pathlib.Path('/proc/meminfo').read_text().splitlines()
              if x.startswith('MemAvailable:')][0]
    free = shutil.disk_usage(root).free
    if memory < minimum_ram_bytes or free < minimum_disk_bytes:
        raise RuntimeError('Evaluator resource reserve crossed: available_ram_bytes={}, free_disk_bytes={}, '
                           'minimum_ram_bytes={}, minimum_disk_bytes={}'.format(
                               memory, free, minimum_ram_bytes, minimum_disk_bytes))
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.free,utilization.gpu',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True,
                         check=True, timeout=10)
    owned = []
    if child_group is not None:
        for proc in pathlib.Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                if proc.stat().st_uid != os.getuid() or os.getpgid(int(proc.name)) != child_group:
                    continue
                status = (proc / 'status').read_text().splitlines()
                owned.append({'pid': int(proc.name), 'group': child_group,
                              'state_and_rss': [s for s in status if s.startswith(('State:', 'VmRSS:'))],
                              'cpu_affinity': sorted(os.sched_getaffinity(int(proc.name)))})
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
    applications = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                                   '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                  check=True, timeout=10)
    pids = {x['pid'] for x in owned}
    per_process = []
    for line in applications.stdout.splitlines():
        fields = [x.strip() for x in line.split(',')]
        if len(fields) == 3 and fields[0].isdigit() and int(fields[0]) in pids:
            per_process.append({'pid': int(fields[0]), 'gpu_uuid': fields[1],
                                'used_gpu_memory_mib': fields[2]})
    return {'available_host_ram_bytes': memory, 'data_free_bytes': free,
            'all_device_usage_including_other_jobs': gpu.stdout.strip(),
            'owned_child_group': child_group, 'owned_group_processes': owned,
            'owned_group_gpu_memory': per_process}

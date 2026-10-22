"""Checkpoint backup/evaluation worker for the explicitly authorized full run.

Runs locally with existing SSH and gh authentication. No credentials are copied
to the server. Releases are append-only; no asset is clobbered. A server file is
removed only after its exact digest is published and a newer checkpoint is
published. Latest and milestone local copies are retained. All simulation scores
are labeled diagnostics, never the paper's benchmark.
"""
import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import time
import traceback

if not __debug__:
    raise RuntimeError('Optimized Python is unsupported: checkpoint safety guards must remain enabled')

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path); partial = path.with_suffix(path.suffix + '.partial')
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(partial, path)

def freeze_package(directory, package):
    """A retry must upload the exact immutable bytes from the first attempt."""
    directory, package = Path(directory), Path(package)
    if package.exists(): return
    partial = package.with_suffix(package.suffix + '.partial')
    with tarfile.open(partial, 'w:gz') as tar:
        for path in sorted(directory.iterdir()):
            if path.suffix == '.pt' or path == package or path.suffix == '.partial': continue
            tar.add(path, arcname=path.name)
    os.replace(partial, package)

def seal_receipt(path):
    path = Path(path)
    atomic_json(path.with_suffix('.integrity.json'), {'bytes':path.stat().st_size, 'sha256':digest(path)})

def check_receipt(path):
    path = Path(path); metadata = path.with_suffix('.integrity.json')
    if not metadata.exists(): raise RuntimeError('Unsealed local evaluation receipt; preserve for manual verification')
    expected = json.loads(metadata.read_text(encoding='utf-8'))
    if expected['bytes'] != path.stat().st_size or expected['sha256'] != digest(path):
        raise RuntimeError('Local evaluation receipt changed after capture; refusing to infer results')

class Worker:
    def __init__(self, config):
        self.c = config; self.root = Path(config['local_output']); self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / 'worker_state.json'
        self.state = json.loads(self.state_path.read_text(encoding='utf-8')) if self.state_path.exists() else {'processed': {}, 'events': []}
        self.ssh = [config['ssh_executable'], '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=4', '-i', config['ssh_key'], config['ssh_host']]

    def log(self, kind, **fields):
        row = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'kind': kind, **fields}
        with (self.root / 'worker_events.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(row, ensure_ascii=False) + '\n')
        self.state.update(last_event=row, worker_pid=os.getpid()); atomic_json(self.state_path, self.state)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def call(self, args, timeout=3600, input=None):
        r = subprocess.run(args, input=input, capture_output=True, text=True, encoding='utf-8', timeout=timeout)
        if r.returncode: raise RuntimeError(f'Command exit {r.returncode}: {r.stderr[-4000:]}')
        return r.stdout

    def remote(self, script, timeout=120):
        return self.call(self.ssh + [self.c['python'] + ' -'], input=script, timeout=timeout)

    def snapshot(self):
        root = repr(self.c['server_output'])
        return json.loads(self.remote("from pathlib import Path, PurePosixPath\nimport json,time\nr=Path(" + root + ")\n"
            "out={'statuses':[], 'manifests':[], 'health':[], 'utc_epoch':time.time()}\n"
            "for p in sorted(r.glob('status_rank*.json')):\n"
            " try:\n  s=json.loads(p.read_text());s['status_file_mtime_epoch']=p.stat().st_mtime;out['statuses'].append(s)\n"
            " except (ValueError,OSError):pass\n"
            "for p in sorted(r.glob('health_rank*.json')):\n"
            " try:out['health'].append(json.loads(p.read_text()))\n"
            " except (ValueError,OSError):pass\n"
            "for p in sorted(r.glob('checkpoint_ready_*.json')):\n"
            " try:out['manifests'].append(json.loads(p.read_text()))\n"
            " except (ValueError,OSError):pass\n"
            "import subprocess\n"
            "matches=subprocess.run(['pgrep','-f'," + repr(self.c.get('training_script','/code/scripts/train_full.py')) + "],capture_output=True,text=True).stdout.split()\n"
            "out['training_pids']=[]\n"
            "for pid in matches:\n"
            " try:\n  args=Path('/proc/'+pid+'/cmdline').read_bytes().decode().split('\\x00')\n"
            "  if '--output' in args and args[args.index('--output')+1]==str(r):out['training_pids'].append(int(pid))\n"
            " except (OSError,ValueError,IndexError):pass\n"
            "out['training_process_running']=bool(out['training_pids'])\n"
            "print(json.dumps(out))\n"))

    def assets(self):
        release = json.loads(self.call(['gh', 'api', f"repos/{self.c['github_repo']}/releases/tags/{self.c['release_tag']}"]))
        result = {}; page = 1
        while True:
            items=json.loads(self.call(['gh','api',f"repos/{self.c['github_repo']}/releases/{release['id']}/assets?per_page=100&page={page}"]))
            for item in items:
                if item['name'] in result:raise RuntimeError('Duplicate Release asset name; refusing ambiguous backup state')
                result[item['name']]=item
            if len(items)<100:return result
            page+=1

    def upload(self, path, assets):
        path = Path(path); expected = digest(path); size = path.stat().st_size
        if size >= 2 * 1024**3: raise RuntimeError('Asset exceeds configured GitHub per-file budget')
        if path.name not in assets:
            self.call(['gh', 'release', 'upload', self.c['release_tag'], str(path), '--repo', self.c['github_repo']])
            assets.update(self.assets())
        item = assets[path.name]
        if item['state'] != 'uploaded' or item['size'] != size or item.get('digest') != 'sha256:' + expected:
            raise RuntimeError('Existing or uploaded GitHub asset digest mismatch; refusing overwrite/deletion')
        return {'name': path.name, 'bytes': size, 'sha256': expected, 'url': item['browser_download_url']}

    def download(self, remote_path, local_path, expected=None, size=None):
        local_path = Path(local_path); local_path.parent.mkdir(parents=True, exist_ok=True)
        if expected is None:
            script='from pathlib import Path\nimport hashlib,json\np=Path('+repr(remote_path)+')\nh=hashlib.sha256()\n'
            script+='with p.open("rb") as f:\n for b in iter(lambda:f.read(1048576),b""):h.update(b)\n'
            script+='print(json.dumps({"bytes":p.stat().st_size,"sha256":h.hexdigest()}))\n'
            info=json.loads(self.remote(script,timeout=300));expected=info['sha256'];size=info['bytes']
        if local_path.exists() and local_path.stat().st_size == size and digest(local_path) == expected:
            return
        if local_path.exists(): raise RuntimeError('Existing final local file mismatches; preserve it for review')
        partial = local_path.with_suffix(local_path.suffix + '.partial')
        args = [self.c['scp_executable'], '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30',
            '-i', self.c['ssh_key'], self.c['ssh_host'] + ':' + remote_path, str(partial)]
        self.call(args)
        if expected and (partial.stat().st_size != size or digest(partial) != expected):
            raise RuntimeError('Downloaded checkpoint digest mismatch')
        os.replace(partial, local_path)

    def verify_model_identity(self, row):
        script='from pathlib import Path\nimport hashlib,json\np=Path('+repr(row['file'])+')\nh=hashlib.sha256()\n'
        script+='with p.open("rb") as f:\n for b in iter(lambda:f.read(1048576),b""):h.update(b)\n'
        script+='print(json.dumps({"sha256":h.hexdigest(),"bytes":p.stat().st_size}))\n'
        current=json.loads(self.remote(script,timeout=180))
        if current != {'sha256':row['sha256'],'bytes':row['bytes']}:
            raise RuntimeError('Evaluation model bytes differ from the verified checkpoint')
        return current

    def evaluate(self, manifest, directory):
        updates = manifest['optimizer_updates']; results = []
        if updates not in self.c.get('evaluation_updates', [1000, 10001, 1000000]) and updates % 100000 not in [0, 1]: return results
        model = next(x['file'] for x in manifest['files'] if Path(x['file']).name.startswith('model'))
        bound = self.c.get('bind_evaluation_model_sha256', False)
        model_row = next(x for x in manifest['files'] if x['file']==model)
        cases = [('stand_still', 'stand still'), ('walk_forward', 'walk forward'), ('punch', 'perform a punch')]
        for case, prompt in cases:
            for seed in [42,43,44]:
                label = f"u{updates:09d}_{case}_s{seed}"
                if bound: label += '_'+model_row['sha256'][:16]
                base_label = label
                # A receipt left before its post-run identity check is unbound.
                # Preserve it and its remote directory; never promote it on retry.
                if bound:
                    for attempt in range(100):
                        label = base_label + (f'_retry{attempt:03d}' if attempt else '')
                        trial_path = directory / (label + '_receipt.json')
                        binding_path = directory / (label + '_checkpoint_identity.json')
                        if trial_path.exists():
                            if binding_path.exists(): break
                            continue
                        destination = self.c['evaluation_root'] + '/' + label
                        query = 'from pathlib import Path\nprint(Path(' + repr(destination) + ').exists())\n'
                        if self.remote(query).strip() == 'True': continue
                        break
                    else: raise RuntimeError('Too many preserved unbound diagnostic attempts')
                    self.verify_model_identity(model_row)
                destination = self.c['evaluation_root'] + '/' + label
                cmd = [self.c['python'], self.c['runtime_root'] + '/code/scripts/run_closed_loop.py',
                    '--runtime-root', self.c['runtime_root'], '--checkpoint', model, '--output', destination,
                    '--steps', '400', '--prompt', prompt, '--seed', str(seed)]
                if case == 'stand_still' and seed == 42: cmd += ['--video']
                import shlex
                shell = 'CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl ' + shlex.join(cmd)
                path = directory / (label + '_receipt.json')
                if path.exists(): check_receipt(path)
                # Freeze a failure receipt too: never retry into an existing output
                # directory or infer physical execution from an SSH exit code.
                if not path.exists():
                    query = 'from pathlib import Path\nprint(Path(' + repr(destination + '/receipt.json') + ').exists())\n'
                    receipt_exists = self.remote(query).strip() == 'True'
                    error = None
                    if not receipt_exists:
                        try:
                            r = subprocess.run(self.ssh + [shell], capture_output=True, text=True, encoding='utf-8', timeout=900)
                            stdout, stderr = r.stdout, r.stderr
                            if r.returncode: error = f'Process exit {r.returncode}'
                        except subprocess.TimeoutExpired as exc:
                            stdout, stderr = exc.stdout or '', exc.stderr or ''
                            if isinstance(stdout, bytes): stdout = stdout.decode('utf-8', errors='replace')
                            if isinstance(stderr, bytes): stderr = stderr.decode('utf-8', errors='replace')
                            error = 'SSH evaluation supervisor timed out at 900 seconds'
                        (directory / (label + '_stdout.txt')).write_text(stdout, encoding='utf-8')
                        (directory / (label + '_stderr.txt')).write_text(stderr, encoding='utf-8')
                        receipt_exists = self.remote(query).strip() == 'True'
                    if receipt_exists:
                        self.download(destination + '/receipt.json', path)
                    else:
                        atomic_json(path, {'status':'EVALUATION_LAUNCH_FAILED', 'physics_evaluated':False,
                            'error':error or 'No remote receipt was produced',
                            'scope':'Supervisor failure; no physical execution or scores inferred.'})
                    seal_receipt(path)
                data = json.loads(path.read_text(encoding='utf-8'))
                if data.get('status') != 'EVALUATION_LAUNCH_FAILED':
                    self.download(destination + '/receipt.json', path)
                if bound:
                    identity=self.verify_model_identity(model_row)
                    data['checkpoint_identity_verified']=identity
                    identity_path=directory/(label+'_checkpoint_identity.json')
                    binding={'model':identity,'receipt':{'bytes':path.stat().st_size,'sha256':digest(path)}}
                    if identity_path.exists() and json.loads(identity_path.read_text())!=binding:
                        raise RuntimeError('Cached evaluation checkpoint identity changed')
                    if not identity_path.exists():atomic_json(identity_path,binding)
                if data.get('status') == 'PHYSICAL_CLOSED_LOOP_EXECUTED':
                    feedback_path = directory / (label + '_feedback.json')
                    if not feedback_path.exists():
                        remote_feedback = destination + '/feedback_verification.json'
                        command = [self.c['python'], self.c['runtime_root'] + '/code/scripts/verify_feedback.py',
                            '--run-dir', destination, '--output', remote_feedback]
                        script = 'from pathlib import Path\nimport subprocess,json\np=Path('+repr(remote_feedback)+')\n'
                        script += 'if not p.exists():\n r=subprocess.run('+repr(command)+',capture_output=True,text=True)\n'
                        script += ' if not p.exists():p.write_text(json.dumps({"status":"FEEDBACK_VERIFIER_FAILED","error":r.stderr[-4000:],"exit_code":r.returncode}))\n'
                        self.remote(script, timeout=120)
                    self.download(destination + '/feedback_verification.json', feedback_path)
                    data['feedback_verification'] = json.loads(feedback_path.read_text(encoding='utf-8'))
                results.append({'case': case, 'seed': seed, **data})
                # Preserve every generated simulation artifact, including failures.
                archive_remote = self.c['evaluation_root'] + '/' + label + '.tar.gz'
                script = 'from pathlib import Path\nimport os,tarfile\np=Path(' + repr(destination) + ')\na=Path(' + repr(archive_remote) + ')\n'
                script += "if not a.exists():\n partial=a.with_suffix(a.suffix+'.partial')\n with tarfile.open(partial,'w:gz') as t:\n  if p.exists():t.add(p,arcname=p.name)\n os.replace(partial,a)\nprint(a.stat().st_size)\n"
                self.remote(script)
                self.download(archive_remote, directory / (label + '.tar.gz'))
        return results

    def report(self, manifest, directory, evaluations, assets):
        updates = manifest['optimizer_updates']; passed = [e for e in evaluations if e.get('status') == 'PHYSICAL_CLOSED_LOOP_EXECUTED']
        upright = [e for e in passed if e.get('upright_entire_rollout_height_proxy')]
        rows = ['# ReactiveBFM 完整课程训练阶段报告', '', f'实际优化器更新：{updates:,} / 1,000,000。',
            self.c.get('training_description', '配置：双 RTX A6000，512×2，全局 batch1024，400k warmup，EMA、BF16、激活重算；使用官方三子集组合与各自归一化。'),
            '该快照不等于已复现原文性能；目标到达数据/官方完整评测仍未公开。', '',
            '## 检查点备份', '', '下列文件大小及 GitHub 服务端 SHA256 与服务器保存记录一致。', '']
        rows += [f"- {x['name']}: {x['bytes']} bytes, SHA256 `{x['sha256']}`" for x in assets]
        rows += ['', '## MuJoCo 诊断', '', f'本阶段执行 {len(evaluations)} 个测试；完整执行闭环 {len(passed)} 个。',
            ('未进行本阶段仿真评估。' if not evaluations else f'根部高度始终≥0.4m 的诊断数量：{len(upright)}/{len(evaluations)}。这是高度阈值代理，不能作为原文成功率。'),
            '未评估阶段不计算动作成功率。CPU 原生 PyTorch 跟踪器沿用官方网络/适配器，未使用 TensorRT，不声称实时性能。', '']
        for e in evaluations:
            rows.append(f"- {e['case']} seed{e['seed']}: {e.get('status')}; 控制步 {e.get('policy_steps', e.get('physics_steps',0))}; 最低根部高度 {e.get('minimum_root_height_m','NA')}m。")
            if 'feedback_verification' in e:
                rows.append(f"  状态反馈一致性检查：{e['feedback_verification'].get('status')}；仅检查真实仿真状态到生成前缀的接线，不代表动作质量。")
        rows += ['', '## 数据与保存', '', '训练数据文件保留授权本地/服务器；GitHub 保存来源、划分、校验值。',
            '本检查点备份前不会清理其副本；仅已验证备份的旧检查点可按保留策略清理，最新检查点始终保留。',
            '仿真 NPZ/视频为模型与物理环境生成的数据，不包含原始训练动作包。', '']
        path = directory / f'full_checkpoint_{manifest["official_step"]:09d}_report.md'
        path.write_text('\n'.join(rows), encoding='utf-8'); return path

    def capture_recovery_trace(self, manifest, directory):
        if not self.c.get('capture_recovery_trace', False): return
        step = manifest['official_step']; updates = manifest['optimizer_updates']
        script = '''from pathlib import Path
import hashlib,io,json,os,tarfile
r=Path(ROOT); archive=r/NAME
mandatory=['recovery_launch.json','source_pins_verified.json','status_rank0.json','status_rank1.json']
for name in mandatory:
 if not (r/name).is_file():raise RuntimeError('Missing mandatory trace member: '+name)
health={}
for rank in [0,1]:
 p=r/('health_rank'+str(rank)+'.jsonl');chosen=[]
 if not p.is_file():raise RuntimeError('Missing rank health trace')
 for line in p.read_bytes().splitlines(keepends=True):
  try:row=json.loads(line)
  except (ValueError,UnicodeError):raise RuntimeError('Malformed rank health row; retry after writer finishes')
  if row.get('loop_step',-1)<=STEP and row.get('optimizer_updates',0)<=UPDATES:chosen.append(line)
 if not chosen or json.loads(chosen[-1])['optimizer_updates']!=UPDATES:raise RuntimeError('Rank trace lacks checkpoint endpoint')
 endpoint=json.loads(chosen[-1])
 if endpoint.get('loop_step')!=STEP or not endpoint.get('step_validated') or not endpoint.get('ema_validated'):raise RuntimeError('Checkpoint endpoint is not validated')
 if endpoint.get('optimizer_step_calls')!=UPDATES or endpoint.get('validated_optimizer_updates')!=UPDATES:raise RuntimeError('Checkpoint endpoint counters disagree')
 health[rank]=b''.join(chosen)
if not archive.exists():
 partial=archive.with_suffix('.partial')
 members={}
 with tarfile.open(partial,'w:gz') as tar:
  for name in mandatory+['supervisor_stdout.txt','supervisor_stderr.txt','supervisor_exit.json']:
   p=r/name
   if p.is_file():
    data=p.read_bytes();item=tarfile.TarInfo(name);item.size=len(data);tar.addfile(item,io.BytesIO(data));members[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
  for rank,data in health.items():
   name='health_rank'+str(rank)+'_until_checkpoint.jsonl';item=tarfile.TarInfo(name);item.size=len(data);tar.addfile(item,io.BytesIO(data));members[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
  proof={'checkpoint_manifest':CHECKPOINT,'members':members,'scope':'Status files are capture-time observations; per-rank health ends at the checkpoint.'}
  data=json.dumps(proof,sort_keys=True,indent=2).encode();item=tarfile.TarInfo('trace_integrity.json');item.size=len(data);tar.addfile(item,io.BytesIO(data))
 os.replace(partial,archive)
with tarfile.open(archive) as tar:
 proof=json.load(tar.extractfile('trace_integrity.json'))
 if proof['checkpoint_manifest']!=CHECKPOINT:raise RuntimeError('Frozen trace checkpoint changed')
 for name,expected in proof['members'].items():
  data=tar.extractfile(name).read()
  if {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}!=expected:raise RuntimeError('Frozen trace member digest mismatch')
 for name in mandatory:
  if name not in tar.getnames():raise RuntimeError('Frozen trace is incomplete')
 for rank in [0,1]:
  data=tar.extractfile('health_rank'+str(rank)+'_until_checkpoint.jsonl').read()
  row=json.loads(data.splitlines()[-1])
  if row['optimizer_updates']!=UPDATES or row.get('loop_step')!=STEP or not row.get('step_validated') or not row.get('ema_validated') or row.get('optimizer_step_calls')!=UPDATES or row.get('validated_optimizer_updates')!=UPDATES:raise RuntimeError('Frozen trace endpoint changed')
print(str(archive))
'''.replace('ROOT',repr(self.c['server_output'])).replace('NAME',repr(f'training_trace_{step:09d}.tar.gz')).replace('STEP',str(step)).replace('UPDATES',str(updates)).replace('CHECKPOINT',repr(manifest))
        remote = self.remote(script).strip()
        self.download(remote, Path(directory)/Path(remote).name)

    def cleanup_server_older(self, newest, manifests, assets):
        candidates = []
        for m in manifests:
            if m['optimizer_updates'] >= newest['optimizer_updates']: continue
            if str(m['official_step']) not in self.state['processed']: continue
            for row in m['files']:
                name = Path(row['file']).name; item = assets.get(name)
                if not item or item.get('digest') != 'sha256:' + row['sha256'] or item['size'] != row['bytes']:
                    raise RuntimeError('Cleanup blocked: published digest unavailable')
                candidates.append(row)
        if not candidates: return
        payload = json.dumps({'root': self.c['server_output'] + '/checkpoints', 'files': candidates})
        script = 'from pathlib import Path, PurePosixPath\nimport json,hashlib,re\na=json.loads(' + repr(payload) + ')\nroot=Path(a["root"]).resolve()\nremoved=[]\n'
        script += r'''for row in a['files']:
 p=Path(row['file'])
 if not p.exists():continue
 if p.is_symlink() or p.resolve().parent!=root:raise RuntimeError('Cleanup path escaped checkpoint directory')
 if not re.fullmatch(r'(model|opt)\d{9}\.pt',p.name):raise RuntimeError('Unexpected cleanup filename')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 if h.hexdigest()!=row['sha256'] or p.stat().st_size!=row['bytes']:raise RuntimeError('Cleanup file changed after verified backup')
 p.unlink();removed.append(str(p))
print(json.dumps(removed))
'''
        removed = json.loads(self.remote(script, timeout=300))
        if removed: self.log('verified_old_server_checkpoint_copies_removed', removed=removed)

    def run(self, once=False):
        while True:
            try:
                snapshot = self.snapshot(); self.log('snapshot', statuses=[{'rank':s['rank'], 'status':s['status'], 'updates':s.get('optimizer_updates',0)} for s in snapshot['statuses']])
                atomic_json(self.root / 'current_training_status.json', snapshot)
                assets = self.assets()
                for m in sorted(snapshot['manifests'], key=lambda x:x['optimizer_updates']):
                    if str(m['official_step']) in self.state['processed']: continue
                    directory = self.root / f'checkpoint_{m["official_step"]:09d}'; directory.mkdir(exist_ok=True)
                    verified = []
                    assert len(m['files'])==2
                    assert {PurePosixPath(row['file']).name for row in m['files']} == {f'model{m["official_step"]:09d}.pt', f'opt{m["official_step"]:09d}.pt'}
                    for row in m['files']:
                        remote_name=PurePosixPath(row['file'])
                        assert str(remote_name.parent)==self.c['server_output']+'/checkpoints'
                        assert remote_name.name in [f'model{m["official_step"]:09d}.pt',f'opt{m["official_step"]:09d}.pt']
                        assert re.fullmatch('[0-9a-f]{64}',row['sha256']) and 0<row['bytes']<2*1024**3
                        path = directory / remote_name.name
                        self.download(row['file'], path, row['sha256'], row['bytes'])
                        verified.append(self.upload(path, assets))
                    atomic_json(directory / f'checkpoint_{m["official_step"]:09d}_manifest.json', m)
                    raw_manifest = directory / 'raw_ready_manifest.json'
                    self.download(self.c['server_output'] + f'/checkpoint_ready_{m["official_step"]:09d}.json', raw_manifest)
                    if json.loads(raw_manifest.read_text(encoding='utf-8')) != m:
                        raise RuntimeError('Raw ready manifest differs from snapshot')
                    atomic_json(directory / 'raw_ready_manifest_integrity.json', {'sha256':digest(raw_manifest), 'bytes':raw_manifest.stat().st_size})
                    self.download(self.c['server_output'] + '/checkpoints/args.json', directory / 'args.json')
                    self.capture_recovery_trace(m, directory)
                    limit=m['optimizer_updates']
                    script='import json\nfrom pathlib import Path, PurePosixPath\np=Path('+repr(self.c['server_output']+'/metrics.jsonl')+')\n'
                    script+='if p.exists():\n for line in p.read_text().splitlines():\n  try:r=json.loads(line)\n  except ValueError:continue\n  if r.get("optimizer_updates",0)<='+str(limit)+':print(json.dumps(r))\n'
                    (directory/'training_metrics.jsonl').write_text(self.remote(script),encoding='utf-8')
                    if not (directory/'observed_status_at_backup.json').exists():
                        atomic_json(directory/'observed_status_at_backup.json',self.snapshot())
                    evaluations = self.evaluate(m, directory)
                    report = self.report(m, directory, evaluations, verified)
                    package = directory / f'checkpoint_{m["official_step"]:09d}_evidence.tar.gz'
                    freeze_package(directory, package)
                    self.upload(package, assets); self.upload(report, assets)
                    self.state['processed'][str(m['official_step'])] = {'optimizer_updates':m['optimizer_updates'], 'assets':verified,
                        'evidence':digest(package), 'report':digest(report), 'evaluated_trials':len(evaluations)}
                    self.log('checkpoint_backup_and_diagnostics_complete', step=m['official_step'], updates=m['optimizer_updates'])
                    shutil.copy2(report, self.root / '最新训练阶段报告.md')
                    self.cleanup_server_older(m, snapshot['manifests'], assets)
                    # Local immutable evidence/report remains; retain model+optimizer
                    # for latest, early1000 and each100k milestone. Github keeps all.
                    for other in snapshot['manifests']:
                        u=other['optimizer_updates']; key=str(other['official_step'])
                        if u>=m['optimizer_updates'] or u==1000 or u%100000 in [0,1] or key not in self.state['processed']:continue
                        folder=self.root/f'checkpoint_{other["official_step"]:09d}'
                        for row in other['files']:
                            path=folder/Path(row['file']).name
                            if path.exists():
                                if path.is_symlink() or path.resolve().parent!=folder.resolve() or not folder.resolve().is_relative_to(self.root.resolve()):raise RuntimeError('Local cleanup path escaped checkpoint folder')
                                if assets[path.name]['digest']!='sha256:'+digest(path) or digest(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:raise RuntimeError('Local cleanup digest mismatch')
                                path.unlink();self.log('verified_old_local_checkpoint_copy_removed', path=str(path))
                # A transient SSH failure after publication must not permanently
                # disable verified cleanup and deadlock the trainer's disk guard.
                published = [m for m in snapshot['manifests'] if str(m['official_step']) in self.state['processed']]
                if published:
                    self.cleanup_server_older(max(published, key=lambda m:m['optimizer_updates']), snapshot['manifests'], assets)
                statuses=snapshot['statuses']
                ranks_ok = len(statuses)==2 and {s.get('rank') for s in statuses}=={0,1} and all(s.get('world_size')==2 for s in statuses)
                if not ranks_ok:self.log('incomplete_status_rank_set_no_completion',ranks=[s.get('rank') for s in statuses])
                stale=[s['rank'] for s in statuses if s['status']=='TRAINING' and snapshot['utc_epoch']-s.get('status_file_mtime_epoch',snapshot['utc_epoch'])>self.c.get('stale_status_seconds',1800)]
                if stale:
                    stamp=int(max(s.get('status_file_mtime_epoch',0) for s in statuses if s['rank'] in stale))
                    notice=self.root/f'stale_training_status_{stamp}.json'
                    if not notice.exists():atomic_json(notice,snapshot)
                    self.upload(notice,assets);self.log('stale_training_status_no_automatic_restart',ranks=stale)
                completed = ranks_ok and all(s.get('full_training_completed') and s.get('optimizer_updates',0)>=1000000 for s in statuses)
                if completed:
                    if any(m['optimizer_updates']>=1000000 and str(m['official_step']) in self.state['processed'] for m in snapshot['manifests']):
                        self.log('full_schedule_backed_up_and_diagnostics_recorded');return
                if any(s['status']=='FAILED' for s in statuses):
                    failure=self.root/('training_failure_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')
                    atomic_json(failure,snapshot);self.upload(failure,assets)
                    self.log('training_failed_worker_stopped_without_restart');return
                if not snapshot.get('training_process_running') and not completed:
                    failure=self.root/('training_process_absent_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')
                    atomic_json(failure,snapshot);self.upload(failure,assets)
                    self.log('training_process_absent_results_incomplete');return
                if once:return
            except Exception as exc:
                self.log('worker_error_no_unverified_cleanup', error=str(exc), traceback=traceback.format_exc())
                if once:raise
            time.sleep(self.c['poll_seconds'])

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--once',action='store_true');a=p.parse_args()
    c=json.loads(a.config.read_text(encoding='utf-8'));worker=Worker(c)
    lock=(worker.root/'worker.lock').open('a+b');lock.seek(0)
    if os.name=='nt':
        import msvcrt
        if lock.read(1)==b'':lock.write(b'1');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        # Prevent idle sleep while backing up; no display/power-plan change.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    else:
        import fcntl
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    worker.log('worker_started', code_sha256=digest(__file__))
    try:worker.run(once=a.once)
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()

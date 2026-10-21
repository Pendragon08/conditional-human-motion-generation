"""Write human-readable acquisition progress; optionally publish a final metadata snapshot."""
import argparse
import ctypes
import datetime
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def alive(pid):
    if os.name != 'nt':
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        kernel.GetExitCodeProcess(wintypes.HANDLE(handle), ctypes.byref(code))
        return code.value == 259
    finally:
        kernel.CloseHandle(wintypes.HANDLE(handle))


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def generate(base, queue, job, pid):
    running = alive(pid)
    source = base / 'results/amass_remaining_20261021'
    progress = read(job / 'progress.json') or {'items': []}
    phases = {x['subset']: x for x in progress['items']}
    rows = []
    model = None
    for item in queue['items']:
        name = item['subset']
        local = None
        for folder in (job, source / 'acquisition_v4', source / 'acquisition_v2', source / 'acquisition_v1'):
            candidate = read(folder / (name + '.json'))
            if candidate and candidate.get('sha256') and candidate.get('local_archive'):
                path = Path(candidate['local_archive'])
                if path.exists() and path.stat().st_size == candidate['archive_bytes']:
                    local = candidate
                    break
        server = None
        for path in (job / 'server' / (name + '.json'),
                     source / 'acquisition_v4/server' / (name + '.json'),
                     source / 'reconciliation_v2' / (name + '_server.json')):
            candidate = read(path)
            if candidate and local and candidate.get('archive_sha256') == local['sha256']:
                server = candidate
                break
        phase = phases.get(name, {})
        row = {'subset': name, 'export': item['export'], 'local_archive_verified': local is not None,
               'archive_bytes': local['archive_bytes'] if local else phase.get('expected_bytes'),
               'sha256': local['sha256'] if local else None,
               'downloaded_bytes': phase.get('downloaded_bytes'),
               'server_verified': server is not None,
               'server_directory': server.get('server_directory') if server else None,
               'server_receipt_scope': 'historical extraction verification; compressed duplicates may be removed after D-drive backup verification',
               'worker_state': ('SERVER_VERIFIED_AFTER_CLIENT_ERROR' if server and phase.get('state') == 'LOCAL_VERIFIED_SERVER_NEEDS_ATTENTION' else phase.get('state', 'QUEUED'))}
        if item.get('kind') == 'model_zip':
            witness = read(source / 'model_document_witness/model_witness_document.json')
            row['independent_cpu_model_forward_passed'] = bool(witness and witness.get('status') == 'SMPLX_MODEL_CPU_FORWARD_PASSED')
            model = row
        else:
            rows.append(row)
    stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    summary = {'updated_at': stamp, 'download_worker_pid': pid, 'download_worker_running': running,
               'authorized_batch_subsets': len(rows), 'local_batch_subsets_verified': sum(x['local_archive_verified'] for x in rows),
               'server_batch_subsets_verified': sum(x['server_verified'] for x in rows),
               'ssm_previously_server_verified': True, 'totalcapture_skipped_by_user': True,
               'authorized_motion_archive_bytes': 57071473311, 'model': model, 'subsets': rows,
               'full_amass_downloaded': False, 'deletion_requires_verified_github_backup': True,
               'restricted_raw_assets_must_be_preserved': True, 'g1_pairs_generated': 0, 'new_262d_sequences': 0,
               'generator_training_started': False}
    lines = ['# AMASS 下载与人体模型安装进度', '', '更新时间：' + stamp, '',
             f"本批15个动作子集：本地已核验 {summary['local_batch_subsets_verified']}/15，服务器已核验 {summary['server_batch_subsets_verified']}/15。另有此前已完成的 SSM。", '',
             '动作压缩包总量为57.071 GB；SMPL-X模型压缩包为0.411 GB。TotalCapture依用户要求跳过。', '',
             '后台下载进程：' + ('运行中' if running else '已结束；请检查各项结果') + f'（PID {pid}）。', '',
             'SMPL-X模型：' + ('服务器文件已核验' if model and model['server_verified'] else '仍在获取或核验')
             + '；独立CPU模型前向：' + ('通过' if model and model['independent_cpu_model_forward_passed'] else '尚未确认') + '。', '',
             '| 子集 | D盘压缩包 | 服务器文件 | 下载状态 |', '|---|---|---|---|']
    for row in rows:
        local = '已核验' if row['local_archive_verified'] else '未完成'
        if row.get('downloaded_bytes') and not row['local_archive_verified']:
            local += f"（已接收{row['downloaded_bytes']/1e6:.1f} MB）"
        lines.append('| ' + row['subset'] + ' | ' + local + ' | ' + ('已核验' if row['server_verified'] else '未完成') + ' | ' + row['worker_state'] + ' |')
    lines += ['', '原始包位于本文件夹的 `archives/`。模型已解压到 `body_models_smplx_lockedhead_20230207/`。', '',
              '服务器接收下一包时保留5 GiB余量。后续删除必须先完成GitHub备份核验；AMASS和SMPL-X原始文件保留。此前清理的服务器压缩副本在D盘有完整校验备份。', '',
              '表中“已核验”指已完成归档CRC、哈希与服务器解压校验；服务器旧压缩副本可能已清理，完整包保留在D盘。这不代表所有动作已通过数值检查或完成重定向。', '',
              '完整AMASS未下载齐；本轮G1配对动作、262D序列均为0，生成器训练未开始。SMPL-X独立检查只确认模型加载与前向计算。', '',
              'AMASS及SMPL-X原始数据和模型权重不上传GitHub。仓库保存获取代码、清单、报告及验证证据。']
    return summary, '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--pid', type=int, required=True)
    p.add_argument('--watch', action='store_true')
    p.add_argument('--publish-on-completion', action='store_true')
    a = p.parse_args()
    a.base = a.base.resolve()
    queue = json.loads(a.queue.read_text())
    a.destination.mkdir(parents=True, exist_ok=True)
    while True:
        summary, report = generate(a.base, queue, a.job, a.pid)
        (a.destination / '下载与安装进度.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        (a.destination / '下载与安装进度.md').write_text(report, encoding='utf-8')
        if not a.watch or not summary['download_worker_running']:
            break
        time.sleep(60)
    if a.publish_on_completion and not summary['download_worker_running']:
        stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
        snapshot = a.base / 'results/amass_remaining_20261021' / ('completed_' + stamp)
        snapshot.mkdir(exist_ok=False)
        (snapshot / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        (snapshot / 'report.md').write_text(report, encoding='utf-8')
        repo = a.base.parent.parent
        def git(*args):
            return subprocess.run(['git', *args], cwd=repo, stdin=subprocess.DEVNULL,
                                  capture_output=True, text=True, timeout=120)
        branch = git('branch', '--show-current')
        staged = git('diff', '--cached', '--name-only')
        if branch.stdout.strip() == 'codex/phygile-reproduction-20261017' and staged.returncode == 0 and not staged.stdout.strip():
            paths = [str((snapshot / name).relative_to(repo)) for name in ('summary.json', 'report.md')]
            steps = [git('add', '--', *paths)]
            if steps[-1].returncode == 0:
                steps.append(git('commit', '-m', 'data: record authorized AMASS batch outcome'))
            if steps[-1].returncode == 0:
                steps.append(git('push', 'origin', 'codex/phygile-reproduction-20261017'))
            result = {'publish_attempted': True, 'exit_codes': [x.returncode for x in steps],
                      'last_stdout': steps[-1].stdout, 'last_stderr': steps[-1].stderr}
        else:
            result = {'publish_attempted': False, 'reason': 'Branch/index preflight did not pass; no Git mutation performed',
                      'branch_exit_code': branch.returncode, 'branch_stdout': branch.stdout, 'branch_stderr': branch.stderr,
                      'staged_exit_code': staged.returncode, 'staged_stdout': staged.stdout, 'staged_stderr': staged.stderr}
        (a.destination / '后台完成保存状态.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'running': summary['download_worker_running'],
                      'local_batch_subsets_verified': summary['local_batch_subsets_verified'],
                      'server_batch_subsets_verified': summary['server_batch_subsets_verified']}), flush=True)


if __name__ == '__main__':
    main()

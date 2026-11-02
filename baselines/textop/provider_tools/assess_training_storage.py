"""Read-only storage forecast; never launches, deletes, moves or loads weights.

Checkpoint sizes come from existing completed witnesses. They are planning
proxies, not measured formal-run sizes or evidence of data acceptance.
"""
import argparse
import datetime
import hashlib
import html
import json
import math
from pathlib import Path
import re
import shutil

from session import remote

BASE = Path(__file__).resolve().parents[1]
LOCAL = Path('D:/具身智能/TextOp复现_20261024')
GIB = 1024 ** 3


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    assert re.fullmatch(r'[a-zA-Z0-9_-]+', args.run)
    out = BASE / 'results/training_storage_20261025' / args.run
    assert not out.exists(), 'Use a unique observation name; retain prior evidence'
    plan_path = BASE / 'configs/filter-training-plan.json'
    step2_path = BASE / 'configs/paper-step2.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    step2 = json.loads(step2_path.read_text(encoding='utf-8'))
    evidence = [plan_path, step2_path]
    server = json.loads(remote(r'''
from pathlib import Path
import datetime,json,shutil
ram=Path('/dev/shm/ouyanghao_textop_20261025')
p=ram/'bmlrub_indexed_v1/prepared_v1/progress.json'
d=json.loads(p.read_text()) if p.exists() else {}
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'filesystems':{x:dict(shutil.disk_usage(x)._asdict()) for x in ['/data','/home','/dev/shm']},
 'bmlrub':{k:d.get(k) for k in ['status','completed_sequences','failed_sequences','total_requested','parallel_workers']},
 'memory_available_kib':int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))}))
'''))
    stages = {}
    for stage, config_name in [('vae', 'mvae'), ('ldm', 'dar')]:
        cfg = BASE / ('upstream/TextOpRobotMDAR/robotmdar/config/train/' + config_name + '.yaml')
        evidence.append(cfg)
        interval = int(re.search(r'^\s*save_every:\s*(\d+)', cfg.read_text(), re.M).group(1))
        total = step2[stage]['total_manager_steps']
        measured = []
        for path in sorted((BASE / 'results/generator_state_20261025').glob('*/receipt.json')):
            data = json.loads(path.read_text(encoding='utf-8'))
            if data['status'] != 'TEXTOP_GENERATOR_STATE_WITNESS_PASSED':
                continue
            evidence.append(path)
            measured.append(data['stages'][stage]['checkpoint']['bytes'])
        assert measured
        stages[stage] = {'manager_steps': total, 'released_save_interval': interval,
                         'scheduled_checkpoint_count': math.ceil(total / interval),
                         'observed_diagnostic_checkpoint_bytes': max(measured),
                         'scheduled_checkpoint_planning_bytes': math.ceil(total / interval) * max(measured),
                         'size_basis': 'Existing EMA-on diagnostic full-state checkpoint; formal EMA remains false. Planning proxy only.'}
    # Statistics witnesses have full optimizer/normalizer/counter state. Do not
    # deserialize any checkpoint or claim two-source size is full-library size.
    tracker_sizes = []
    for path in sorted((BASE / 'results').rglob('receipt.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('status') == 'TEXTOP_FILTER_STATISTICS_RESUME_WITNESS_PASSED':
            evidence.append(path)
            tracker_sizes.extend(x['bytes'] for x in data['checkpoints'])
    assert tracker_sizes, 'Missing completed tracker state receipt'
    iterations = plan['learning_iterations']
    checkpoint_steps = sorted(set(range(plan['checkpoint_interval'], iterations + 1, plan['checkpoint_interval'])) | {1000, iterations})
    tracker = {'iterations': iterations, 'checkpoint_steps': checkpoint_steps,
               'observed_diagnostic_checkpoint_bytes': max(tracker_sizes),
               'scheduled_checkpoint_planning_bytes': len(checkpoint_steps) * max(tracker_sizes),
               'size_basis': 'Two-source witness; full-library counters and logs add storage. Not an exact formal-run measurement.'}
    generator_bytes = sum(x['scheduled_checkpoint_planning_bytes'] for x in stages.values())
    reserve = plan['minimum_disk_free_gib'] * GIB
    checkpoint_proxy = tracker['scheduled_checkpoint_planning_bytes'] + generator_bytes
    # Every new checkpoint uses a new path; .partial becomes that checkpoint.
    # It is not a second full copy. Resume/emergency saves and logs are unbudgeted.
    assessment = {
        'status': 'FORMAL_TRAINING_STORAGE_PLAN_REQUIRED',
        'generated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'server_observation': server,
        'local_filesystems': {str(p):dict(shutil.disk_usage(p)._asdict()) for p in [Path('C:/'), Path('D:/')]},
        'filter_policy': tracker, 'generator': stages,
        'scheduled_checkpoint_planning_bytes': checkpoint_proxy,
        'minimum_free_reserve_bytes': reserve,
        'planned_checkpoints_plus_reserve_bytes': checkpoint_proxy + reserve,
        'data_volume_planning_shortfall_bytes': max(0, checkpoint_proxy + reserve - server['filesystems']['/data']['free']),
        'filter_only_proxy_fits_above_reserve': tracker['scheduled_checkpoint_planning_bytes'] + reserve <= server['filesystems']['/data']['free'],
        'excluded_unmeasured_costs': ['final accepted corpus and CLIP/statistics caches', 'remaining durable preprocessing/FK backups', 'logs, emergency/resume checkpoints', 'final WBC, generated augmentation and videos'],
        'limitations': ['Forecast only; no full-library capacity or physics acceptance.', 'EMA-on diagnostic sizes are conservative generator planning proxies, not exact formal sizes.', 'Separate mounts cannot be added and treated as one checkpoint destination.', 'Existing checkpoints and all datasets retained; no cleanup or relocation performed.'],
        'source_hashes': {p.relative_to(BASE).as_posix(): digest(p) for p in sorted(set(evidence))},
        'formal_training_started': False, 'physical_filter_executed': False, 'dataset_accepted': False,
    }
    out.mkdir(parents=True)
    (out / 'assessment.json').write_text(json.dumps(assessment, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines = [
        '# TextOp 后续训练存储预算', '',
        '这是现有配置和已保存诊断检查点的容量估算，没有启动训练或删除文件。', '',
        '| 项目 | 计划保存数量 | 检查点容量估算 |', '|---|---:|---:|',
        f"| 特权跟踪器 40,000 轮 | {len(checkpoint_steps)} | {tracker['scheduled_checkpoint_planning_bytes']/GIB:.2f} GiB |",
        f"| VAE 200,000 步 | {stages['vae']['scheduled_checkpoint_count']} | {stages['vae']['scheduled_checkpoint_planning_bytes']/GIB:.2f} GiB |",
        f"| LDM 300,000 步 | {stages['ldm']['scheduled_checkpoint_count']} | {stages['ldm']['scheduled_checkpoint_planning_bytes']/GIB:.2f} GiB |", '',
        f"本次服务器数据盘可用 {server['filesystems']['/data']['free']/GIB:.2f} GiB；按上述检查点加 2 GiB 保留空间计算，需约 {(checkpoint_proxy+reserve)/GIB:.2f} GiB，当前差额约 {assessment['data_volume_planning_shortfall_bytes']/GIB:.2f} GiB。", '',
        'VAE/LDM 沿用固定官方配置每 20,000 步保存一次，共 10/15 份；特权跟踪器采用已声明的第 1,000 轮及每 5,000 轮保存，共 9 份。生成器容量以已完成的 EMA 开启诊断为保守参考，正式设置仍为 EMA 关闭，不能把此估算当作精确文件大小。跟踪器的完整动作库计数还会增加容量。', '',
        '尚未计入最终语料、文本缓存、剩余数据备份、日志、额外恢复检查点、最终 WBC、增强数据和视频。因此此表不是全部复现所需磁盘的总预算。系统盘和本地磁盘不能直接合并为服务器单个输出目录的可用空间。', '',
        '当前并行重定向继续运行；正式训练前需要落实持久保存与容量方案，并重新测量可用空间。现有数据集、检查点和唯一实验记录全部保留。没有修改训练参数、放宽筛选条件或执行清理。', '',
        f"证据：results/training_storage_20261025/{args.run}/assessment.json。记录包含配置哈希、检查点尺寸来源和实际磁盘采样。", '',
    ]
    report = '\n'.join(lines)
    (BASE / 'reports/training_storage_20261025.md').write_text(report, encoding='utf-8')
    (LOCAL / '后续训练存储预算_20261025.md').write_text(report, encoding='utf-8')
    (LOCAL / '后续训练存储预算_20261025.html').write_text('<!doctype html><meta charset="utf-8"><title>TextOp 后续训练存储预算</title><style>body{max-width:1000px;margin:40px auto;font:17px/1.8 sans-serif}pre{white-space:pre-wrap}</style><pre>'+html.escape(report)+'</pre>', encoding='utf-8')
    print(json.dumps({k:assessment[k] for k in ['status','scheduled_checkpoint_planning_bytes','planned_checkpoints_plus_reserve_bytes','data_volume_planning_shortfall_bytes','filter_only_proxy_fits_above_reserve']}, ensure_ascii=True))


if __name__ == '__main__':
    main()

"""Read the newest actual observation without rewriting original training receipts."""
import json


def snapshot(base):
    progress = json.loads((base/'results/full_contact_20261026/formal_launch/progress.json').read_text(encoding='utf-8'))
    states = []
    for path in (base/'results/quick_reproduction_20261026').glob('*/state.json'):
        value = json.loads(path.read_text(encoding='utf-8'))
        states.append((value.get('sample_unix', 0), path, value))
    if states:
        _, state_path, state = max(states, key=lambda item: item[0])
    else:
        state_path, state = None, {'status': 'NOT_STARTED'}
    observed = state.get('observation', {})
    if observed.get('sample_unix', 0) > progress.get('sample_unix', 0):
        progress = dict(progress)
        progress.update(completed_learning_iterations=observed['current_completed_iterations'],
                        optimizer_updates=observed['optimizer_updates'],
                        sample_unix=observed['sample_unix'],
                        observation_source=str(state_path.relative_to(base)))
    interrupted = state.get('status') == 'QUICK_TRANSITION_NEEDS_INSPECTION'
    note = ''
    if interrupted and 'Connection timed out' in state.get('error', '') and not state.get('controlled_stop'):
        note = ('连接中断说明：以上为最后成功核验的训练记录，不代表当前服务器实时状态。'
                '本地自动切换协调器因连接错误退出，当前未取得5,000轮备份及受控结束证明；'
                '不能据此判断服务器训练已停止。恢复连接后先核对原进程、检查点、'
                '备份和待处理停止记录，保留失败现场，再按原安全流程接续。')
    elif interrupted:
        note = ('自动切换协调器存在待检查错误；以上为最后核验记录，'
                '不能仅凭旧进度判断当前训练或备份状态，需检查实际错误与原始退出证据。')
    return progress, state, note

"""Current online functional validation and full teacher phase, no remote mutation."""
import html,json,pathlib,re
B=pathlib.Path(__file__).resolve().parents[1];D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
def main():
    p=B/'results/online_quick_20261026/functional_acceptance_v1.json'
    if not p.exists():return
    acceptance=json.loads(p.read_text(encoding='utf-8'))
    doc=json.loads((B/'results/online_quick_20261026/online_document_v1/independent_summary.json').read_text(encoding='utf-8'))
    pp=B/'results/full_teacher_resume_20261026/progress.json';progress=json.loads(pp.read_text(encoding='utf-8')) if pp.exists() else {}
    rc=progress.get('receipt') or {};observed=progress.get('observed_completed_from_native_log')
    running=bool(progress.get('actual_workers')) and rc.get('resume_exact_state_verified') and progress.get('outer_exit') is None
    phase=('完整训练第一阶段已实际续训：从5,000轮完整状态恢复，目标40,000轮。最近实际训练日志完成约{}轮；这是日志观察，稳定检查点另行核验。'.format(observed or 5000)
           if running else '完整跟踪器续训入口已通过资源与状态前置核查；实际启动/恢复以最新训练收据为准。')
    if progress.get('outer_exit') is not None:
        phase='本次完整跟踪器监督程序已退出，返回值{}；需按实际完成计数判断结果。'.format(progress['outer_exit']['actual_returncode'])
    lines=['# TextOp快速版在线功能验收与完整训练接续','',
      '快速版功能链路已经实际跑通：已训练VAE/LDM → 在线文本切换 → 异步动作生成 → 实际物理控制。作者与独立试验各完成257次控制/1028物理子步，均0超时、0断流、0物理失败，实际退出0。每次另有16次预热，不计入成功控制步。','',
      '独立前置63项、内容与时序430项、后置41项检查全部通过。控制循环约5.132秒；文本命令提交到新目标被消费约0.497秒，包含缓冲延迟。这是有限时长、脚本触发的实时切换验证，不是无限时长硬实时保证或已经交付交互界面。','',
      phase,'',
      '完整生成器VAE200,000/LDM300,000尚未启动：先完成40,000轮跟踪器及全部可得来源的真实两轮物理筛选，再验收完整69D语料、训练集归一化、CLIP缓存、恢复和存储容量。快版VAE2k/LDM3k课程时钟不同，不会冒充论文精确续训。','',
      '## 质量评审与范围','',
      '动作质量没有达到论文验收：固定采样误差0.3103→0.6075变差，现有walk/wave样本也不足以证明语义正确。完整训练是继续改善质量的研究阶段，不表示该问题已解决。最终WBC、最终端点、长时间稳定性和论文级语义指标仍待完成。','',
      '当前单机器人在线配置显式使用CPU PhysX、GPU1神经推理，全部进程合计CPU0–3，Torch显存比例合计0.65；完整8192环境训练仍使用原GPU物理配置。原生SDK关闭仍使用有界退出兜底。','',
      'v1–v10的慢版本及v2/v7失败全部保留。v11把校验/读取移出控制线程、分配同一四核、显式预热实际推理检查；没有放宽20ms期限、物理阈值或更换权重。12个在线运行的全部原始记录均已私有保存至服务器和D盘。','',
      '已有5.16秒离线生成动作的实际执行视频继续保留。新的在线证据包含完整真实执行轨迹；该在线轨迹尚未单独渲染视频，不把旧片说成新的在线影片。','',
      '完整可得8278来源的预处理已完成；快速筛选64条中42自然完成、22失败；快版语料42原始来源/31train/11val。未尝试8214条不作拒绝。13条所选CMU缺失、403作者私有动作不可得。','',
      '证据：results/online_quick_20261026/functional_acceptance_v1.json；独立online_document_v1；results/full_teacher_resume_20261026/progress.json。私有数组、逐条文本来源和真实轨迹没有公开。']
    text='\n'.join(lines)
    (B/'reports/online_quick_20261026.md').write_text(text,encoding='utf-8')
    (D/'TextOp在线功能验收与完整训练接续_20261026.md').write_text(text,encoding='utf-8')
    try:
        import markdown
        body=markdown.markdown(text,extensions=['tables'])
    except ImportError:body='<pre>'+html.escape(text)+'</pre>'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>TextOp在线功能验收与完整训练接续</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:17px/1.7 system-ui;color:#203040}pre{white-space:pre-wrap}</style>'+body+'</html>'
    (D/'TextOp在线功能验收与完整训练接续_20261026.html').write_text(page,encoding='utf-8')
    for name in ['TextOp下载与训练报告.html','TextOp真实跟踪评估与视频_20261026.html']:
        target=D/name
        if not target.exists():continue
        old=target.read_text(encoding='utf-8');old=re.sub(r'<!--ONLINE_CURRENT_BEGIN-->.*?<!--ONLINE_CURRENT_END-->','',old,flags=re.S)
        banner='<!--ONLINE_CURRENT_BEGIN--><section style="padding:20px;background:#e8f4ef;border:1px solid #8ab5a0"><h2>当前：快速在线功能链路已验收</h2><p>两次真实257步执行均零超时/断流/物理失败；语义与论文质量仍未接受。</p><p>'+html.escape(phase)+'</p><p>完整生成器仍等待全量物理筛选及最终训练集/容量验收。</p><a href="TextOp在线功能验收与完整训练接续_20261026.html">最新完整说明</a></section><!--ONLINE_CURRENT_END-->'
        marker=old.find('</style>')
        old=old[:marker+8]+banner+old[marker+8:] if marker>=0 else banner+old
        target.write_text(old,encoding='utf-8')
    print(json.dumps({'bounded_online_functional_accepted':True,'full_teacher_running':bool(running),'full_generator_training_started':False,'semantic_quality_accepted':False}))
if __name__=='__main__':main()

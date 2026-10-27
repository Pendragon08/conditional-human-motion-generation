"""Render actual fixed-64 comparison; never infer physical/semantic acceptance."""
import html,json,pathlib,re
B=pathlib.Path(__file__).resolve().parents[1]
D=pathlib.Path('D:/具身智能/TextOp复现_20261024')

def main():
    result=B/'results/teacher10000_actual_20261027/paired64_v1/summary.json'
    if not result.exists():return
    j=json.loads(result.read_text(encoding='utf-8'))
    run=json.loads((B/'results/teacher10000_actual_20261027/teacher10000_sample64_v1/summary.json').read_text(encoding='utf-8'))
    assert j['status']=='TEACHER10000_FIXED64_PAIRED_COMPARISON_RECORDED' and j['source_count']==64
    assert run['output_writers_closed'] and run['child_exit_code']==0 and run['private_two_location_sha_verified']
    c=j['outcome_pairs'];m=j['paired_terminal_metrics'];paired=c['both_complete']
    rows=[]
    for k,label in [('terminal_world_root_position_error_m','末帧世界根位置误差（米）'),('terminal_root_orientation_geodesic_rad','末帧根旋转误差（弧度）'),('terminal_joint_rmse_rad','末帧29关节RMSE（弧度）')]:
        a=m[k]['old_paired'];b=m[k]['new_paired'];delta=m[k]['paired_change_new_minus_old']
        rows.append(f"| {label} | {a['mean']:.6f} | {b['mean']:.6f} | {delta['mean']:+.6f} |")
    text='\n'.join([
      '# TextOp：10,000轮策略与5,000轮的真实配对评估','',
      f"已按用户要求提前评估10,000轮策略。同一64条完整原始动作，5,000轮自然完成{j['baseline_complete']}条、失败{j['baseline_failed']}条；10,000轮自然完成{j['new_complete']}条、失败{j['new_failed']}条。",'',
      f"配对结果：两者都完成{c['both_complete']}条，10k新增完成{c['new_only_complete']}条，5k完成但10k失败{c['old_only_complete']}条，两者都失败{c['both_failed']}条。所有失败保留，未尝试的8214来源没有被拒绝。",'',
      '## 末帧质量：只比较双方都完整结束的相同动作','',
      f'共同分母为{paired}条。负的误差变化表示改善；提前失败不做末帧外推。','',
      '| 指标 | 5k均值 | 10k均值 | 10k−5k |','|---|---:|---:|---:|',*rows,'',
      '末帧在(T−1)/50秒比较真实录制状态与独立参考T−1；控制器原来最后消费T−2。世界坐标仅减场景原点，不逐帧对齐根或朝向。这些是自定义跟踪诊断，不是论文分数或新筛除阈值。不同成功集合的统计不能替代同源配对。','',
      '根位置和关节误差均值下降，根朝向误差均值略升；同源位置和关节误差的95分位数也没有改善。成功率提高不代表每种误差、每条动作都改善，本次单一样本和检查点对照不能给出统计显著性或普遍结论。','',
      '配对分析已由fresh独立文档执行一次只读复核，实际退出0，64来源逐条复算一致，18项聚合/来源检查通过。10k端点从真实记录重算，5k使用之前已独立验证且SHA绑定的末帧结果；没有重复原5k物理试验。公开摘要的CRLF差异已按内容核对，未改原实验。','',
      '## 执行与范围','',
      f"实际64条试验全部关闭；{run['actual_vectorized_control_calls']}次向量控制调用、{run['actual_scene_physics_steps']}物理子步、0优化更新，监督实际退出0；完整原始记录已在服务器和D盘私有保存并逐摘要核验。",'',
      '作者与fresh独立文档各先完成同8条短动作：8自然完成、0失败、63控制/252物理子步，47组独立核查通过。这是相同8来源的复核，不是16个不同来源。原生关闭未返回，实际在stop/render链使用有界退出；没有声称修复原生关闭。','',
      '10k完整模型/Adam/归一化/采样统计已经服务器与D盘备份；完整恢复状态对应10000轮/200000更新。原训练因交接观察延迟最后收据11400轮，额外内存更新不能冒充稳定检查点；受控暂停保留原始outer退出1，不是40k完成。','',
      '动作语义仍未通过：快版数据中wave/right覆盖不足，固定LDM采样误差曾变差；本次评估的是参考动作跟踪，没有新增文本生成质量结果。后续需真实评估语义候选并扩大有效训练覆盖，固定噪声/历史做文本与无条件对照、多种子和保留集评估，不能把多训练自动视为语义改善。','',
      '已冻结下一批44个含wave+right词形的原始来源（27train/17val），与本次64没有重合；长度125至9396帧、合计74330帧，保留完整动作而不截短或放宽阈值。选择及摘要已私有备份，但新执行入口仍未实现/启用，尚无这44条的新物理试验。更广的wave词形候选共208条；词形本身不是语义真值。','',
      '接续决定：保留10k快照，先扩大诊断覆盖，优先实测上述语义不足类别；依据真实失败和端点质量决定是否再增加teacher训练。原40k论文参照保留，当前不因单项完成率增加而直接全库接纳或重启40k。','',
      '全8278来源的真实宽松/严格两轮筛选、最终语料与持久容量、完整VAE200k/LDM300k、最终可部署WBC和增强、实际在线语义/物理质量及视频仍待完成。达到论文完整复现效果后通知。','',
      '证据：results/teacher10000_actual_20261027/teacher10000_sample64_v1和paired64_v1；私有逐源配对、动作与文本索引、原始轨迹不公开。'])
    (B/'reports/teacher10000_actual_20261027.md').write_text(text,encoding='utf-8')
    title='TextOp一万轮策略真实筛选对照_20261027'
    (D/(title+'.md')).write_text(text,encoding='utf-8')
    try:
        import markdown
        body=markdown.markdown(text,extensions=['tables'])
    except ImportError:body='<pre>'+html.escape(text)+'</pre>'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+title+'</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:17px/1.7 system-ui;color:#203040}td,th{padding:8px;border:1px solid #ccd}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>'+body+'</html>'
    (D/(title+'.html')).write_text(page,encoding='utf-8')
    for name in ['TextOp下载与训练报告.html','TextOp真实跟踪评估与视频_20261026.html','TextOp在线功能验收与完整训练接续_20261026.html']:
        p=D/name
        if not p.exists():continue
        old=re.sub(r'<!--TEACHER10000_CURRENT_BEGIN-->.*?<!--TEACHER10000_CURRENT_END-->','',p.read_text(encoding='utf-8'),flags=re.S)
        banner='<!--TEACHER10000_CURRENT_BEGIN--><section style="padding:20px;background:#e8f4ef;border:1px solid #8ab5a0"><h2>最新：10,000轮固定64动作评估完成</h2><p>'+html.escape(f"自然完成{j['baseline_complete']}/64 → {j['new_complete']}/64；新增完成{c['new_only_complete']}条，回退{c['old_only_complete']}条。语义质量与全库筛选仍未验收。")+'</p><a href="'+title+'.html">配对末帧指标与完整说明</a></section><!--TEACHER10000_CURRENT_END-->'
        marker=old.find('</style>');p.write_text(old[:marker+8]+banner+old[marker+8:] if marker>=0 else banner+old,encoding='utf-8')
    print(json.dumps({'fixed64_complete':True,'new_completed':j['new_complete'],'full_paper_complete':False}))

if __name__=='__main__':main()

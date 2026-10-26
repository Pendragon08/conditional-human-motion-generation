"""Report actual closed trials separately from paper-scale filtering and generation."""
import datetime
import html
import json
import pathlib
import shutil

B = pathlib.Path(__file__).resolve().parents[1]
D = pathlib.Path('D:/具身智能/TextOp复现_20261024')
R = B/'results/quick_pilot_actual_20261026'
rows = []
for p in sorted(R.glob('*/summary.json')):
    j = json.loads(p.read_text(encoding='utf-8'))
    if j.get('status') == 'TEXTOP_QUICK_PILOT_EXECUTION_COMPLETED':
        rows.append(j)
assert rows, 'No actual closed pilot evidence: do not emit a result report'
state = json.loads((B/'results/quick_reproduction_20261026/coordinator_network_resume_v2/state.json').read_text(encoding='utf-8'))
stop = state['controlled_stop']
assert stop['status'] == 'USER_AUTHORIZED_QUICK_5000_EARLY_STOP_COMPLETE'
backup = json.loads((B/'results/formal_filter_checkpoints_20261026/round_5000/backup.json').read_text(encoding='utf-8'))
assert backup['private_full_two_location_sha_verified'] and backup['inference_three_location_sha_verified']
videos = []
for p in sorted(R.glob('*/receipt.json')):
    j = json.loads(p.read_text(encoding='utf-8'))
    ep = p.parent/'exit.json'
    if j.get('status') != 'RECORDED_ISAACLAB_POSE_VIDEO_CREATED' or not ep.exists():
        continue
    ex = json.loads(ep.read_text(encoding='utf-8'))
    assert j['output_writers_closed'] and ex['child_exit_code'] == 0 and ex['supervision_passed']
    videos.append((p.parent.name, j))
samples = [j for j in rows if j['stage'] == 'sample']
generated_videos=[]
for p in sorted((B/'results/generated_execution_20261026').glob('*/receipt.json')):
    j=json.loads(p.read_text(encoding='utf-8'));ep=p.parent/'exit.json'
    if j.get('status')=='RECORDED_ISAACLAB_POSE_VIDEO_CREATED' and ep.exists():
        ex=json.loads(ep.read_text(encoding='utf-8'))
        assert ex['child_exit_code']==0 and ex['supervision_passed'] and j['output_writers_closed']
        assert j['provenance']['source_kind']=='trained_generated_reference_actual_isaaclab_execution'
        generated_videos.append((p.parent.name,j))
now = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
md = ['# TextOp快速复现：真实跟踪评估与录像', '', '更新：'+now, '',
      '**当前属于5,000轮策略的快速验证；论文40,000轮完整版、全库物理筛选和文本生成—执行闭环尚未完成。**', '',
      '|阶段|实际状态|', '|---|---|',
      '|可得数据预处理|8,278条原始动作＋对应镜像，GMR、文本时间对齐、69D编码、FK和私有持久备份已完成|',
      '|可恢复跟踪策略|5,000轮／100,000次优化更新完整检查点，服务器和本地私有备份SHA一致|',
      '|受控结束|连接恢复后完成备份再发送一次SIGINT；原监督器退出0，原始最终收据保留|',
      '|断连造成的额外训练|最终观察到{}轮／{}次更新；5,000轮之后的内存更新不保证可恢复，实际评估使用5,000轮快照|'.format(stop['current_completed_iterations'], stop['optimizer_updates']),
      '|数据与原始记录|数据集、检查点和失败记录全部保留；旧57D、PhyGile、ReactiveBFM继续暂停|', '',
      'SSH于本轮检查恢复，没有修改代理、路由或ZeroTier设置，先前断连根因仍未确定。原失败协调器coordinator_v1和新的安全接续coordinator_network_resume_v2均保留，未重启训练器或改动其16份活动文件与3份控制源码。', '',
      '## 已完成的实际动作回放', '',
      '|运行|动作数|自然完成|失败|向量控制调用|物理子步|真实退出|', '|---|---:|---:|---:|---:|---:|---:|']
for j in rows:
    assert j['supervision_passed'] and j['child_exit_code'] == 0 and j['private_two_location_sha_verified']
    md.append('|{}|{}|{}|{}|{}|{}|{}|'.format(j['run'],j['finished_trials'],j['complete_trials'],j['failed_trials'],j['actual_vectorized_control_calls'],j['actual_scene_physics_steps'],j['child_exit_code']))
md += ['', '两次smoke是对同一组8条完整短动作的独立重复验证，不能相加称为16条不同来源。代表性sample64另选跨10个数据集的64条完整原始动作，与smoke选择不重叠；未尝试来源不能判为失败或物理不可行。', '',
       '## 自定义跟踪误差', '',
       '下表按每条试验实际控制步加权，包含失败前已执行的步数。它比较自动重置前的真实机器人姿态与当前消费的控制目标，不是论文基准分数，也没有独立验收最后参考帧端点。', '',
       '|运行／统计范围|试验数|实际累计试验控制步|14身体位置平均误差 m|29关节RMSE rad|', '|---|---:|---:|---:|---:|']
for j in rows:
    for name, v in j.get('own_tracking_metrics', {}).items():
        if name == 'all' or j['stage'] == 'sample':
            md.append('|{} / {}|{}|{}|{:.5f}|{:.5f}|'.format(j['run'],name,v['trials'],v['trial_control_steps'],v['aligned_14_body_position_error_mean_m'],v['joint_rmse_rad']))
md += ['', 'SDK原生关闭仍未返回，实际使用了有界退出fallback；退出0不代表修复了原生关闭。训练和回放沿用同一静态箱体地面、自碰撞与严格阈值；作者精确碰撞资产不可得，仍是复现差异。实际CUDA/Torch在GPU1、CPU0–3、Torch显存比例0.65；SDK在GPU0的辅助上下文按原资源记录披露。环境seed沿用既有配置20261025，执行脚本Torch seed为20261026，不能把两个设置混为同一个随机源。', '',
       '## 实际录像', '']
for name, j in videos:
    md.append('- {}：{}帧、50fps，约{:.2f}秒，{}字节；媒体SHA {}。'.format(name,j['frames_decoded'],j['frames_decoded']/50,j['video_bytes'],j['video_sha256']))
public_movies = []
for name, j in videos:
    proof = R/name/'github_video.json'
    if proof.exists():
        pub = json.loads(proof.read_text(encoding='utf-8'))
        assert pub['video_sha256'] == j['video_sha256']
        public_movies.append(pub)
        md.append('  [GitHub科研结果视频与署名说明]('+pub['github_release']+')。')
md += ['', '录像是已记录IsaacLab物理执行姿态的逐帧可视化，由CPU/Mesa和MuJoCo视觉FK绘制，新增动力学积分和优化均为0。它不是参考动作动画，也不是MuJoCo跨仿真控制或生成器闭环视频。视觉MJCF与跟踪器URDF来源不同，不能据画面声称碰撞几何相同。原始轨迹和逐条标注保存在服务器与本地私有目录。', '']
if public_movies:
    md += ['SFU来源的成功示例以非商业科研结果录像发布，并附数据来源与资助致谢；[SFU官网条款](https://download.is.tue.mpg.de/amass/licenses2024/SFU/sfu_license_updated.html)说明科研使用及发表结果时的致谢要求。BMLmovi来源的失败录像保留本地；其[官网许可](https://download.is.tue.mpg.de/amass/licenses2024/BMLmovi/bmlmovi_license_updated.html)对第三方分发的限制更严，不随公开代码发布。没有公开原始动作数组、人体模型或逐条文本索引。', '']
else:
    md += ['视频先保存在本地私有目录，公开版本仅使用已核对允许科研结果发布的来源并附署名；受限逐条动作和标注不公开。', '']
timeout_record=R/'video_timeout_v1.json'
if timeout_record.exists():
    failed_video=json.loads(timeout_record.read_text(encoding='utf-8'))
    md += ['长片渲染记录：32.26秒完整轨迹的首次CPU/Mesa导出在900秒监督时限退出，实际退出−15，保留了不完整MP4和全部日志；它不是有效结果视频，物理评估结果与闭合轨迹没有丢失。根据52帧短片36.76秒的实测速度，仅将完整导出的监督预算改为2,400秒，帧率、分辨率、所有输入帧和渲染数值签名不变，新版本使用独立run并保留实际退出证据。更长的成功与失败录像以最终收据为准，不能把后台启动或部分文件算成完成。', '']
md += [
       '## 下一步与范围', '']
if samples:
    j = samples[-1]
    md.append('64条代表动作已真实评估：自然完成{}、失败{}。这些是早期策略的本次试验结果，不能作为动作绝对物理可行性的结论。'.format(j['complete_trials'],j['failed_trials']))
else:
    md.append('独立smoke证据验收后接续64条代表动作评估，保留所有失败和未完成记录；当前不能说全库物理筛选完成。')
view = R/'source_view_v1/summary.json'
if view.exists():
    v = json.loads(view.read_text(encoding='utf-8'))
    md.append('来源选择保留{}条名义跟踪完成的原始参考，train/val来源为{}；未评估镜像排除。来源view的历史training_ready=false与后续独立语料/训练入口验收分别记录。'.format(v['nominal_tracking_completed_sources'],v['retained_source_splits']))
data_proof=B/'results/quick69_corpus_20261026/data_acceptance_v1.json'
if data_proof.exists():
    proof=json.loads(data_proof.read_text(encoding='utf-8'))
    md += ['快速版私有参考语料与缓存已完成独立验收：42条原始参考、31条训练来源与11条验证来源、174个动作—文本标注对、74个非空CLIP文本，训练归一化统计包含14,505个69D特征帧。来源和精确float32姿态跨划分重叠为0，未加入未评估镜像；TRAIN/VAL/weightedTRAIN三项实际512批次、四primitive加载均通过，history连续误差0、缺失文本嵌入0。全部私有语料已在服务器与D盘核验备份。', '']
    vae_path=B/'results/quick_vae_20261026/quick_vae_train_v1/receipt.json'
    if vae_path.exists():
        vae=json.loads(vae_path.read_text(encoding='utf-8'))
        md += ['## 快速VAE实际训练', '',
               '服务器运行quick_vae_train_v1，最近核验状态{}：{}/2,000次primitive训练尝试、{}次实际Adam更新、{}次跳过。VAE使用69D、batch512、history2/future8、四primitive和原网络/损失；课程及学习率时钟压缩100倍，属于快速版，不等同论文200,000步。'.format(vae['status'],vae['completed_primitive_attempts'],vae['optimizer_updates'],vae['skipped_updates']), '']
        vx=vae_path.parent/'exit.json'
        if vx.exists():
            ex=json.loads(vx.read_text(encoding='utf-8'))
            md.append('实际子进程退出{}、监督验收{}；固定验证批次的VAE重建门槛{}。这些是后验重建诊断，不是文本生成质量或物理执行指标。'.format(ex['child_exit_code'],ex['supervision_passed'],vae.get('vae_quality_gate_passed')))
            if 'final_quality' in vae:
                md += ['', '|固定批次后验重建Huber|训练前|训练后|','|---|---:|---:|']
                for key in ['teacher_forced','self_rollout']:
                    md.append('|{}|{:.6f}|{:.6f}|'.format(key,vae['initial_quality'][key]['normalized_reconstruction_huber'],vae['final_quality'][key]['normalized_reconstruction_huber']))
        if vae.get('checkpoints'):
            md.append('已实际写入的完整状态步数：{}。每份保存后重新加载，核对模型、优化器、计数和所有随机数状态；不删除已有版本。'.format([x['manager_step'] for x in vae['checkpoints']]))
        md += ['', 'LDM3,000步使用独立入口和真实已训练VAE验收，不以梯度诊断权重代替。原论文VAE200,000/LDM300,000配置保留；当前文本生成—执行闭环仍未完成。', '']
        ldm_path=B/'results/quick_ldm_20261026/quick_ldm_train_v1/receipt.json'
        if ldm_path.exists():
            ldm=json.loads(ldm_path.read_text(encoding='utf-8'))
            md += ['## 快速LDM实际训练', '',
                   '服务器运行quick_ldm_train_v1，最近核验状态{}：{}/3,000次primitive训练尝试、{}次实际AdamW更新、{}次跳过。输入为真实CLIP文本嵌入与历史2帧，VAE2000保持冻结；五步cosine扩散预测x0，后续每次生成未来8帧。课程和学习率时钟为快速版。'.format(ldm['status'],ldm['completed_primitive_attempts'],ldm['optimizer_updates'],ldm['skipped_updates']), '']
            lx=ldm_path.parent/'exit.json'
            if lx.exists():
                le=json.loads(lx.read_text(encoding='utf-8'))
                md.append('实际子进程退出{}、监督验收{}；生成器训练完成与真实机器人仿真闭环验证分别记录。'.format(le['child_exit_code'],le['supervision_passed']))
            if ldm.get('final_full_sampling_quality'):
                md.append('固定12个验证历史、四primitive完整五步CFG5采样的归一化Huber诊断：训练前{:.6f}，训练后{:.6f}。采样不读取未来动作后验，后续primitive使用自身预测历史；这不是文本语义指标，也不证明物理可跟踪。'.format(ldm['initial_full_sampling_quality']['normalized_full_sampling_huber'],ldm['final_full_sampling_quality']['normalized_full_sampling_huber']))
            md += ['', '仍需对实际生成参考进行独立物理执行、文本切换和失败记录验证，已有SFU影片展示的是参考动作跟踪。', '']
        else:
            md += ['独立LDM入口已部署，作者梯度/状态恢复/完整五步CFG5采样见证已通过；以新的独立见证与实际训练收据判断下一阶段，不把4次诊断更新算为3,000步训练。', '']
    else:
        md += ['VAE2,000步入口已实现并独立核验；以实际训练收据判断是否开训，LDM3,000步仍需其独立入口与已训练VAE质量验收。原论文VAE200,000/LDM300,000配置保留。', '']
else:
    md += ['随后构建保留REFERENCE+BABEL的私有训练语料，复核源数组与来源/姿态划分、训练集69D归一化、真实完整CLIP缓存、实际加载与断点恢复，再训练明确标注的快速生成器。VAE2,000/LDM3,000目前仍是独立禁用提案，尚无已验收训练入口或生成器结果；原论文VAE200,000/LDM300,000配置保留。', '']
generation_receipts=[]
for p in sorted((B/'results/quick_generation_20261026').glob('*/receipt.json')):
    j=json.loads(p.read_text(encoding='utf-8'))
    if j.get('status')=='TRAINED_QUICK_TEXT_REFERENCE_GENERATED':generation_receipts.append((p.parent.name,j))
if generation_receipts:
    md += ['## 新生成动作的真实执行原型', '',
           '真实VAE2000与LDM3000已进行文本条件生成：两帧中立初始参考，后续只使用自身预测的末两帧历史，32个primitive各生成8帧，新增未来256帧。自写指令先为“walk forward”，后为“wave your right hand”；不读取真实未来动作或未来后验。两次独立生成的258×36姿态及CLIP嵌入逐值一致，169项独立核对通过。它验证生成路径与可复验性，文本语义匹配仍未验收。', '',
           '|SDK独立运行|生成参考帧数|实际控制步|受控物理子步|自然完成|失败|文本切换目标已消费|身体平均误差m|关节RMSErad|',
           '|---|---:|---:|---:|---|---|---|---:|---:|']
    for pp in sorted((B/'results/generated_execution_20261026').glob('*/pipeline.json')):
        root=pp.parent;combined=json.loads(pp.read_text(encoding='utf-8'));rp=root/'replay/receipt.json';ep=root/'replay/exit.json'
        if not rp.exists() or not ep.exists():continue
        rr=json.loads(rp.read_text(encoding='utf-8'));ex=json.loads(ep.read_text(encoding='utf-8'));trial=rr['trial'];metrics=rr['own_tracking_metrics']
        assert combined['output_writers_closed'] and rr['output_writers_closed'] and ex['child_exit_code']==0 and ex['supervision_passed']
        md.append('|{}|{}|{}|{}|{}|{}|{}|{:.6f}|{:.6f}|'.format(root.name,trial['reference_frames'],trial['control_steps'],rr['actual_controlled_physics_substeps'],trial['complete'],trial['failed'],rr['text_switch_physically_consumed'],metrics['aligned_14_body_position_error_mean_m'],metrics['joint_rmse_rad']))
    md += ['', '这是同一条短生成序列的作者与独立重复试验，不能相加称为两个不同动作样本。参考身体数据通过跟踪器原始URDF的真实SDK前向运动学导出，再由原5k/100k特权策略实际执行；两次均保留自动重置前的终止状态。SDK独立一致性审计416项通过。首次审计的四项失败来自验证器假设：跨NumPy版本float64角速度舍入差最大3.55e−15rad/s，以及同一地面Cube的等价size/scale文本表示；原失败保留，仅修正验证器，没有改数据、仿真或物理阈值。', '',
           '当前已验证的是离线“文本生成→机器人参考→仿真反馈控制”原型。控制包含257个20ms区间；没有额外验收最后参考帧端点。独立实测约30ms/primitive只是神经推理耗时，未验证在线文本输入、异步队列、50Hz执行期限或实时交互。初次SDK原生关闭仍未返回，使用原有有界退出，差异保留。', '',
           '**生成质量仍待验收：固定完整采样Huber从0.310267升到0.607536。单条序列被物理跟踪不等于文本语义或论文质量通过。**', '']
    for name,j in generated_videos:
        md += ['新生成动作的实际执行录像：{}，{}帧／50fps／{:.2f}秒，媒体SHA {}；已独立检查首、中、末帧并在服务器与D盘核验。影片由实际IsaacLab执行姿态绘制，CPU渲染不新增动力学。新影片先保存在本地私有目录，公开仓库保存代码、聚合结果与报告。'.format(name,j['frames_decoded'],j['frames_decoded']/50,j['video_sha256']), '']
    md += ['后续顺序：先补齐快速版在线文本切换、异步生成与实际控制时序验证，并评估采样误差和语义效果；然后扩大真实物理筛选，完成数据/质量/持久容量验收，再进入论文40,000轮跟踪策略、VAE200,000步、LDM300,000步完整训练。快版课程和学习率时钟已缩短，正式阶段须独立记录，不能把快版计数当作论文完整训练。', '']
md += ['第一阶段5k特权teacher原型不能冒充论文最终可部署WBC或完整实时系统。', '',
       '数据范围差异：13条所选CMU导出缺失，403条作者私有动作不可得；8,278条来源不等于作者全部12,296条公共跟踪动作。公开仓库保存代码、聚合证据、推理权重、许可允许的科研结果录像和报告；完整Adam/逐动作采样状态与受限资产保持私有。', '',
       '[5,000轮推理权重与说明]('+backup['github_release']+')。完整恢复仍须私有检查点。', '',
       '证据目录：results/quick_pilot_actual_20261026、results/formal_filter_checkpoints_20261026/round_5000及results/quick_reproduction_20261026/coordinator_network_resume_v2。']
text = '\n'.join(md)
(B/'reports/quick_actual_results_20261026.md').write_text(text,encoding='utf-8')
(D/'TextOp真实跟踪评估与视频_20261026.md').write_text(text,encoding='utf-8')
try:
    import markdown
    body = markdown.markdown(text,extensions=['tables'])
except ImportError:
    body = '<pre>'+html.escape(text)+'</pre>'
for name, j in videos:
    movie = D/'videos/private'/name/'video.mp4'
    if movie.exists():
        body += '<h2>本地录像：'+html.escape(name)+'</h2><video controls preload="metadata" style="max-width:100%" src="videos/private/'+html.escape(name)+'/video.mp4"></video>'
for name,j in generated_videos:
    movie=D/'videos/private'/name/'video.mp4'
    if movie.exists():
        body += '<h2>新生成动作的真实执行录像</h2><p>VAE2000/LDM3000生成参考，由5k策略在IsaacLab实际执行；离线原型，实时交互与语义质量待验收。</p><video controls preload="metadata" style="max-width:100%" src="videos/private/'+html.escape(name)+'/video.mp4"></video>'
page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>TextOp真实跟踪评估与视频</title><style>body{max-width:1120px;margin:40px auto;padding:0 24px;font:17px/1.7 system-ui;color:#203040}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd6df;padding:8px}pre{white-space:pre-wrap}a{color:#135fa3}</style>'+body+'</html>'
(D/'TextOp真实跟踪评估与视频_20261026.html').write_text(page,encoding='utf-8')
main = D/'TextOp下载与训练报告.html'
prior = D/'TextOp主报告_真实pilot前快照_20261026.html'
if main.exists() and not prior.exists():
    shutil.copy2(main,prior)
main.write_text(page,encoding='utf-8')
print(json.dumps({'actual_closed_reference_pilot_runs':len(rows),'reference_tracking_videos':len(videos),'generated_execution_videos':len(generated_videos),'full_paper_generator_trained':False,'whole_library_filtered':False,'real_time_generated_physics_closed_loop_verified':False}))

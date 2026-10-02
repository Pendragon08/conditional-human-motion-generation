"""Complete local media, logger, artifact documentation; preserve baseline-only scope."""
from pathlib import Path
import json,html,hashlib,shutil,csv,re,datetime
from urllib.parse import quote,unquote,urlparse
R=Path(r'D:\具身智能\conditional-human-motion-generation')
OUT=Path(r'D:\具身智能\通用条件人体运动生成_已完成实验结果_20260929')
PACK=OUT/'论文复现汇报资料_20261002';HUMAN=OUT/'人体生成视频';TRAIN=OUT/'PPO训练曲线';ROBOT=OUT/'机器人执行视频'
REPORT=OUT/'实验结果汇报.html'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def esc(t):return html.escape(str(t))
def href(path,label):return f'<a href="{quote(str(path).replace(chr(92),"/"),safe="/:#")}">{esc(label)}</a>'
def table(headers,rows):return '<div class="tablewrap"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def section(id,title,body):return f'<section id="{id}"><div class="eyebrow">流程与可复查产物</div><h2>{title}</h2>{body}</section>'
def standalone(title,body,script=''):
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+esc(title)+'</title><style>body{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1200px;margin:30px auto;padding:0 20px;color:#18344b;line-height:1.8;background:#f2f6f9}h1{font-size:30px}a{color:#126678}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px;border-bottom:1px solid #d3dfe8;text-align:left}th{background:#e4eef3}.tablewrap{overflow:auto}video{width:100%;max-height:600px;background:#142b3b}select,button{padding:10px;margin:8px;font-size:15px}#info{background:white;padding:15px}img{width:100%}.note{background:#fff1db;padding:15px;border-left:4px solid #bd7743}</style><h1>'+esc(title)+'</h1>'+body+'<script>'+script+'</script></html>'

human=read(HUMAN/'summary.json');training=read(TRAIN/'summary.json');assert human['status']=='COMPLETE' and human['exported_videos']==200 and training['status']=='COMPLETE'
assert len(training['tasks'])==6
for row in human['rows']:assert sha(HUMAN/row['file'])==row['sha256']
for task in training['tasks']:
    for e in task['source_events']:assert sha(TRAIN/e['local_relative'])==e['sha256']
    assert task['metrics']['Train/mean_reward']['points']==2000

# Separate, directly playable local folders; avoid double storage for existing media.
high=HUMAN/'高清展示6条';high.mkdir(exist_ok=True)
for p in (PACK/'media/original_human_generation_20261002').glob('*.mp4'):
    dest=high/p.name
    if not dest.exists():dest.hardlink_to(p)
    assert sha(dest)==sha(p)
browser=read(PACK/'页面验收.json');duration_map={Path(unquote(urlparse(x['file']).path)).name:x['duration_seconds'] for x in browser['videos']}
robots=[]
for p in sorted((PACK/'media').rglob('*.mp4')):
    if 'original_human_generation_20261002' in p.parts:continue
    rel=p.relative_to(PACK/'media')
    category='01_Isaac_Lab_官方基线' if len(rel.parts)==1 else ('02_MuJoCo_跨仿真' if rel.parts[0].startswith('sim2sim') else '03_扰动与实体接触')
    dest=ROBOT/category/p.name;dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():dest.hardlink_to(p)
    assert sha(dest)==sha(p)
    robots.append({'file':dest.relative_to(ROBOT).as_posix(),'category':category,'duration_seconds':duration_map[p.name],'sha256':sha(p),'source':str(p.relative_to(OUT))})
assert len(robots)==27
(ROBOT/'视频清单.json').write_text(json.dumps({'videos':robots,'count':27,'scope':'6 official Isaac baseline videos, 9 MuJoCo videos and 12 robustness/contact videos; no CLoSD official-evaluation video was saved'},ensure_ascii=False,indent=2),encoding='utf-8')
with (HUMAN/'逐条质量筛查.csv').open('w',encoding='utf-8-sig',newline='') as f:
    fields=['seed','sample_id','duration_seconds','any_proxy_flag','foot_below_nominal_ground','near_ground_slide_proxy','both_feet_airborne_proxy','large_per_frame_joint_change','bone_length_variation_proxy','foot_min_y_m','sliding_longest_frames','max_joint_change_m_per_frame','max_bone_relative_length_deviation','file']
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for r in human['rows']:
        qa=r['quality_proxy'];row={k:r[k] for k in ['seed','sample_id','duration_seconds','file']};row.update({k:qa[k] for k in ['any_proxy_flag','foot_min_y_m','sliding_longest_frames','max_joint_change_m_per_frame','max_bone_relative_length_deviation']});row.update(qa['flags']);w.writerow(row)

flags={'foot_below_nominal_ground':'脚端持续低于名义地面','near_ground_slide_proxy':'近地面疑似滑脚','both_feet_airborne_proxy':'双脚持续离地','large_per_frame_joint_change':'关节单帧位移提醒','bone_length_variation_proxy':'骨段长度变化提醒'}
items=[]
for r in human['rows']:
    items.append({'file':r['file'],'label':f"Kick｜seed{r['seed']}｜编号{r['sample_id']:03d}",'info':'9.8秒；20fps；原始22关节骨架。命中代理：'+('、'.join(flags[k] for k,v in r['quality_proxy']['flags'].items() if v) or '无')})
def playlist(title,items,preamble,links):
    body='<p>'+preamble+'</p><p>'+links+'</p><select id="select">'+''.join(f'<option value="{i}">{esc(x["label"])}</option>' for i,x in enumerate(items))+'</select><button id="prev">上一段</button><button id="next">下一段</button><div id="info"></div><video id="player" controls preload="metadata"></video><p id="download"></p>'
    data=json.dumps(items,ensure_ascii=False).replace('</','<\\/')
    js='const clips='+data+';const s=document.querySelector("#select"),v=document.querySelector("#player");function show(i){s.value=i;v.src=clips[i].file;document.querySelector("#info").textContent=clips[i].info;const a=document.createElement("a");a.href=clips[i].file;a.textContent="直接打开/另存当前MP4";document.querySelector("#download").replaceChildren(a)}s.onchange=()=>show(+s.value);document.querySelector("#prev").onclick=()=>show((+s.value+clips.length-1)%clips.length);document.querySelector("#next").onclick=()=>show((+s.value+1)%clips.length);show(0);'
    return standalone(title,body,js)
(HUMAN/'人体视频观看.html').write_text(playlist('原始人体生成｜全部200条Kick','' if False else items,'两批各100条，共200条完整动作，每条9.8秒。保留全部编号，不按结果筛选。单一右腿踢文本，种子10/42的空间提示也不同。近地面滑动等为几何代理，不能据此认定真实物理失败；快踢动作也可能触发单帧位移提醒。',href('逐条质量筛查.csv','逐条筛查CSV')+' · '+href('summary.json','阈值与完整数据')+' · '+href('高清展示6条/','6条高清双视图MP4')+' · '+href('../实验结果汇报.html','返回汇报')),encoding='utf-8')
ritems=[{'file':r['file'],'label':r['category']+'｜'+Path(r['file']).stem,'info':f"时长{r['duration_seconds']:.2f}秒。类别：{r['category']}；这是机器人执行，不是人体扩散输出。"} for r in robots]
(ROBOT/'机器人视频观看.html').write_text(playlist('机器人执行｜本地27段视频',ritems,'官方Isaac基线6段、MuJoCo跨仿真9段、扰动与实体接触12段。包括失败片段；视频播放结束不等于任务成功。CLoSD两组官方4096环境评估没有保存视频。',href('视频清单.json','时长与SHA清单')+' · '+href('../实验结果汇报.html','返回汇报')),encoding='utf-8')
curvebody='<p>六项官方已完成基线的真实TensorBoard数据，2000轮/任务。原始events、全量标量CSV、静态PNG/SVG均在本目录。Punch按初次0—1599＋续训1600—1999拼接；原事件仍保留回滚部分，查看原始TensorBoard时需按运行段区分。</p>'
for x in training['tasks']:
    task=x['task'];curvebody+='<h2>'+esc(task)+'</h2><img src="'+quote(task+'/training_curves.svg')+'" alt="'+esc(task)+'训练曲线"><p>'+href(task+'/iteration_scalars.csv','每轮所有指标CSV')+' · '+href(task+'/all_scalars_raw.csv.gz','原始全部标量（压缩CSV）')+' · '+href(task+'/summary.json','摘要与来源事件哈希')+'</p>'
(TRAIN/'训练曲线总览.html').write_text(standalone('Isaac Lab PPO训练曲线｜六项官方基线',curvebody),encoding='utf-8')
(TRAIN/'打开TensorBoard说明.txt').write_text('此目录保存原始TensorBoard事件文件，并已导出无需服务即可观看的训练曲线总览.html、PNG/SVG和CSV。\n若本机已有TensorBoard，可运行：tensorboard --logdir "'+str(TRAIN)+'" --port 6020 --host 127.0.0.1\n浏览器地址：http://127.0.0.1:6020/\n此说明不代表6020服务已启动。Punch必须区分初次与恢复两个run；初次1600—1618属于恢复前回滚段，合并CSV/图已去掉重复段。\nTrain/mean_reward/time的横轴为各run累计秒数，不是PPO迭代；iteration_scalars.csv仅使用迭代轴指标。\n',encoding='utf-8')

# Build additions from verified exported data. No innovation experiment is introduced.
quality='<div class="note"><b>本地完整人体视频：</b>'+href('人体生成视频/人体视频观看.html','打开全部200条人体视频')+' · '+href('人体生成视频/逐条质量筛查.csv','逐条质量筛查CSV')+'。每条原始轨迹都导出了一段完整MP4；另附原先6条高清双视图展示，是相同样本的另一种画面，不增加独立动作数量。</div>'
quality+=table(['生成批次','文本类别','原始轨迹 / 完整MP4','每段时长 / 帧率','近地面疑似滑脚','宽松代理至少一项提醒'],[[f"seed{b['seed']}",'右腿Kick',f"{b['count']} / {b['video_count']}",'9.8秒 / 20fps',f"{b['flag_counts_overlapping']['near_ground_slide_proxy']} / 100",f"{b['proxy_flagged_count']} / 100"] for b in human['batches']])
quality+='<p><b>数量口径：</b>200条独立轨迹、200段完整视频，总时长1960秒（32分40秒）；6段高清展示副本覆盖其中6条。22关节、每条196帧；只有一个右腿踢文本类别。机器人官方基线使用各自原参考库，<b>没有把这200条混入六项论文基线的训练集</b>。</p>'
quality+=table(['代理项目','总数（可重叠）','本次筛查阈值 / 含义'],[
 ['脚端持续穿名义地面','0 / 200','脚端y&lt;−0.03m，连续≥3帧；名义地面保持原y=0，脚端关节不等同脚底网格'],
 ['近地面疑似滑脚','194 / 200','脚端连续两帧位于−0.03—0.05m内，水平速度&gt;0.2m/s，连续≥3个帧间隔（约0.15秒）'],
 ['双脚持续离地','0 / 200','左右脚端均高于0.10m，连续≥6帧（约0.30秒）；腾空踢腿也可能合理'],
 ['关节单帧位移提醒','200 / 200','任一关节单帧位移&gt;0.15m；20fps下相当于速度&gt;3m/s，快速踢腿也能触发，<b>不是突变/失败判定</b>'],
 ['骨段长度变化提醒','200 / 200','相对该序列骨段长度中位数偏差&gt;10%，连续≥3帧；短骨段对此比例尤其敏感'],
 ['真正动力学不合理的动作数','尚未测定','这200条人体骨架尚未逐条进行接触力、力矩、碰撞和动态稳定性验收；<b>不能把200条代理提醒写成200条物理失败，也不能把无提醒当物理通过</b>']])
quality+='<p>这些阈值是本次报告的探索性几何筛查，不是论文统一的物理质量判据，也没有人工质量评分。全部样本和负结果都保留；未做平滑、额外IK、足部锁定或人为挪动身体。'+href('人体生成视频/summary.json','原始数组SHA、代理阈值与逐条统计')+'</p>'

trainbody='<p>人体扩散生成器与机器人控制器是两套模型：这里PPO更新的是机器人Actor/Critic和控制策略，<b>不更新OmniControl扩散权重</b>。控制器读取机器人当前状态与参考动作/目标，通过Isaac Lab中的物理执行结果获得奖励。</p>'
trainbody+=table(['训练项目','实际设置 / 文件依据'],[
 ['仿真与控制频率','Isaac Lab + Isaac Sim 4.5/PhysX；物理dt=0.005秒（200Hz），decimation=4，控制周期0.02秒（50Hz），单回合10秒'],
 ['并行与预算','每任务8192环境，2000个PPO迭代，seed42；每环境每轮采24步＝每轮196608条转移，2000轮合计约3.932亿条环境转移（不是3.932亿条人体视频）'],
 ['网络与归一化','Actor/Critic隐藏层512→256→256，ELU；empirical_normalization=true；策略输出关节控制命令，经执行器/物理引擎作用到机器人'],
 ['PPO更新','每批5个学习epoch、4个mini-batch；clip=0.2，gamma=0.99，GAE lambda=0.95，entropy coefficient=0.008，max_grad_norm=1.0'],
 ['学习率与保存','配置名义learning_rate=0.001、adaptive、desired_kl=0.01；实际学习率会调整，以Loss/learning_rate原始记录为准。每50轮保存，最终model_1999.pt'],
 ['优化目标','存活、参考关节/关键点跟踪、位置/方向误差、任务目标等；同时惩罚力矩、加速度、动作变化和滑脚。不同任务的项与尺度有差别，以各env.yaml和CSV为准'],
 ['logger与可视化','logger=tensorboard；直接读取原events，无人工补点。训练与独立画面监控分进程，画面故障不能阻塞训练']])
trainbody+='<div class="note">浅色线为每轮原始记录，深色线为因果50轮滑动均值；价值损失采用对数纵轴。Punch恢复点1600以虚线标出，平滑不跨恢复点。数据摘要比较前100轮与末100轮的均值；不同任务reward不可直接比较高低。</div>'
rows=[]
for x in training['tasks']:
    m=x['metrics'];r=m['Train/mean_reward'];v=m['Loss/value_function'];s=m['Loss/surrogate']
    rows.append([esc(x['task']),f"{r['first100_mean']:.3f} → {r['last100_mean']:.3f}",f"{r['last100_std']:.3f}",f"{v['first100_mean']:.6f} → {v['last100_mean']:.6f}",f"{s['first100_mean']:.6f} → {s['last100_mean']:.6f}",'0—1999；2000点'])
trainbody+=table(['任务','平均reward：前100→末100','末100 reward标准差','Value loss：前100→末100','Surrogate loss：前100→末100','记录范围'],rows)
trainbody+='<p><b>观察：</b>六项任务的训练回报由约−10提高至正值，价值函数误差显著减小，末期平均回合长度达到500个控制步附近。<b>解释：</b>策略在训练参考和奖励下改善了存活与执行，价值网络更接近该策略回报。PPO surrogate loss是带符号的代理目标；接近零不单独证明好动作。熵和动作噪声反映探索程度，也不是“越低越好”。<b>结论边界：</b>这些曲线不能替代正式成功率、自然动作、实际接触或MuJoCo迁移验证；已有迁移失败仍保留。</p>'
for i,x in enumerate(training['tasks']):
    t=x['task'];trainbody+='<details '+('open ' if i==0 else '')+'class="training-chart"><summary>'+esc(t)+'｜reward、value loss、PPO loss、熵、回合长度、动作噪声</summary><img class="curve-img" loading="lazy" src="'+quote('PPO训练曲线/'+t+'/training_curves.svg')+'" alt="'+esc(t)+'真实训练曲线"><p>'+href('PPO训练曲线/'+t+'/iteration_scalars.csv','每轮所有指标CSV')+' · '+href('PPO训练曲线/'+t+'/all_scalars_raw.csv.gz','原始全量标量CSV.gz')+' · '+href('PPO训练曲线/'+t+'/summary.json','来源run、events SHA与摘要')+'</p></details>'
trainbody+='<p>'+href('PPO训练曲线/训练曲线总览.html','独立离线曲线总览')+' · '+href('PPO训练曲线/打开TensorBoard说明.txt','使用原始events重开本地TensorBoard的方法')+'</p><p>原始二进制事件已本地保存。Punch初次0—1618轮中，1600—1618是恢复后回滚的旧分支，合并曲线只取初次0—1599＋恢复1600—1999；保留两run的原events和raw CSV，不把重叠段算成额外训练成果。/time标签横轴是每run累计秒数，因此未混入迭代轴图。仍在训练的后续任务不混进这六项已完成数据。</p>'

artifactbody='<div class="method-flow"><span>条件 / 预训练权重</span><b>→</b><span>人体results.npy / 视频</span><b>→</b><span>G1参考.pkl</span><b>→</b><span>场景与PPO配置.yaml</span><b>→</b><span>events / model.pt</span><b>→</b><span>评估.json / 执行.mp4</span><b>→</b><span>MuJoCo迁移与接触记录</span></div>'
artifactbody+='<p>按文件理解流程：视频供人观看，数组供重定向和检查，参考轨迹供控制器跟踪，检查点保存策略，日志记录优化过程，JSON保存评估口径和结果。它们的用途不同，不能用某一段漂亮的视频替代所有步骤。</p>'
artifactbody+=table(['步骤','输入 → 导出产物','里面记录什么 / 后续用途','本次在哪里查看'],[
 ['1 条件扩散推理','官方model_humanml3d.pt＋文本/空间hint＋seed → results.npy','motion：100×22×3×196人体世界坐标；text、lengths、hint、num_samples、num_repetitions。不是G1关节角，也不包含接触力。两批各100条。',href('人体生成视频/summary.json','原始路径与SHA')],
 ['2 人体可视化与代理筛查','results.npy → 原始数组.npz、200个.mp4、逐条CSV/summary.json','NPZ无平滑地压缩保存motion/text/hint/lengths；MP4为原速骨架，两视图；CSV是阈值提醒，不是物理通过标签。',href('人体生成视频/人体视频观看.html','200条视频入口')+' · '+href('人体生成视频/Kick_seed10_original_arrays.npz','seed10数组')+' · '+href('人体生成视频/Kick_seed42_original_arrays.npz','seed42数组')],
 ['3 人体→G1原始重定向与修整','人体关节 → *_sim1/*.pkl → *_sim2/*.pkl','将人体轨迹转成G1的根位置/姿态、关节和关键点参考；官方修整后作为MotionLib参考。PPO读取sim2，MP4本身不作为训练输入。原参考.p​​kl留服务器。',href(PACK.name+'/results/r3_dreamcontrol_remaining_manifest.json','逐任务参考文件与哈希')],
 ['4 场景/策略配置','机器人USD/任务物体/碰撞、参考库 → env.yaml＋agent.yaml','env记录dt、驱动、参考路径、奖励、终止和随机化；agent记录网络、PPO、logger。视频场景显示适配与物理配置差异须单独记录。',href(PACK.name+'/results/configs/','六项保存的配置目录')],
 ['5 PPO训练与检查点','8192环境交互 → model_0/50/…/1999.pt；events.out.tfevents.*；train.log','检查点保存Actor/Critic、优化器和观测归一化等恢复信息；events保存step、wall_time、tag、value；train.log记录迭代、运行路径/异常。完整模型和文本训练日志主要仍在服务器，路径/SHA在结果JSON。',href('PPO训练曲线/训练曲线总览.html','真实logger曲线')],
 ['6 logger离线导出','原events → iteration_scalars.csv、all_scalars_raw.csv.gz、PNG/SVG、summary.json','逐轮长表可自己作图；raw保留全部tag和/time轴；summary含源事件SHA和前/末100轮统计。原events也本地归档，可重新开TensorBoard。',href('PPO训练曲线/','完整logger导出目录')],
 ['7 官方评估与机器人视频','冻结model_1999.pt＋官方play_eval.py → eval.log、结果JSON、MP4','1000环境、500帧的任务成功率；JSON给出口径、退出码、SHA；六段501帧视频各10.02秒。高分不单独证明自然坐姿或真实抓握。',href('机器人执行视频/机器人视频观看.html','本地机器人视频')+' · '+href(PACK.name+'/results/','评估JSON')],
 ['8 冻结策略导出 / MuJoCo','actor＋normalizer → TorchScript策略；参考→NPZ；MJCF/XML → rollout JSON/MP4、接口/接触记录','执行相同已训练策略，不继续学习。核对actor/normalizer张量、观测动作、参考插值与驱动；逐参考记录摔倒、接触和抬物等。完整部署权重/机器人资产留服务器，结果和代码路径已归档。',href(PACK.name+'/results/sim2sim_completed_20261002/summary.json','四项迁移逐条记录')+' · '+href(PACK.name+'/results/mujoco_robust_contact/summary.json','踢拳扰动/接触记录')],
 ['9 汇报与追溯','数组 / events / 视频 / JSON → 本HTML、观看目录、曲线目录、产物清单','本地离线直接观看。保留种子、样本编号、原始时长、哈希和失败记录；新增清单覆盖本次媒体与logger资料。',href('汇报新增产物清单_20261002.json','新增产物与SHA清单')]])
artifactbody+='<p><b>本地三处直接观看：</b>'+href('人体生成视频/人体视频观看.html','① 人体生成视频')+' · '+href('机器人执行视频/机器人视频观看.html','② 机器人执行视频')+' · '+href('PPO训练曲线/训练曲线总览.html','③ PPO训练曲线')+'。发送给老师时可复制整个本汇报文件夹，或者单独复制这三个目录；各观看页与目录内MP4均可离线使用。主报告仍需与“论文复现汇报资料_20261002”及这三个目录一起携带。</p>'

backup=OUT/'实验结果汇报_仅方法与6段人体展示备份.html'
if not backup.exists():shutil.copy2(REPORT,backup)
page=REPORT.read_text(encoding='utf-8')
for id in ['training','artifacts']:page=re.sub(r'<section id="'+id+r'">.*?</section>','',page,flags=re.S)
page=re.sub(r'<!--HUMAN_EXPORT_DETAILS-->.*?<!--END_HUMAN_EXPORT_DETAILS-->','',page,flags=re.S)
where=page.index('<section id="generation">');a=page[:where];b=page[where:]
insert=b.index('<div class="video-grid">');b=b[:insert]+'<!--HUMAN_EXPORT_DETAILS-->'+quality+'<!--END_HUMAN_EXPORT_DETAILS-->'+b[insert:];page=a+b
page=page.replace('<strong>6段</strong><span>原始扩散生成人体动作展示</span>','<strong>200段</strong><span>原始人体完整视频本地保存</span>')
page=page.replace('<section id="baseline">',section('training','Isaac Lab PPO：真实训练过程与曲线',trainbody)+section('artifacts','整个流程导出的文件：每一步有什么',artifactbody)+'<section id="baseline">',1)
nav=[('overview','01 汇报概况'),('methods','02 人体运动怎样生成'),('generation','03 人体视频与质量筛查'),('training','04 PPO训练过程与曲线'),('artifacts','05 每步产物文件'),('baseline','06 DreamControl控制基线'),('closd','07 CLoSD复现'),('sim','08 MuJoCo跨仿真'),('robust','09 扰动与实体接触'),('failures','10 问题与后续'),('videos','11 机器人执行视频'),('appendix','12 复现明细与证据')]
page=re.sub(r'<nav><b>实验汇报</b>.*?<small>','<nav><b>实验汇报</b>'+''.join(href('#'+k,v) for k,v in nav)+'<small>',page,count=1,flags=re.S)
page=page.replace('</style>','.curve-img{display:block;width:100%;height:auto;margin:18px 0;border-radius:10px}.training-chart summary{font-size:16px}.training-chart p{font-size:13px}@media print{.curve-img{max-width:100%}} </style>',1)
page=page.replace('供老师观察动作质量','供老师观察动作质量；新增200条完整视频与明确阈值筛查')
page=page.replace('数据截至 <b>2026年10月2日 16:04（北京时间）</b>','实验成绩截至 <b>2026年10月2日 16:04（北京时间）</b>；历史视频/日志本次完整归档于 <b>'+datetime.datetime.now().strftime('%m月%d日 %H:%M')+'</b>')
REPORT.write_text(page,encoding='utf-8')

files=[]
for folder in [HUMAN,ROBOT,TRAIN]:
    for p in sorted(folder.rglob('*')):
        if p.is_file():files.append({'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
metadata={'created_local':datetime.datetime.now().isoformat(),'unique_human_motions':200,'full_human_videos':200,'high_resolution_human_display_copies':6,'human_seconds_each':9.8,'robot_execution_videos':27,'completed_baseline_logger_tasks':6,'source_event_files':sum(len(t['source_events']) for t in training['tasks']),'physically_bad_original_human_motion_count':'not determined','near_ground_sliding_proxy_count':194,'proxy_thresholds_file':'人体生成视频/summary.json','report_sha256':sha(REPORT),'files':files}
(OUT/'汇报新增产物清单_20261002.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'打开汇报说明.txt').write_text('请打开实验结果汇报.html。新增人体生成视频（200段完整＋6段高清副本）、机器人执行视频（27段）、PPO训练曲线（6项，含原events）。三个目录均含可独立打开的观看HTML。\n人体视频的几何代理提醒不是动力学失败率，真正物理合理性尚未逐条判定。各步骤产物的用途与存放位置在主报告“每步产物文件”章节。\n发给老师可复制整个文件夹；主报告依赖论文复现汇报资料_20261002和上述三个目录。原创新验证版资料保留在完整档案备份，未加入当前汇报。\n',encoding='utf-8')
(HUMAN/'文件说明.txt').write_text('videos：两批全部200条原始人体完整视频，各9.8秒、20fps。高清展示6条：此前固定编号的高清双视图副本，不是新增6条动作。\nKick_seed*_original_arrays.npz：原始人体motion/text/hint/lengths；summary.json：源SHA、阈值、逐条检查与视频SHA。逐条质量筛查.csv为Excel可打开的UTF-8长表。\n这不是机器人执行，不包含力学验证，也没有平滑或额外IK。请用人体视频观看.html观看。\n',encoding='utf-8')
(ROBOT/'文件说明.txt').write_text('01_Isaac_Lab_官方基线：6个任务各1段，10.02秒。02_MuJoCo_跨仿真：9段。03_扰动与实体接触：12段。视频清单.json记录每段时长、来源和SHA。\n本目录与原归档采用同盘文件链接节省空间；复制到老师电脑后就是正常MP4文件。请用机器人视频观看.html观看。\n',encoding='utf-8')
for name in ['enhance_teacher_report_20261002.py','export_training_report_20261002.py','export_all_original_humans_20261002.py']:shutil.copy2(R/'scripts'/name,PACK/'生成与验收'/name)
print(json.dumps({'status':'COMPLETE','human_videos':200,'highres_display_copies':6,'robot_videos':27,'logger_tasks':6,'raw_event_files':metadata['source_event_files'],'new_local_files':len(files),'new_local_bytes':sum(x['bytes'] for x in files),'report_bytes':REPORT.stat().st_size},ensure_ascii=False))

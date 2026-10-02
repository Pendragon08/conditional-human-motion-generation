from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import quote,unquote,urlparse
import json,hashlib,shutil,datetime
OUT=Path(r'D:\具身智能\通用条件人体运动生成_已完成实验结果_20260929')
REPORT=OUT/'实验结果汇报.html';DATA=OUT/'人体生成视频/六任务官方原始样本';PROOF=OUT/'论文复现汇报资料_20261002/生成与验收'
manifest=json.loads((DATA/'six_official_tasks/manifest.json').read_text(encoding='utf8'))
sources=json.loads((DATA/'source_manifest.json').read_text(encoding='utf8'))
for row in sources:
    row['pointer_checkout_commit']=row.pop('commit',row.get('pointer_checkout_commit'))
    row['provenance_note']='Download URL uses official main branch; exact bytes verified against SHA256 and size of existing Git LFS pointers in local experiment checkout. Checkout commit is not claimed to be an official upstream commit.'
    assert hashlib.sha256((DATA/row['file']).read_bytes()).hexdigest()==row['sha256']
(DATA/'source_manifest.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf8')
for row in manifest['videos']:
    assert hashlib.sha256((DATA/'six_official_tasks'/row['file']).read_bytes()).hexdigest()==row['sha256']
    assert row['frames']==196 and row['duration_seconds']==9.8
old=REPORT.read_text(encoding='utf8');backup=OUT/'实验结果汇报_六任务视频补充前备份.html'
if not backup.exists():backup.write_text(old,encoding='utf8')
soup=BeautifulSoup(old,'html.parser');g=soup.find(id='generation')
qa=next(t for t in g.select('.tablewrap') if '真正动力学不合理' in t.get_text())
quality=next(t for t in g.select('.tablewrap') if '生成批次' in t.get_text())
qa_html=str(qa);quality_html=str(quality)
def esc(x):
    import html
    return html.escape(str(x))
def link(p,label):return '<a href="'+quote(str(p).replace('\\','/'),safe='/:#')+'">'+esc(label)+'</a>'
names={'Kick':'Kick｜右腿踢击','Punch':'Punch｜右手出拳','Jump':'Jump｜向前跳跃','Sit':'Sit｜走向座椅并坐下','Pick':'Pick｜单手取物并抬起','Bimanual_Pick':'Bimanual-Pick｜双手抬工具箱'}
texts={'Kick':'站立后用右腿踢击','Punch':'用右手完成一次拳击','Jump':'向前跳跃','Sit':'走向椅子并坐下','Pick':'走向杯子、从侧面抓取并抬起','Bimanual_Pick':'用双手抬起工具箱'}
body='<div class="eyebrow">阶段一 · 条件人体运动生成</div><h2>原始人体生成：六类任务</h2><p>采用DreamControl官方发布的OmniControl人体生成样本，输入为任务文本及关节位置—时间约束。下列视频呈现重定向前的22关节人体轨迹，不是机器人策略执行结果。</p>'
body+='<div class="note"><b>展示配置：</b>六类任务各固定取编号000、001、002，共18段；每段196帧、20fps、9.8秒、原速播放。原始坐标和空间提示保持不变，未进行平滑、额外IK或姿态修正。蓝色为人体，橙色为右侧肢体，红叉为当前有效空间约束。该骨架动画不含椅子、杯子或工具箱的物理模型，不能据此认定真实接触或抓取成功。</div>'
body+='<div class="tablewrap"><table><thead><tr><th>任务</th><th>文本条件</th><th>官方原始人体轨迹</th><th>本节展示</th></tr></thead><tbody>'
for task in manifest['tasks']:body+='<tr><td>'+esc(names[task['task']])+'</td><td>'+esc(texts[task['task']])+'</td><td>'+str(task['source_count'])+'条</td><td>3段 × 9.8秒</td></tr>'
body+='</tbody></table></div><p>六类官方原始轨迹共342条，本节导出18段代表样本（合计176.4秒）；全部原始数组已保存本地。342条是重定向与筛选前的人体样本数，与各任务实际PPO采用的G1参考条数不同。样本由作者发布，本次进行资产核验和视频渲染，未重新执行六类扩散推理。</p>'
body+='<div class="video-grid">'
for row in manifest['videos']:
    if row['sample_id']!=0:continue
    rel='人体生成视频/六任务官方原始样本/six_official_tasks/'
    body+='<article class="video-card human-preview" data-human-task="'+row['task']+'"><h3>'+esc(names[row['task']])+'</h3><video controls preload="metadata" poster="'+quote(rel+row['poster'])+'"><source src="'+quote(rel+row['file'])+'" type="video/mp4"></video><p>'+esc(row['text'])+'</p><p>官方原始人体 · 编号000 · 9.8秒 · 20fps</p></article>'
body+='</div><details><summary>同类别其他固定样本（编号001、002，共12段）</summary><div class="video-grid">'
for row in manifest['videos']:
    if row['sample_id']==0:continue
    rel='人体生成视频/六任务官方原始样本/six_official_tasks/'
    body+='<article class="video-card human-additional"><h3>'+esc(names[row['task']])+'｜'+str(row['sample_id']).zfill(3)+'</h3><video controls preload="none" poster="'+quote(rel+row['poster'])+'"><source src="'+quote(rel+row['file'])+'" type="video/mp4"></video><p>官方原始人体 · 9.8秒 · 20fps</p></article>'
body+='</div></details><p class="source">来源：'+link('https://github.com/GenRobo/DreamControl/tree/main/TrajGen/sample','DreamControl官方生成资产')+' · '+link('人体生成视频/六任务官方原始样本/source_manifest.json','六类原始数组来源与SHA256')+' · '+link('人体生成视频/六任务官方原始样本/six_official_tasks/manifest.json','18段视频编号、时长与SHA256')+'</p>'
body+='<details id="kick-original-geometric-audit"><summary>补充：本课题已生成的200条Kick及几何质量筛查</summary><p>另有此前在本课题完成的seed10/42两批Kick推理，各100条、9.8秒。200段完整视频已保留。这两批与上方官方发布的342条分开统计，未混入六项官方控制基线的训练集。下列194/200等筛查数仅适用于这两批Kick，不适用于六类官方样本。</p>'+quality_html+qa_html+'<p>上述阈值仅为探索性几何代理，不能转换为动力学失败率。全部样本保留，未做足部锁定、平滑或额外IK。</p><p>'+link('人体生成视频/人体视频观看.html','200条Kick完整视频')+' · '+link('人体生成视频/逐条质量筛查.csv','逐条几何代理记录')+' · '+link('人体生成视频/summary.json','代理阈值与来源')+'</p></details>'
g.clear()
for element in list(BeautifulSoup(body,'html.parser').contents):g.append(element)
sim=soup.find(id='sim');existing=sim.find(id='sim-purpose-paper')
if existing:existing.decompose()
purpose='''<div id="sim-purpose-paper"><h3>跨仿真验证的研究目的与论文依据</h3><p><b>研究目标：</b>在不同物理引擎中执行冻结策略，检验训练环境中的高分能否转化为可迁移的稳定控制与场景交互能力，为实机部署提供前置证据。</p><div class="tablewrap"><table><thead><tr><th>原文或官方说明</th><th>对应的验证动机</th></tr></thead><tbody><tr><td>DreamControl §III-A：人类运动先验旨在产生自然动作并支持sim-to-real。</td><td>生成骨架的自然外观不能替代机器人动力学验证，需要检验重定向和PPO后的实际执行。</td></tr><tr><td>§IV-D：部署时移除仿真特权观测并重新训练。</td><td>训练仿真成绩不能直接等同于部署能力；需要单独审计策略输入、驱动和控制接口。</td></tr><tr><td>附录§IX-D：仿真到实机的差异使精细取物成功率下降，部分任务采用减速处理。</td><td>迁移可能影响稳定性与精确接触，因此须保留跨平台失败并检查接触和动作时序。</td></tr><tr><td>官方Sim2Real README：“Test on mujoco simulator”位于G1硬件测试之前。</td><td>官方部署代码提供MuJoCo预测试入口；本次将其作为跨引擎执行检验的依据。</td></tr></tbody></table></div><p><b>本次验证定位：</b>“先在另一物理引擎中排查部署接口与动力学问题”是结合论文迁移目标和官方部署流程形成的实验设计解释。原文没有单列本报告这套六任务MuJoCo量化表，也未给出我们采用的扰动矩阵或实体打击阈值；这些属于复现扩展测试。当前迁移的是已训练的仿真基线策略，未将其冒充论文经去特权观测重训的实机策略。sim-to-sim通过也不等于sim-to-real通过。</p><p class="source">依据：<a href="https://arxiv.org/html/2509.14353v3#S3.SS1">DreamControl §III-A</a> · <a href="https://arxiv.org/html/2509.14353v3#S4.SS4">§IV-D</a> · <a href="https://arxiv.org/html/2509.14353v3#S9.SS4">附录§IX-D</a> · <a href="https://github.com/GenRobo/DreamControl/blob/main/Sim2Real/README.md#test-on-mujoco-simulator">官方MuJoCo部署预测试说明</a></p></div>'''
sim.find('h2').insert_after(BeautifulSoup(purpose,'html.parser'))
for text in soup.find_all(string=True):
    if '发送给老师时' in text:text.replace_with(str(text).replace('发送给老师时可复制整个本汇报文件夹','报告及媒体可整体复制本汇报文件夹'))
    if '人体运动怎样生成' in text:text.replace_with(str(text).replace('人体运动怎样生成','条件人体运动生成方法'))
# No instructional addressing of the presentation audience remains.
assert not any(p in soup.get_text() for p in ['老师可观察的问题','怎样看视频','怎么样看视频','供老师观察'])
for metric in soup.select('header strong'):
    if metric.get_text()=='200段':metric.string='218段'
for text in soup.find_all(string=True):
    if str(text)=='原始人体完整视频本地保存':text.replace_with('原始人体视频已保存（官方18＋自生成200）')
artifact=soup.find(id='artifacts')
artifact.append(BeautifulSoup('<p><b>本次新增官方人体资产：</b>六类原始results.npy共342条、六类固定展示MP4共18段及来源/哈希清单，均位于“人体生成视频/六任务官方原始样本”。这是作者发布的人体生成输出，本次只核验和可视化，不计入本课题新扩散推理数量。'+link('人体生成视频/六任务官方原始样本/source_manifest.json','官方资产清单')+'</p>','html.parser'))
REPORT.write_text(str(soup),encoding='utf8')
note={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'official_human_sequences':342,'official_task_count':6,'official_human_display_videos':18,'fixed_ids':[0,1,2],'seconds_each':9.8,'new_diffusion_inference':False,'independent_existing_kick_videos':200,'paper_basis':{'paper':'https://arxiv.org/html/2509.14353v3','sections':['III-A','IV-D','IX-D'],'official_deployment':'https://github.com/GenRobo/DreamControl/blob/main/Sim2Real/README.md#test-on-mujoco-simulator','interpretation':'sim2sim experimental rationale inferred from sim2real objective and explicit official MuJoCo pretest; six-task table, perturbations and contact metrics are local extensions'},'report_sha256':hashlib.sha256(REPORT.read_bytes()).hexdigest()}
(PROOF/'六任务人体视频与迁移目的修订.json').write_text(json.dumps(note,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(note,ensure_ascii=False))

"""Write the disabled quick-generator proposal, never a training-result report."""
import datetime, html, json, pathlib
B=pathlib.Path(__file__).resolve().parents[1];D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
p=json.loads((B/'configs/quick-generator-proposal-20261026.json').read_text(encoding='utf-8'))
r=json.loads((B/'results/quick_generator_plan_20261026/budget_v1/assessment.json').read_text(encoding='utf-8'))
assert p['launch_enabled'] is False and p['corpus_build_enabled'] is False
v=p['storage']['volumes'];G=1024**3
text=f'''# TextOp：快速生成器小样本方案与存储预算

更新：{datetime.datetime.now().astimezone().isoformat(timespec='seconds')}

**这是独立的禁用方案，不是已启动训练，也不是数据通过筛选。** 当前仍先完成5,000轮teacher及真实quick跟踪评估；本次只读取原服务器固定参数与磁盘元数据、生成本地配置和预算，没有运行模型、读取检查点张量、创建服务器保存目录或修改活动配置。采样teacher为{r['sample']['teacher_completed']:,}/5,000轮，更新{r['sample']['teacher_updates']:,}次。

为用户希望先快速跑通的目标，准备一个保留论文网络的小样本首轮方案：VAE2,000、LDM3,000个manager primitive attempts，各为原步数1%。它仍未启用，实际采用与否应由真实pilot、数据及模型质量检查决定；原VAE200k/LDM300k配置未变。

|项目|论文配置|独立快速方案（尚未启用）|
|---|---|---|
|VAE步数/三个课程阶段|200,000；100,000/50,000/50,000|2,000；1,000/500/500|
|LDM步数/三个课程阶段|300,000；100,000/100,000/100,000|3,000；1,000/1,000/1,000|
|网络与表示|G1 29DoF/69D；history2→future8，4 primitives|保持；batch512、lr1e-4，VAE9层/LDM8层、hidden512|
|扩散与条件|冻结CLIP；训练后冻结VAE；5步cosine、预测clean latent、CFG5|保持，不能替换成其他训练目标|
|自回归训练上限|VAE1.0、LDM0.8|沿用已验证运行适配公式|
|保存|原20,000步间隔|1,000步间隔；VAE保存2份、LDM3份，分别另留1份紧急恢复槽|
|评估|发布配置2,000间隔/10 primitives|500间隔/12 primitives；单独声明的quick计数选择|

每个数据batch含4个primitive，VAE/LDM预算对应500/750次数据batch抽取。manager计数包括跳过非有限梯度的训练尝试，**不保证等于实际优化更新数**；验证不得更新参数，跳过次数必须另记。quick评估12是3个完整四段batch，不能冒称与发布配置10完全一致。

课程和学习率退火时钟同步压缩100倍；这比“提前结束原论文训练”多了一项明确差异。quick权重不能标成原论文训练的精确续训；以后迁移到完整版时必须另标warm-start，不能改计数或混用checkpoint绑定。原40000轮teacher和200k/300k完整版继续作为保留方案。

## 数据与启动条件

64条代表动作目前只是候选（原train/val47/17），真实保留数量和文本对数均未知。只用实际quick评估后保留的**重定向参考动作**及原BABEL时间标注训练，不把执行轨迹视频里的机器人状态悄悄替换成训练目标。早期teacher失败不能证明动作绝对不可行；其余动作和所有原始文件保留。

启动前须重建与保留来源绑定的私有语料：原始/镜像同源划分、精确姿态重合检查、35原始姿态帧生成34特征帧的合法窗口、训练集专用69D归一化、真实冻结CLIP覆盖全部非空文本，并记录动作权重统计范围。不得复用未筛选2,296条诊断视图的mean/std或缓存作为已接纳quick数据。随后验证真实batch512、四段重叠、断点恢复与梯度，再决定启用独立quick入口。

当前该方案的训练入口仍未实现/验收，launch_enabled与corpus_build_enabled均false。不能将准备配置当作可以立即派发的训练器。VAE重建/rollout质量检查后才进入LDM，有限loss单独不能判质量通过。弱数据、过拟合与欠训练须保留在报告中。

## 两个卷分别预算

使用既有EMA关闭状态的序列化审计作参考：每份约VAE518,648,281字节、LDM225,964,011字节（含原1%估算余量），本次未重读权重或重跑序列化。拟为每份分别留600/300MiB，**这仍是保存上限的计划值，不是实测保证**。

|卷|快照可用|本方案预算（含保留空间）|快照余量|
|---|---:|---:|---:|
|个人/data|{v['data']['snapshot_free_bytes']/G:.2f}GiB|{v['data']['planned_total_bytes']/G:.2f}GiB|{v['data']['snapshot_margin_bytes']/G:.2f}GiB|
|个人/home|{v['home']['snapshot_free_bytes']/G:.2f}GiB|{v['home']['planned_total_bytes']/G:.2f}GiB|{v['home']['snapshot_margin_bytes']/G:.2f}GiB|

data预算包括VAE两份+一份紧急checkpoint、后续teacher5000完整/推理预留192MiB、quick语料/CLIP1GiB、少量执行轨迹/视频512MiB、日志512MiB及2GiB保留；home包括LDM三份+一份紧急checkpoint、日志512MiB及2GiB保留。两个卷都实际核查为本人UID1013、持久ext4、非符号链接父目录、可写，设备号不同；没有把两盘空闲直接相加。

拟保存VAE到个人项目quick_generator目录，LDM到/home/ouyanghao/textop-quick69-checkpoints-20261026/ldm；这些新目录尚未创建或启用。启动和每次保存仍需重查归属、持久性、实际文件大小与余量，超限安全停止并保留已有checkpoint。最终论文WBC、全量生成增强、更多种子/恢复及全库视频不在预算内，不能声称整个完整版空间已够。

首次链路若使用5k特权teacher，只能称“文本生成参考→仿真跟踪原型”；它不是论文最终可部署WBC或完整实时TextOp系统。真实生成—执行指标和实际视频仍未完成。

证据：configs/quick-generator-proposal-20261026.json、results/quick_generator_plan_20261026/budget_v1/assessment.json。固定论文参数与原服务器SHA一致，活动16文件和live控制代码未改；GPU/模型工作0、检查点读取或写入0、服务器新目录/配置写入0。
'''
(B/'reports/quick_generator_proposal_20261026.md').write_text(text,encoding='utf-8')
(D/'TextOp快速生成器方案与预算_20261026.md').write_text(text,encoding='utf-8')
try:
 import markdown
 body=markdown.markdown(text,extensions=['tables'])
except ImportError:body='<pre>'+html.escape(text)+'</pre>'
page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>TextOp快速生成器方案与预算</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:17px/1.7 system-ui;color:#203040}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd6df;padding:9px}th{background:#edf3f8}pre{white-space:pre-wrap}</style>'+body+'</html>'
(D/'TextOp快速生成器方案与预算_20261026.html').write_text(page,encoding='utf-8')
print(json.dumps(dict(status='DISABLED_QUICK_GENERATOR_PROPOSAL_REPORT_SAVED',teacher_completed=r['sample']['teacher_completed'],generator_started=False)))

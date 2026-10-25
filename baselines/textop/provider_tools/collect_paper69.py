"""Collect numerical evidence only; no motion assets, weights, or training."""
import hashlib,html,json,pathlib,shutil
from session import REPO,remote
BASE=REPO/'baselines/textop';LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024')
ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
payload=json.loads(remote("""
import pathlib,json
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024/receipts');out={}
for name in ['paper69_author_v1.json','paper69_doc_v1.json']:
    p=r/name
    if p.exists():out[name]=json.loads(p.read_text())
out['failures']={p.name:p.read_text() for p in r.glob('paper69_author_v1.failure_*.txt')}
print(json.dumps(out))
"""))
dest=BASE/'results/paper69_runtime_20261025';dest.mkdir(parents=True,exist_ok=True)
for name,data in payload.items():
    if name=='failures':
        for key,trace in data.items():(dest/key).write_text(trace,encoding='utf-8')
    else:(dest/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
author=payload['paper69_author_v1.json'];doc=payload['paper69_doc_v1.json']
assert author['sentinel']==doc['sentinel']=='TEXTOP_PAPER69_NUMERICAL_WITNESS_PASSED'
assert author['source_hashes']==doc['source_hashes']
for name,digest in author['source_hashes'].items():assert hashlib.sha256((BASE/'scripts'/name).read_bytes()).hexdigest()==digest
record={'core69_numerical_validation_passed':True,'fresh_document_witness_passed':True,'source_hashes_match':True,'optimizer_updates':0,'training_performed':False,'dataset_accepted':False,'full_dataloader_integration_validated':False,'paper_geometry_body_selection_exactly_known':False,'source_failures_preserved':len(payload['failures'])}
(dest/'summary.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
planpath=BASE/'configs/paper-step2.json';plan=json.loads(planpath.read_text(encoding='utf-8'))
plan['status']='AUTHORIZED_WAITING_FOR_STEP1_AND_ACCEPTED_CORPUS_INTEGRATION'
plan['launch_reason']='69D core numerical witness and independent documented rerun passed; dynamic data filtering, accepted-corpus statistics/cache and full dataloader integration are still pending'
plan['adapter_implementation']={'path':'scripts/paper69_runtime.py','process_local':True,'source_hashes':author['source_hashes'],'core_numerical_validation_passed':True,'fresh_document_witness_passed':True,'full_training_ready':False}
plan['source_differences_resolved']=['29DoF/69D codec and38-body differentiable FK verified against NumPy fixture and MuJoCo, six wrist gradients nonzero','LDM rollout capped0.8 while VAE remains1.0','int32-to-int64 timestep compatibility at denoiser and coefficient indexing','Numerical sampling uses FIXED_LARGE to match paper beta_t reverse variance at noise-bearing steps; upstream FIXED_SMALL preserved']
plan['source_differences_pending']=['The final69D geometry-body subset and endpoint velocity convention are not fully specified by authors; own adapter uses all38 non-world GMR frames and repeats last finite difference at50Hz','Real filtered corpus export, training-only statistics, full CLIP cache and end-to-end loader integration have not yet been validated','Author-private data and evaluation resources may be unavailable; public-only runs remain explicitly scoped']
planpath.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
report=f'''# 69D生成器核心适配与独立验证

已完成独立适配层，并通过作者执行和另一代理按文档原样执行两次验证。固定upstream和共享环境未修改，未安装新软件。完整调用见PAPER69_ENVIRONMENT.md，环境规格hash934d8221。

| 检查 | 实测 |
| --- | --- |
| VAE输出/潜变量 | [2,8,69] / [1,2,128] |
| LDM潜变量输出 | [2,1,128] |
| VAE/LDM有限梯度参数张量数 | {author['vae']['finite_gradient_parameters']} / {author['ldm']['finite_gradient_parameters']} |
| FK对MuJoCo最大位置误差 | {author['fk_mujoco_error']['position_m']:.3g}m |
| FK最大旋转矩阵误差 | {author['fk_mujoco_error']['rotation_matrix']:.3g} |
| 编码对已保存NumPy特征最大误差 | {author['codec_error']['stored_numpy_feature']:.3g} |
| 六个腕关节几何梯度 | 全部非零、有限 |
| 文本/历史条件 | 均能影响输出；LDM检查期间VAE、CLIP冻结 |
| 连续数值采样 | 5步DDPM、CFG5，两段8帧，严格复用末2帧历史 |
| 小批量CUDA峰值 | {author['peak_allocated_bytes']/1024**2:.2f}MiB，batch2 |

**这不是训练结果：没有构造优化器，更新次数0；未接受数据集，未评价语义质量或机器人执行。** 所用两条未过滤候选运动仅作数值夹具；归一化为数值检查的恒等变换，不代替真实训练统计。上述显存不能外推完整batch512训练或RL显存。

适配包含29关节/69D编解码、MuJoCo关节地址与固定连杆映射的可微FK、正确的四元数顺序、50Hz关节差分、LDM rollout上限0.8和时间步int64兼容。复用官方VAE、Denoiser、几何损失及DDPM主体。两次早期采样失败日志完整保留，分别暴露了旧PyTorch下位置编码及系数索引的int32兼容问题。

几何损失使用固定GMR资产的38个非world骨架帧；论文未公开最终69D骨架帧子集，本选择明确列为自实现假设。末帧速度重复最后一个差分；采样采用FIXED_LARGE，噪声步骤方差对应beta_t，终止步不加噪声。旧公开FIXED_SMALL配置保留在upstream，未冒称两者相同。

**仍需完成**：真实策略筛选、合格配对集、训练集统计及CLIP缓存、全量数据加载器集成验证和训练启动前磁盘/GPU检查。正式VAE→LDM训练仍受这些条件约束，原57D训练保持暂停，已有数据/检查点/实验记录保留。
'''
(BASE/'reports/paper69_runtime_validation_20261025.md').write_text(report,encoding='utf-8')
(LOCAL/'69D核心适配与数值验证_20261025.md').write_text(report,encoding='utf-8')
page='<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8"><title>69D核心适配数值验证</title><style>body{font:16px/1.8 system-ui;max-width:1000px;margin:36px auto;padding:24px;background:#f6f8fa}pre{white-space:pre-wrap;background:#fff;padding:24px;font:15px/1.9 system-ui}</style><h1>69D核心适配与数值验证</h1><p>数值验证已通过；正式训练和数据物理筛选未完成。</p><pre>'+html.escape(report)+'</pre></html>'
(LOCAL/'69D核心适配与数值验证_20261025.html').write_text(page,encoding='utf-8')
with (BASE/'SERVER_NOTES.md').open('a',encoding='utf-8') as f:f.write('\nValidated textop-paper69@934d8221: seeded author and fresh verbatim document runs both exit0, identical source hashes, TEXTOP_PAPER69_NUMERICAL_WITNESS_PASSED. Core codec/FK/VAE/LDM/CLIP/sampling only; batch2 peak701244928 bytes; optimizer_updates0, dataset_acceptedfalse. Full corpus integration and training remain pending. Failed int32 sampling traces preserved.\n')
for p in [BASE/'PAPER69_ENVIRONMENT.md',planpath]:shutil.copy2(p,LOCAL/p.name)
print(json.dumps(record))

// Synchronize bounded evidence and current phase; never mutate live queue state.
const fs = require('fs');
const cp = require('child_process');
const path = require('path');
const local = path.resolve(__dirname, '..');
const remote = '/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029';
const py = `import json,hashlib
from pathlib import Path
R=Path('${remote}')
paths={
 'pick_primary_evaluation.json':'stage3_skill_sequence_20261103/evaluations/pick_final_3000_v1/report.json',
 'pick_contact_diagnostic.json':'stage3_skill_sequence_20261103/evaluations/pick_contact_diagnostic_3000_v2/report.json',
 'pick_independent_audit.json':'stage3_skill_sequence_20261103/completions/pick_independent_audit_20261104.json',
 'model_author_smoke.json':'stage3_supermimic29_20261103/reports/physx_model_author_smoke_v1/status.json',
 'model_doc_witness.json':'stage3_supermimic29_20261103/reports/physx_model_doc_witness_v1/status.json',
 'runtime_bridge_spec.json':'stage3_supermimic29_20261103/runtime_bridge_spec.json',
 'sequence_progress.json':'stage3_skill_sequence_20261103/progress.json',
 'online_author_witness.json':'stage3_supermimic29_20261103/reports/online_author_v1/report.json',
 'online_doc_witness.json':'stage3_supermimic29_20261103/reports/online_doc_v1/report.json'}
out={}
for name,rel in paths.items():
 p=R/rel
 if p.exists():out[name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'data':json.loads(p.read_text())}
print(json.dumps(out))
`;
const raw = cp.execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -'],{input:py,encoding:'utf8',maxBuffer:2e6});
const evidence=JSON.parse(raw);
const out=path.join(local,'reports/stage3_supermimic29_20261103');fs.mkdirSync(out,{recursive:true});
for(const [name,item] of Object.entries(evidence))fs.writeFileSync(path.join(out,name),JSON.stringify(item.data,null,2)+'\n');
fs.writeFileSync(path.join(out,'evidence_locations_20261104.json'),JSON.stringify(Object.fromEntries(Object.entries(evidence).map(([k,v])=>[k,{path:v.path,sha256:v.sha256}])),null,2)+'\n');
const note=`## 2026-11-04：Pick PPO已完成3000次更新及100环境评估，转入SuperMimic适配

- Pick最终model_2999.pt SHA256 bc38324f751513e0f72da5d17eb47cacd96fef2a172ef20388c3156e094b6d7e；独立核对续训251..2999连续，累计3000更新，模型102输入/31输出有限。原训练、保护器和交接控制器均正常退出，不得重启Pick或将自然退出当故障。正式100环境评估1/100满足当前项目高度判据，不代表论文分布/真机成功；结果保持原样归档。
- 原评估接触时刻统计的浮点跨界诊断有误，独立新目录按整数控制步复测；同种子/模型仍1/100，75例到接触时刻，腕目标误差平均0.27223m。原主评估不覆盖，诊断报告pick_contact_diagnostic_3000_v2。低成功率如实保留，不能将完成预算当任务成功。
- 当前顺序pick_supermimic→其余七技能。TextOp原生IsaacLab3/MuJoCo-Warp场景无手/物体且Actor含未来参考，不能直接替换当前PhysX任务。选择复用固定SuperMimic源码的网络/归一化原语及世界模型+BPTT优化策略，在已有Isaac4.5运行时适配；不重装、不修改TextOp项目。属于明确标记的SuperMimic-style改编，不冒称原生CLI或作者实现。保持同PhysX资产/场景/奖励、29身体+14手指物理关节、31策略动作、102实时Actor且无参考；参考仅用于奖励头/初始化。
- 真实8环境128步/32世界模型更新/1次BPTT及独立照文档4环境64步/16模型更新/1次BPTT均通过；观测与原任务最大差1.19e-7，六项模型/优化器保存读回一致。证明有限接线，不证明任务成功、全程或8912环境资源。runtime_bridge_spec.json c7724ac5保留；其smoke_scope描述作者8环境版本，独立命令实际为4环境版本，二者已明确区分。
- 新online训练循环、32行有界回放和完整断点恢复正在验证。4项CPU回放顺序/回环/恢复/损坏拒绝测试通过；8912正式预算仍3000×24=72000真实向量步、641664000转移，世界模型热身计入，想象步不计；正式未启动，不能据原型通过标记就绪。检查stage3_supermimic29_20261103/adaptation_status.json及最新reports，勿重复启动已有见证。预留11GiB启动和8GiB+356MiB运行底线、12GiB显存/32GiB RSS；当前/data约11.55GB低于启动线，需真实资源核算与合规空间安排，不以降低预算放行。
- 本地证据reports/stage3_supermimic29_20261103；原数据、全部结果和检查点保留。其余29DoF任务尚待适配，不用Pick改名。只做仿真，Nymeria暂停，不重做数据/重定向/额外源语义筛选。

`;
for(const file of ['PLAN.md','STATUS.md']){
 const target=path.join(local,file);let value=fs.readFileSync(target,'utf8');
 if(!value.includes('## 2026-11-04：Pick PPO已完成3000次更新')){
  const end=value.indexOf('\n');value=value.slice(0,end+1)+'\n'+note+value.slice(end+1);fs.writeFileSync(target,value);
 }
}
for(const file of ['STAGE3_PICK_29DOF.md','STAGE3_SUPERMIMIC_HANDOFF.md']){
 const target=path.join(local,'reports',file);
 if(fs.existsSync(target)&&!fs.readFileSync(target,'utf8').includes('## 2026-11-04：Pick PPO已完成3000次更新'))fs.appendFileSync(target,'\n'+note);
}
const referencePlan=path.join(local,'reports/stage3_reference500_20261104/plan.json');
const state={phase:fs.existsSync(referencePlan)?'pick_reference500_expansion_before_further_formal_training':'supermimic_online_loop_validation',pick_ppo_completed_updates:3000,
 pick_primary_success_count:1,pick_primary_environments:100,pick_final_sha256:'bc38324f751513e0f72da5d17eb47cacd96fef2a172ef20388c3156e094b6d7e',
 primary_evaluation_preserved:true,formal_supermimic_started:false,
 core_author_witness:'reports/physx_model_author_smoke_v1/status.json',core_doc_witness:'reports/physx_model_doc_witness_v1/status.json',
 online_author_witness:evidence['online_author_witness.json']?.data||{status:'pending_check'},
 online_doc_witness:evidence['online_doc_witness.json']?.data||{status:'pending_check'},
 runtime_choice:'existing Isaac4.5 PhysX + pinned TextOp SuperMimic optimization adaptation; no installation',
 reference_target_per_task:fs.existsSync(referencePlan)?500:null,
 reference_policy:fs.existsSync(referencePlan)?'/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/stage3_pick29_20261103/reference_scale500_20261104/plan.json':null,
 next_required:fs.existsSync(referencePlan)?['500 admitted generated Pick references','new500-referencePPO baseline','matched500-referenceSuperMimiccomparison']:['online complete replay/optimizer/budget resume verification','8912 environment optimizer/resource check','formal resource/checkpoint admission'],
 other_tasks:'pending29DoFimplementation',training_ready:false};
fs.writeFileSync(path.join(out,'adaptation_status.json'),JSON.stringify(state,null,2)+'\n');
const remoteWrite=`import json\nfrom pathlib import Path\nD=Path('${remote}/stage3_supermimic29_20261103')\nstate=json.loads(${JSON.stringify(JSON.stringify(state))})\n(D/'adaptation_status.json').write_text(json.dumps(state,indent=2)+'\\n')\nwith (D/'COMPUTE_LEDGER.md').open('a') as f:f.write('\\n2026-11-04: author8env/128step/32WM and fresh independent4env/64step/16WM witnesses passed;1 BPTT each,all6 checkpoint roundtrips,live observation parity1.19e-7. Spec c7724ac5 smoke_scope names the author variant; documented independent command intentionally uses4env. No environment rebuild or dependency change. Full online runner and formal capacity are separately pending; not full-training ready.\\n')\nprint('RECORDED')\n`;
console.log(cp.execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -'],{input:remoteWrite,encoding:'utf8'}));
console.log(JSON.stringify({downloaded:Object.keys(evidence),local:out,phase:state.phase}));

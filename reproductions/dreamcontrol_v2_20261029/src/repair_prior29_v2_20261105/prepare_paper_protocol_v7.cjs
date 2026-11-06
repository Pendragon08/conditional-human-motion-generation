const fs = require('fs');
const path = require('path');
const base = path.join(__dirname, 'paper_protocol_v7');
function change(name, fn) {
  const file = path.join(base, name);
  const old = fs.readFileSync(file, 'utf8');
  const updated = fn(old.replace(/\r\n/g, '\n'));
  fs.writeFileSync(file, updated);
}
function once(s,a,b){ if(s.split(a).length!==2) throw Error('missing/nonunique '+a.slice(0,100)); return s.replace(a,b); }
change('sample.py', s => {
  if(s.includes('from conditions import build')) s=once(s,'from conditions import build','from paper_conditions import build');
  if(!s.includes('task_sources["paper_conditions.py"] =')) s=once(s,'    urdf, task_sources = load_task_dependencies(cfg)','    urdf, task_sources = load_task_dependencies(cfg)\n    task_sources["paper_conditions.py"] = sha(Path(__file__).with_name("paper_conditions.py"))');
  s=s.replace('task_sources["conditions.py" if args.task == "pick"\n                                                                  else "stage3_supported_task_conditions.py"]','task_sources["paper_conditions.py"]');
  return s;
});
change('postopt_v2.py', s=>{
  const start=s.indexOf('        @jaxls.Cost.create_factory\n        def stage_a_preservation');
  const end=s.indexOf('        @jaxls.Cost.create_factory\n        def geometry_cost', start);
  if(start>=0 && end>start) s=s.slice(0,start)+s.slice(end);
  s=s.replace('"schema": "motion_preserving_postopt_v2"','"schema": "paper_protocol_postopt_v7", "paper_url": "https://arxiv.org/html/2604.00202v1#S5.SS4", "non_admission_metrics": "raw fidelity, correction size, wrist angle, foot speed, joint/frame changes; solver scales are not paper thresholds"');
  return s;
});
change('pipeline_controller_v2.py', s=>{
  s=once(s,'    first_id = 10_000_000 +','    first_id = 20_000_000 +');
  s=once(s,'    reuse = plan["reused_pick_raw"].get(str(number))','    reuse = None');
  s=once(s,'    tasks["pick"] = {"passed": True, "attempts": prior_pick}\n','');
  const start=s.indexOf('    audit = read(bound(plan["pick_seed_audit"]))');
  const end=s.indexOf('    def add(task, count):',start);
  if(start<0||end<start)throw Error('resume segment');
  s=s.slice(0,start)+s.slice(end);
  s=s.replace('plan["sources"]["postopt_pick" if task == "pick" else "postopt"]','plan["sources"]["postopt"]');
  s=s.replace('"native_pick": "stage3_pick500_replay.py", "native_supported": "stage3_supported_native_replay.py",','"native_pick": "native_pick.py", "native_supported": "native_supported.py",');
  return s;
});
const project = path.resolve(__dirname,'../..');
const update='## 2026-11-06｜用户授权按论文条件和筛选重建（覆盖历史自加门槛）\n\n'+
'- 用户明确要求取消自行加严限制。论文 V-D 腕目标误差 <15 cm，腕点避障 >3 cm；后者不能当腕目标阈值。原始动作保真、非腕位移、根/关节修正窗、腕角及帧间变化改为诊断指标，仍保留姿态、平滑、脚稳定与避障优化；保留关节限位、有限值、独立 FK、数值穿透检查。论文未给出的角度曲线、alpha_l 和 29DoF 场景坐标适配显式标注。\n'+
'- 独立 `src/repair_prior29_v2_20261105/paper_protocol_v7` 修改 Pour 肘提示与腕姿共享相位；先在保留的 10100005 上验证，再新四类先导与各500。现有 raw 加肘目标仅验证后优化，不能声称该 raw 生成时已有肘提示，不能算新协议生成样本。旧成果保留，旧 v6/PPO 不恢复。\n'+
'- 当前尚在检查/部署准备，不能称已生成或已开训。新库完整后五指标→SuperMimic→四项动态评估和视频。资源与正式训练磁盘门槛保持，GitHub 全部流量继续服务器发起。\n\n';
for(const name of ['PLAN.md','STATUS.md']){
 const file=path.join(project,name),s=fs.readFileSync(file,'utf8');
 const newline=s.indexOf('\n');fs.writeFileSync(file,s.slice(0,newline+1)+'\n'+update+s.slice(newline+1));
}
console.log('Prepared independent source and recorded current user authorization.');

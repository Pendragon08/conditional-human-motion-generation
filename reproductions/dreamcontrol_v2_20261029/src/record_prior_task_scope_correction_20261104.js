const fs=require('fs'),path=require('path'),{execFileSync}=require('child_process');
const L=path.resolve(__dirname,'..'),D=path.join(L,'reports/current_prior_task_coverage_20261104');
const report=JSON.parse(fs.readFileSync(path.join(D,'caption_retrieval_all_captions_v2.json'),'utf8'));
const md=`# 当前扩散先验的任务覆盖复核\n\n绑定当前10000更新检查点SHA ${report.checkpoint_sha256}，真实训练清单SHA ${report.manifest_sha256}。全数据3161条，训练2091条：GRAB903、OmniRetarget/OMOMO1188。已配对的ReactiveBFM11424条尚未用于本检查点训练；Nymeria暂停。MDM人体预训练可能保留其他运动能力，但不能替代机器人任务覆盖的证据。\n\n核对训练加载器实际随机使用的全部candidate_captions，而非只看首句。按轨迹计数，文字检索不是逐条语义验收，不证明完整抓取或物理成功，也不将零关键词结果当作动作绝不存在。\n\n| 任务/动作 | 当前训练文字证据 | 当前决策 |\n|---|---|---|\n| Pick/抓取抬起 | 1490条含pick/grab/grasp/lift，其中GRAB900、Omni590 | Pick继续500参考与独立PPO计划；仍需生成和真实仿真评估 |\n| Pour/倾倒 | GRAB16条，包含杯/茶壶/水瓶倒出内容物 | 有直接描述，后续优先小批任务验证 |\n| 双手搬箱/旋转/推拉 | Omni1188条共有7种发布描述 | 与当前数据更贴近的后续候选任务；尚未替用户改写论文任务库 |\n| Drawer Open | drawer文字检索0 | 未证覆盖，不扩展500批量生成 |\n| Punch/Step & Punch | punch/boxing检索0；唯一walk描述是玩具象行走 | 未证覆盖，不扩展500批量生成 |\n| Deep Squat/Deep Squat & Pick | squat/crouch检索0；搬箱不等于深蹲抓取 | 未证覆盖，需运动/条件小样或补充先验数据 |\n| Vertical Wipe | 2条wipe为擦眼镜，无竖直表面擦拭描述 | 不混同擦眼镜与擦墙 |\n\n论文III-A与IV-A：八技能结果默认来自Full-Mix（AMASS/HumanML3D+Nymeria+GRAB+OmniRetarget），不是仅GRAB+Omni子集；论文指出不同来源覆盖互补运动区域，Nymeria用于抽屉/柜子等交互。来源：https://arxiv.org/html/2604.00202#S4.SS1\n\n八任务各500保留为长期复现目标，不声明当前先验已支持全部八项。未证覆盖的其余任务需先得到数据/小批生成验证支持，或通过独立版本扩散训练补齐，再进行500条扩展。保持已授权Pick生成，不中断/重启已有流水线，不自动热换模型；Pick500完成后按既定质量/资源条件优先独立PPO3000。源GRAB/Omni不新增物理动作筛选、逐条人工语义门槛。\n\n视频里物体固定由参考回放设置造成，与训练覆盖是两个不同问题。后优化手腕到点也不能证明扩散先验学会完整物体交互或实际抓取。\n`;
fs.writeFileSync(path.join(D,'任务覆盖与范围说明.md'),md);
const script=String.raw`import json
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');Q=R/'stage3_skill_sequence_20261103';O=R/'stage3_pick29_20261103/reference_scale500_20261104'
p=Q/'sequence.json';j=json.loads(p.read_text())
j['policy']['task_distribution_gate']='Before extending a task to500, establish support in current robot prior data or bounded generation tests;8-skill paper Full-Mix is not evidence that the current GRAB+Omni-only fine-tune supports all8.'
j['policy']['current_model_task_scope']='Pick continues; remaining bulk generation conditional on task coverage/validation or a separately trained expanded-data prior. Preserve the long-term8x500 target and all existing outputs.'
for t in j['tasks']:
 if t['id']=='pick':t['current_prior_evidence']='pick/grasp/lift descriptions in actual training data; continue active reference generation'
 elif t['id']=='pour':t['current_prior_evidence']='16GRAB training trajectories with explicit pour captions; bounded task-generation/scene validation still required'
 elif t['id']!='pick_supermimic':t['current_prior_evidence']='task-specific support unverified in current2091-training-trajectory GRAB+Omni fine-tune; do not treat the paper8-task list as coverage proof'
tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(j,ensure_ascii=False,indent=2));tmp.replace(p)
print(json.dumps({'sequence_updated':str(p),'current_pick_kept_running':True,'other_bulk_generation_not_started':True}))
`;
process.stdout.write(execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -'],{input:script,encoding:'utf8',timeout:30000}));
const title='## 2026-11-04：当前GRAB＋Omni先验不能直接假定覆盖论文八技能';
const note=title+'\n\n- 根据用户质疑，核真实训练清单5fc1a0...及随机使用的全部candidate_captions：训练2091=GRAB903+Omni1188。pick/grasp/grab/lift检索1490条（GRAB900/Omni590），pour16条；drawer、punch/boxing、squat/crouch均0，wipe两条为擦眼镜而非竖直擦拭。Omni训练只有7类大箱子抬起/移动/旋转/推拉/踢动原文。只是文字覆盖线索，零关键词不证明动作绝不存在，有关键词不证明完整任务成功；不新增逐条源语义/物理筛选。完整报告reports/current_prior_task_coverage_20261104/caption_retrieval_all_captions_v2.json。\n- 论文八技能默认Full-Mix还包含AMASS/HumanML3D及Nymeria；已配对11424条未加入当前固定扩散检查点，MDM预训练残留不能当已核机器人覆盖。此前直接按论文八任务安排当前小基线全部扩展，范围过大。八项各500保留长期目标，Pick继续且优先PPO；其余批量需先证当前先验支持/小批验证，或独立版本补训练，再扩展，不自动热换当前模型，也不把推箱等新候选擅自替换用户的论文任务库。\n\n';
for(const name of ['PLAN.md','STATUS.md']){let f=path.join(L,name),s=fs.readFileSync(f,'utf8');if(!s.includes(title)){let i=s.indexOf('\n');fs.writeFileSync(f,s.slice(0,i+1)+'\n'+note+s.slice(i+1));}}

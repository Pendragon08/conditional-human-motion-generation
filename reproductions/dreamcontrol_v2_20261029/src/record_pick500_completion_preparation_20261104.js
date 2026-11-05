const fs=require('fs'),path=require('path'),{execFileSync}=require('child_process');
const script=String.raw`import json
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');O=R/'stage3_pick29_20261103/reference_scale500_20261104'
paths={'library_progress':O/'library_progress.json','replacement_status':O/'replacement_queue128_v1/status.json',
 'original500_cross_audit':O/'replacement_queue128_v1/original500_cross_audit.json',
 'completion_controller_launch':O/'completion_controller_v1_launch.json','completion_controller_status':O/'completion_controller_v1/status.json',
 'evaluation_entry4':O/'evaluation_entry4_v1/report.json','evaluation_entry4_guard':O/'evaluation_entry4_v1_guard.json'}
out={k:json.loads(p.read_text()) for k,p in paths.items()}
a=json.loads((O/'replacement_queue128_v1/remaining496_candidate_audit.json').read_text())
out['original496_audit_summary']={k:v for k,v in a.items() if k!='details'};out['original496_audit_summary']['detail_count']=len(a['details'])
out['original496_audit_summary']['all_noise_recomputed']=all(r['noise_recomputed'] for r in a['details'])
print(json.dumps(out))
`;
const j=JSON.parse(execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -'],{input:script,encoding:'utf8',timeout:30000}));
const L=path.resolve(__dirname,'..'),D=path.join(L,'reports/pick500_completion_preparation_20261104');fs.mkdirSync(D,{recursive:true});
for(const[k,v]of Object.entries(j))fs.writeFileSync(path.join(D,k+'.json'),JSON.stringify(v,null,2));
const title='## 2026-11-04：原500候选审计通过并顺序补样；最终库到PPO评估接续已排入';
const note=title+'\n\n- 原496候选已全部生成、独立逐文件/时间/目标绑定/GPU初始噪声复算496/496通过；与pilot4合并核ID0..499、噪声/特征全局去重500/500通过。原采样1402368与优化1405357已正常结束，不因PID消失而重启。顺序队列1424784已自动启动补样128（ID500..627），采样PID1425966、后优化PID1425968，目录candidate_replacement128_v1与postopt_replacement128_v1；registry已更新真实PID，native控制器1411803继续串行接纳。当前'+j.library_progress.native_admitted+'/500，只看实际library_progress；补样不是最终接纳。\n- 新stage3_pick500_finalize_library.py已准备CPU最终500构建/源噪声审计证明/每条状态与目标/FK/限位/原生证明及去重全量复算，恰选500个已接纳ID，保留多余与失败结果；目前未运行，须真正500及候选审计齐备才执行。输出final_library_v1/reference_bank.json和cpu_audit.json；CPU通过仍需默认ReferenceBank500和8912环境128步实测，不能仅改ready。\n- 新stage3_pick500_evaluate.py真实4环境32步读取2更新接线检查点测试通过，实际新物体XY随机位置、102实时Actor无参考、31动作、29DoF、策略/检查点不修改、零训练更新；仅评估入口验证，不是新正式策略成功率。默认正式100环境评估用新seed20261105、当前桌面+-0.06m连续新物体位置，参考仅初始化/无学习时奖励，接触目标从实际对象计算；主要成功指标仍height>0.95m且过接触时间，同时记录持续抬起时长和腕误差，非手指抓握认证。证据O/evaluation_entry4_v1/report.json和guard，勿重复。\n- 最终接续控制器PID1427653，O/completion_controller_v1、同名launch/.log及active_completion_controller.json，源码stage3_pick500_completion_controller.py SHA c7a3b1c758528a323a952090a7471461e8fe63b1f7570a46b22582be9478d9ef。实际状态'+j.completion_controller_status.status+'；它等待原生500+全628源审计，真实CPU最终库→默认500库8912env128步→结合实测生成final_library_audit.json→等待11GiB/实测检查点预算→另目录Pick3000 PPO→真实100env评估归档。不是已开始正式训练。六个入口/env/core源已绑定，不并发改源码或重复控制器；阶段失败留证，不重启旧队列。原生/补样/最终控制器均各司其职，不互相替代。\n- 正式运行若满足预算将用S/runs/pick500_8912_3000_v1和O/active_pick500_run.json，保留旧2参考pointer/模型。守本项目合计GPU12GiB/RSS32GiB、启动11GiB、运行8GiB+356MiB；不足启动预算会明确等待，绝不降门槛或删无备份检查点。实测500库/容量/正式训练/最终评估仍待完成，不把接续代码当结果。\n\n';
for(const name of ['PLAN.md','STATUS.md']){let p=path.join(L,name),s=fs.readFileSync(p,'utf8');if(!s.includes(title)){let i=s.indexOf('\n');fs.writeFileSync(p,s.slice(0,i+1)+'\n'+note+s.slice(i+1));}}
console.log(JSON.stringify({reference_count:j.library_progress.native_admitted,original496_audited:j.original496_audit_summary.candidates,
 replacement:j.replacement_status.status,completion:j.completion_controller_status.status,evaluation_entry:j.evaluation_entry4.status}));

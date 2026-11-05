const fs=require('fs'),path=require('path'),cp=require('child_process');
const L=path.resolve(__dirname,'..');
const py=`import json,hashlib
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103';O=S/'reference_scale500_20261104'
paths={'postopt_pilot':O/'postopt_pilot_jit_v1/report.json','native_pilot':O/'native_pilot_v1_launch/report.json',
 'object_reset':O/'object_reset_pilot_v1/report.json','native_reference_bank':O/'native_pilot_v1/reference_bank.json',
 'sampling_status':O/'candidate_remaining496_v1/status.json','postopt_status':O/'postopt_remaining488_jit_v3/status.json',
 'postopt_launch':O/'postopt_remaining488_jit_v3_launch.json'}
out={k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'data':json.loads(p.read_text())} for k,p in paths.items() if p.exists()}
assert out['native_pilot']['data']['admitted_pilot_references']==4
assert out['object_reset']['data']['maximum_object_reset_error_m']==0.
library=json.loads((O/'library_progress.json').read_text()) if (O/'library_progress.json').exists() else {'native_admitted':4}
plan=json.loads((O/'plan.json').read_text());plan['tasks'][0].update(new_admitted=library['native_admitted'],status='sampling_postoptimization_and_native_admission')
plan['current_processes']={'sampling_wrapper_pid':1402368,'postopt_wrapper_pid':1405357}
plan['pilot_results']={'native_references':4,'formal500_reference_count_satisfied':False,'object_reset_error_m':0.,
                      'warm_solver_seconds':out['postopt_pilot']['data']['solver_seconds'][1:]}
(O/'plan.json').write_text(json.dumps(plan,indent=2)+'\\n')
out['plan']={'path':str(O/'plan.json'),'data':plan,'sha256':hashlib.sha256((O/'plan.json').read_bytes()).hexdigest()}
print(json.dumps(out))
`;
const data=JSON.parse(cp.execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -'],{input:py,encoding:'utf8',maxBuffer:2e6}));
const dir=path.join(L,'reports/stage3_reference500_20261104');
for(const [k,v] of Object.entries(data))fs.writeFileSync(path.join(dir,k+'.json'),JSON.stringify(v.data,null,2)+'\n');
fs.writeFileSync(path.join(dir,'evidence_locations.json'),JSON.stringify(Object.fromEntries(Object.entries(data).map(([k,v])=>[k,{path:v.path,sha256:v.sha256}])),null,2)+'\n');
const n=`## 2026-11-04：四条新Pick参考已通过完整几何/场景链路，余496条边采样边后优化

- 当前新的500目标库已接纳4条（非500完成）；四条JAX/PyRoki后优化、微小限位投影后的独立NumPy FK及全碰撞网格地面检查、每条120帧真实PhysX几何回放均通过。新物体位置与参考目标一致，原生物体重置误差0；8环境32控制步四次重置覆盖4参考ID，Actor改变私有参考ID时观测差0。默认500条训练入口正确拒绝四条试库，未启动PPO/SuperMimic；几何回放不等于PD跟踪/抓取成功。
- 可复用JIT求解器stage3_pick500_postopt.py：首次46.46秒，后续2.44/2.64/2.84秒。新src/stage3_pick500_postopt_follow_v2.py持续读取完整采样记录，包装PID1405357，输出postopt_remaining496_jit_v2；先查收据/进程/日志/status/records，不重复。继续现有采样PID1402368，candidate_remaining496_v1；二者各限5%GPU，磁盘底线8GiB+456MiB。后优化库仍需原生回放再计接纳，不直接改标志。
- 新隔离stage3_pick500_env.py与stage3_pick500_replay.py绑定每参考物体位置，避免沿用旧固定物体使动作失配。最终库需500不同源噪声/特征及机器人状态内容、文件哈希、场景回放及可加载证明；旧资产/旧两参考PPO不修改。完整证据reports/stage3_reference500_20261104，服务器reference_scale500_20261104/native_pilot_v1_launch/report.json。
- 用户本轮仅询问联合PPO是否可行，未授权切成多任务统一Actor。当前仍每任务一套策略，每任务500参考一起参加该任务PPO（不是每条分别训），PPO通过仿真交互更新。若做统一8任务策略还需任务条件、多任务场景和奖励/采样机制，不能简单把4000条放进Pick环境。按Pick实测纯采样500约2小时、4000约16小时（单路当前5%GPU；不含其他7任务尚待实现的场景/条件、补采样及最终检查），不能把它当4000合格完工期限。

`;
for(const name of ['PLAN.md','STATUS.md']){const f=path.join(L,name);let x=fs.readFileSync(f,'utf8');if(!x.includes('## 2026-11-04：四条新Pick参考已通过')){const i=x.indexOf('\n');fs.writeFileSync(f,x.slice(0,i+1)+'\n'+n+x.slice(i+1));}}
console.log(JSON.stringify({admitted:4,postopt_pid:1405357,sampling:data.sampling_status?.data,postopt:data.postopt_status?.data}));

const fs=require('fs'),path=require('path'),{execFileSync}=require('child_process');
const script=String.raw`import json
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');O=R/'stage3_pick29_20261103/reference_scale500_20261104'
paths={'library_progress':O/'library_progress.json','replacement_queue_launch':O/'replacement_queue128_v1_launch.json',
 'replacement_queue_status':O/'replacement_queue128_v1/status.json','native_controller_status':O/'native_drain_controller_v1/status.json',
 'native_registry':O/'native_pipeline_registry.json'}
print(json.dumps({k:json.loads(v.read_text()) for k,v in paths.items()}))
`;
const j=JSON.parse(execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -'],{input:script,encoding:'utf8',timeout:30000}));
const L=path.resolve(__dirname,'..'),D=path.join(L,'reports/data_supported_four_20261104');fs.mkdirSync(D,{recursive:true});
for(const[k,v]of Object.entries(j))fs.writeFileSync(path.join(D,k+'.json'),JSON.stringify(v,null,2));
const note='- Pick顺序补采样队列已排入：PID1424784，O/replacement_queue128_v1及同名launch/.log，源码stage3_pick500_replacement_queue.py。当前'+j.replacement_queue_status.status+'；原496采样与488优化完成且实际退出后，独立核496文件/目标/时间/GPU噪声并与pilot4全500去重，再启动ID500..627共128个新seed候选和通用v3后优化，不并发重复旧模型。registry已原子登记两个新目录和完成路径，原生控制器将正常等待并接纳；新真实PID由队列启动后登记。此批按当前缺口补样，不是已完成500参考；如仍不足需新ID继续，保留失败前缀。数据8GiB+456MiB、合计GPU12GiB/RSS32GiB保护不降低，只能停止该队列新启动组。\n';
for(const name of ['PLAN.md','STATUS.md']){const p=path.join(L,name),s=fs.readFileSync(p,'utf8'),title='## 2026-11-04最新授权：近期改为四项数据支持任务，每项500，Pick优先';if(!s.includes('Pick顺序补采样队列已排入'))fs.writeFileSync(p,s.replace(title,title+'\n\n'+note));}
console.log(JSON.stringify({current_library:j.library_progress.native_admitted,target:500,replacement_queue:j.replacement_queue_status.status,formal_training_started:j.library_progress.formal_training_started}));

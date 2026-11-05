const f=require('fs'),p=__dirname+'/';
let s=f.readFileSync(p+'native_tracker500_capacity_worker.py','utf8');
const edits=[['choices=[\'capacity\',\'benchmark\']','choices=[\'resume\']'],["'10','--buffer_steps'","'12','--resume',str(W/'benchmark_8912_v1/logs/last.pt'),'--buffer_steps'"],["{'opt_wm':100,'opt_pi':9,'opt_v':180}","{'opt_wm':120,'opt_pi':11,'opt_v':220}"],['793168','1408096'],["stored['iteration']==10","stored['iteration']==12"],['native_benchmark_passed','native_resume_passed']];
for(const [a,b] of edits){if(!s.includes(a))throw Error(a);s=s.split(a).join(b)}
f.writeFileSync(p+'native_tracker500_resume_worker.py',s,{flag:'wx'});
let g=f.readFileSync(p+'native_tracker500_capacity_guard.py','utf8');
for(const [a,b] of [['choices=[\'capacity\',\'benchmark\']','choices=[\'resume\']'],["('capacity_','benchmark_')","('resume_',)"],['native_tracker500_capacity_worker.py','native_tracker500_resume_worker.py'],['native_benchmark_passed','native_resume_passed']]){if(!g.includes(a))throw Error(a);g=g.split(a).join(b)}
f.writeFileSync(p+'native_tracker500_resume_guard.py',g,{flag:'wx'});

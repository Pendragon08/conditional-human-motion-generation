'use strict';
// Explicit maintenance launcher; no GitHub network requests originate locally.
const cp=require('child_process');
const mode=process.argv[2],prefix='/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/publication_snapshot_20261105/';
let script,args=[];
if(mode==='check'){script='release_verified_pour_intermediates_20261105.py';}
else if(mode==='release'){
 if(!process.argv.includes('--execute'))throw Error('Explicit --execute required after parent review');
 const ix=process.argv.indexOf('--count'),count=Number(process.argv[ix+1]);if(ix<0||!Number.isInteger(count)||count<1||count>90)throw Error('Reviewed --count 1..90 required');
 script='release_verified_pour_intermediates_20261105.py';args=['--execute','--count',String(count)];
}else if(mode==='restore'){
 const ix=process.argv.indexOf('--iteration'),n=Number(process.argv[ix+1]);if(ix<0||!Number.isInteger(n)||n<0||n>4450||n%50)throw Error('Exact --iteration 0..4450 in increments of50 required');
 script='restore_pour_github_archive_20261105.py';args=['--source-path','/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/stage3_supported_references_20261104/task_ppo_20261104/pour/ppo_8912_5000_v1/ppo/model_'+n+'.pt'];
}else{console.log('Modes: check (no source deletion), release --execute --count N (requires parent review), restore --iteration N.');process.exit(0);}
let token=cp.execFileSync('gh',['auth','token'],{encoding:'utf8',stdio:['ignore','pipe','pipe'],windowsHide:true}).trim();
const child=cp.spawn('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=20','10.147.17.32','nice','-n','10','python3','-B','-u',prefix+script,...args],{stdio:['pipe','inherit','inherit'],windowsHide:true});
child.stdin.on('error',e=>console.error(e.code||'SSH stdin error'));child.on('error',e=>{console.error(e.code||'SSH error');process.exitCode=1;});child.stdin.end(JSON.stringify({token}));token='';child.on('close',code=>process.exitCode=code??1);

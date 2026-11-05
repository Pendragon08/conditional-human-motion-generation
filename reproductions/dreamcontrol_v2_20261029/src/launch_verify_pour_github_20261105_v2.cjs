'use strict';
const cp=require('child_process');
if(!process.argv.includes('--verify')){console.log('Pass --verify for read-only server-direct two-pass archive verification.');process.exit(0);}
let token=cp.execFileSync('gh',['auth','token'],{encoding:'utf8',stdio:['ignore','pipe','pipe'],windowsHide:true}).trim();
const child=cp.spawn('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=20','10.147.17.32','nice','-n','10','python3','-u','/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/publication_snapshot_20261105/verify_pour_github_archives_20261105_v2.py'],{stdio:['pipe','inherit','inherit'],windowsHide:true});
child.stdin.on('error',e=>console.error(e.code||'SSH stdin error'));child.on('error',e=>{console.error(e.code||'SSH launch error');process.exitCode=1;});child.stdin.end(JSON.stringify({token}));token='';child.on('close',code=>process.exitCode=code??1);

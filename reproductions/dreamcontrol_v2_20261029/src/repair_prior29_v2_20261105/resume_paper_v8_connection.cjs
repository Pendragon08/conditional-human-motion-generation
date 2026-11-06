// Restore the original SSH channel, then launch only the prepared, verified v8 chain.
const fs = require('node:fs');
const path = require('node:path');
const {spawnSync, spawn} = require('node:child_process');
const {createHash} = require('node:crypto');
const base = path.resolve(__dirname, '../..');
const out = path.join(base, 'reports/paper_v8_connection');
const remote = '/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/repair_prior29_v2_20261105';
const python = '/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python';
const launcher = fs.readFileSync(path.join(__dirname, 'launch_prepared_paper_v8.py'));
const hash = createHash('sha256').update(launcher).digest('hex');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const write = (name, value) => fs.writeFileSync(path.join(out, name), JSON.stringify(value, null, 2));
fs.mkdirSync(out, {recursive:true});
const lock = fs.openSync(path.join(out, 'supervisor_launch.json'), 'wx');
fs.writeFileSync(lock, JSON.stringify({pid:process.pid,source:__filename,launcher_sha256:hash}));fs.closeSync(lock);
const bootstrap = `import base64,hashlib,os,pathlib
p=pathlib.Path('${remote}/src/launch_prepared_paper_v8.py')
data=base64.b64decode('${launcher.toString('base64')}')
assert hashlib.sha256(data).hexdigest()=='${hash}'
if p.exists():assert p.read_bytes()==data
else:
 with p.open('xb') as f:f.write(data)
os.execv('${python}',['${python}','-B','-u',str(p)])
`;
async function main(){
 for(let attempt=1;attempt<=240;attempt++){
  write('status.json',{status:'connecting_prepared_launch',attempt,updated:new Date().toISOString()});
  const r=spawnSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ServerAliveInterval=5',
   '-o','ServerAliveCountMax=2','10.147.17.32','python3 -B -'],
   {input:bootstrap,encoding:'utf8',windowsHide:true,timeout:45000,maxBuffer:2*1024*1024});
  fs.appendFileSync(path.join(out,'attempts.log'),JSON.stringify({at:new Date().toISOString(),attempt,status:r.status,error:r.error?.message,stdout:r.stdout,stderr:r.stderr})+'\n');
  if(r.status===0){
   const receipt=JSON.parse(r.stdout);
   if(receipt.status!=='launched'||receipt.plan_sha256!=='eb3c79702a08589d7eb73a40605e1f0d1515254420bdfbd20bfbf9de72d2a0c2')throw Error('Unexpected launch receipt');
   write('remote_launch.json',receipt);
   const renderer=path.join(__dirname,'render_report_videos_paper_v8.cjs');
   const stdout=fs.openSync(path.join(base,'reports/paper_video_renderer_v8.log'),'wx');
   const stderr=fs.openSync(path.join(base,'reports/paper_video_renderer_v8.err'),'wx');
   const child=spawn(process.execPath,[renderer],{cwd:base,detached:true,windowsHide:true,stdio:['ignore',stdout,stderr]});
   child.unref();fs.closeSync(stdout);fs.closeSync(stderr);
   write('renderer_launch.json',{pid:child.pid,source:renderer});
   write('status.json',{status:'remote_chain_launched_renderer_waiting',processes:receipt.processes,renderer_pid:child.pid,at:new Date().toISOString()});
   return;
  }
  if(r.status!==255&&!r.error)throw Error('Prepared launcher reported a non-network failure; inspect, do not rerun blindly');
  write('status.json',{status:'waiting_original_ssh_reconnection',attempt,updated:new Date().toISOString()});
  await sleep(Math.min(attempt*15000,120000));
 }
 throw Error('Original SSH channel unavailable after bounded reconnect budget');
}
main().catch(error=>{write('status.json',{status:'failed_preserved',error:error.stack,at:new Date().toISOString()});process.exitCode=1;});

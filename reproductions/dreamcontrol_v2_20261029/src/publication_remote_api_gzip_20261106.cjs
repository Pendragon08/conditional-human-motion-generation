'use strict';
const {execFileSync}=require('node:child_process');
const {gunzipSync}=require('node:zlib');
const token=execFileSync('gh',['auth','token'],{encoding:'utf8',stdio:['ignore','pipe','pipe']}).trim();
const repo='Pendragon08/conditional-human-motion-generation';
function api(endpoint,method='GET',body){
 const request={token,endpoint:endpoint.startsWith('repos/')?endpoint:`repos/${repo}/${endpoint}`,method};
 if(body!==undefined)request.body=body;
 let last;
 for(let attempt=0;attempt<4;attempt++){
  try{
   const result=execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=20','10.147.17.32','python3','/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/publication_snapshot_20261105/relay_publication_api_gzip_20261106.py'],{input:JSON.stringify(request),maxBuffer:20*1024*1024,timeout:120000,stdio:['pipe','pipe','pipe'],windowsHide:true});
   const data=gunzipSync(result).toString('utf8');return data?JSON.parse(data):null;
  }catch(error){
   last=new Error(`Server relay ${method} ${endpoint}: ${error.code||error.message}; ${error.stderr?error.stderr.toString().slice(0,1000):''}`);
   if(method!=='GET')throw last;
  }
 }
 throw last;
}
module.exports={api};

// Only ZeroTier SSH carries local requests; GitHub network traffic originates on the server.
const {execFileSync}=require('node:child_process');
const token=execFileSync('gh',['auth','token'],{encoding:'utf8',stdio:['ignore','pipe','pipe']}).trim();
const repo='Pendragon08/conditional-human-motion-generation';
function api(endpoint,method='GET',body){
 const request={token,endpoint:endpoint.startsWith('repos/')?endpoint:`repos/${repo}/${endpoint}`,method};if(body!==undefined)request.body=body;
 let last;for(let attempt=0;attempt<4;attempt++){try{const s=execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=20','10.147.17.32','python3','/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/publication_snapshot_20261105/relay_publication_api_20261105.py'],{input:JSON.stringify(request),encoding:'utf8',maxBuffer:50*1024*1024,timeout:120000,stdio:['pipe','pipe','pipe']});return s?JSON.parse(s):null;}catch(e){last=e;if(method!=='GET')throw e;}}throw last;
}
module.exports={api};

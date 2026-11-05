const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),{execFileSync}=require('node:child_process');
const L=path.resolve(__dirname,'..'),stage='/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/repair_prior29_v2_20261105';
const entries=process.argv.slice(2).map(name=>{
 if(path.basename(name)!==name)throw Error('basename only');
 const b=fs.readFileSync(path.join(__dirname,'repair_prior29_v2_20261105',name));
 return{path:'src/'+name,data:b.toString('base64'),sha256:crypto.createHash('sha256').update(b).digest('hex')};
});if(!entries.length)throw Error('files required');
const script=`import sys,json,base64,hashlib,pathlib\np=json.load(sys.stdin);s=pathlib.Path(p['stage'])\nfor e in p['files']:\n b=base64.b64decode(e['data']);assert hashlib.sha256(b).hexdigest()==e['sha256'];t=s/e['path']\n if t.exists(): assert t.read_bytes()==b\n else:\n  with t.open('xb') as f:f.write(b)\nprint(json.dumps({'stage':str(s),'files':[{k:v for k,v in e.items() if k!='data'} for e in p['files']]}))\n`;
const cmd="python3 -c 'import base64;exec(base64.b64decode(\""+Buffer.from(script).toString('base64')+"\"))'";
const result=execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32',cmd],{input:JSON.stringify({stage,files:entries}),encoding:'utf8',timeout:120000,maxBuffer:2**20});
const dir=path.join(L,'reports','repair_prior29_v2_20261105');fs.mkdirSync(dir,{recursive:true});fs.writeFileSync(path.join(dir,'addition_'+Date.now()+'.json'),result,{flag:'wx'});console.log(result);

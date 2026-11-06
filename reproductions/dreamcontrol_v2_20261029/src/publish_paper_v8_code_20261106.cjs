// Incremental code/evidence publication. All GitHub requests use the server relay.
'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const {api}=require('./publication_remote_api_20261105.cjs');
const root=path.resolve(__dirname,'..'),prefix='reproductions/dreamcontrol_v2_20261029/';
const out=path.join(root,'reports/publication_inventory_20261105');
const receipt=path.join(out,'paper_v8_code_20261106_receipt.json');
const blob=b=>crypto.createHash('sha1').update(Buffer.from('blob '+b.length+'\0')).update(b).digest('hex');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
function walk(dir){return fs.existsSync(dir)?fs.readdirSync(dir,{withFileTypes:true}).flatMap(x=>x.isDirectory()?(x.name==='__pycache__'?[]:walk(path.join(dir,x.name))):[path.join(dir,x.name)]):[];}
const files=walk(path.join(root,'src/repair_prior29_v2_20261105/paper_protocol_v8')).filter(x=>/\.py$/.test(x));
for(const name of ['verified_initialization_v8.py','deploy_paper_seed_resume_v8.py','launch_prepared_paper_v8.py',
 'resume_paper_v8_connection.cjs','render_report_videos_paper_v8.cjs','evaluation_controller_paper_v8.py',
 'supermimic_repaired_controller_paper_v8.py','report_video_sources_paper_v8.py',
 'diagnose_paper_pick_seed_20261106_v1.py','diagnose_paper_pick_seed_20261106_v2.py'])
 files.push(path.join(root,'src/repair_prior29_v2_20261105',name));
for(const name of ['PLAN.md','STATUS.md','reports/PAPER_PROTOCOL_INITIALIZATION_V8_20261106.md',
 'src/publish_paper_v8_code_20261106.cjs','src/prepare_v8_code_publication_20261106.cjs'])files.push(path.join(root,name));
files.push(...walk(path.join(root,'reports/paper_pipeline_v8_launch')).filter(x=>/\.json$/.test(x)));
const entries=[],manifest=[];
for(const file of [...new Set(files)].sort()){
 const bytes=fs.readFileSync(file),content=bytes.toString('utf8'),rel=path.relative(root,file).split(path.sep).join('/');
 if(!Buffer.from(content).equals(bytes)||/(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|\bAKIA[0-9A-Z]{16}\b)/.test(content))throw Error('Publication hold: '+rel);
 entries.push({path:prefix+rel,mode:'100644',type:'blob',content});
 manifest.push({path:rel,bytes:bytes.length,sha256:sha(bytes),blob_sha:blob(bytes)});
}
const manifestContent=JSON.stringify({status:'paper_v8_initialization_audited_pilots_in_progress',files:manifest,
 scope:'Incremental code and recorded evidence only. New 2000-reference generation, quality evaluation, SuperMimic and object evaluation remain pending their receipts. Old 275 project weights and cold logs were previously uploaded; repaired final model asset715377849 has passed full TAR/member readback. One new Pick reference has passed; other pilots, full-library quality, new tracking training and dynamic evaluation remain pending.'},null,2)+'\n';
entries.push({path:prefix+'PAPER_V8_SOURCE_MANIFEST_20261106.json',mode:'100644',type:'blob',content:manifestContent});
const signature=sha(Buffer.from(JSON.stringify(entries)));
if(fs.existsSync(receipt)){
 const old=JSON.parse(fs.readFileSync(receipt));
 if(old.signature===signature&&old.status==='published_verified'){console.log(JSON.stringify(old));process.exit(0);}
 throw Error('Existing different snapshot preserved; choose a new publication receipt version.');
}
const head=api('git/ref/heads/main').object.sha,base=api('git/commits/'+head),before=api('git/trees/'+base.tree.sha+'?recursive=1');
if(before.truncated)throw Error('Truncated base tree');
let tree=base.tree.sha;
if(!process.argv.includes('--publish')){console.log(JSON.stringify({status:'reviewed',files:manifest.length,bytes:manifest.reduce((a,r)=>a+r.bytes,0),head,signature}));process.exit(0);}
for(let i=0;i<entries.length;i+=60)tree=api('git/trees','POST',{base_tree:tree,tree:entries.slice(i,i+60)}).sha;
if(api('git/ref/heads/main').object.sha!==head)throw Error('Branch changed; preserve staged content, do not overwrite');
const commit=api('git/commits','POST',{message:'Align reference optimizer initialization and resume verified paper-protocol pilots',tree,parents:[head]});
api('git/refs/heads/main','PATCH',{sha:commit.sha,force:false});
const actual=api('git/trees/'+tree+'?recursive=1');if(actual.truncated)throw Error('Truncated published tree');
const byPath=new Map(actual.tree.map(x=>[x.path,x])),changed=new Set(entries.map(x=>x.path));
for(const e of entries)if(byPath.get(e.path)?.sha!==blob(Buffer.from(e.content)))throw Error('Blob mismatch '+e.path);
for(const e of before.tree)if(e.type!=='tree'&&!changed.has(e.path)&&byPath.get(e.path)?.sha!==e.sha)throw Error('Unrelated file changed '+e.path);
if(api('git/ref/heads/main').object.sha!==commit.sha)throw Error('Head changed after publication; inspect');
const result={status:'published_verified',signature,commit:commit.sha,url:commit.html_url,files:entries.length,source_bytes:manifest.reduce((a,r)=>a+r.bytes,0),transport:'GitHub HTTPS from server; ZeroTier relay only locally',verified_utc:new Date().toISOString()};
fs.writeFileSync(receipt,JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));

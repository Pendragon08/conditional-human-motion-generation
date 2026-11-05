// Preserve the existing repository; add only the reviewed reproduction snapshot.
// GitHub HTTPS is performed by the server helper, never by local gh api.
'use strict';
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),zlib=require('node:zlib');
const root=path.resolve(__dirname,'..'),out=path.join(root,'reports/publication_inventory_20261105');
const repo='Pendragon08/conditional-human-motion-generation',prefix='reproductions/dreamcontrol_v2_20261029';
const api=require('./publication_remote_api_20261105.cjs').api;
const exclusions=['src/repair_prior29_v2_20261105/','tests/test_repair_prior29'];
function walk(dir){if(!fs.existsSync(dir))return [];return fs.readdirSync(dir,{withFileTypes:true}).flatMap(d=>d.isDirectory()?(['__pycache__','.git','node_modules'].includes(d.name)?[]:walk(path.join(dir,d.name))):d.isFile()?[path.join(dir,d.name)]:[]);}
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const blobSha=b=>crypto.createHash('sha1').update(Buffer.from('blob '+b.length+'\0')).update(b).digest('hex');
const rel=p=>path.relative(root,p).split(path.sep).join('/');
const excluded=p=>exclusions.some(x=>rel(p).startsWith(x));
const files=[...walk(path.join(root,'src')).filter(p=>/\.(py|js|cjs|mjs|ps1|sh|json|yaml|yml|toml|txt|md)$/.test(p)),...walk(path.join(root,'tests')).filter(p=>/\.(py|js|cjs|json|md)$/.test(p)),...walk(path.join(root,'reports')).filter(p=>p.endsWith('.md')),...['README.md','PLAN.md','STATUS.md','LICENSE.upstream-DreamControl','LICENSE.upstream-SuperMimic','NOTICE.third-party.md'].map(p=>path.join(root,p))];
for(const name of ['generated_reference_evaluation_20261105','native2000_object_task_evaluation_20261105','four_task2000_final_20261105','diffusion_pipeline_audit_20261105','reference_videos_upstream_20261105'])
 for(const p of walk(path.join(root,'reports',name)))if(/\.(json|diff|py|txt)$/.test(p)&&fs.statSync(p).size<2*1024*1024)files.push(p);
const rows=[],entryMap=new Map();
function add(publishedPath,bytes,sourcePath){
 const content=bytes.toString('utf8');if(!bytes.equals(Buffer.from(content)))throw Error('Non-UTF8 source: '+sourcePath);
 if(/(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|\bAKIA[0-9A-Z]{16}\b)/.test(content))throw Error('Credential-like content held: '+sourcePath);
 if(entryMap.has(publishedPath))throw Error('Duplicate publication path: '+publishedPath);
 rows.push({path:publishedPath,source_path:sourcePath,bytes:bytes.length,sha256:sha(bytes),git_blob_sha:blobSha(bytes)});
 entryMap.set(publishedPath,{path:prefix+'/'+publishedPath,mode:'100644',type:'blob',content});
}
for(const file of [...new Set(files)].filter(p=>!excluded(p)).sort())add(rel(file),fs.readFileSync(file),rel(file));
const artifactPaths=['archived-weights-v3-receipt.json','completed-logs-review-v3-receipt.json','local_results_upload_receipt.json','remote_upload_receipt.json','pour_addendum_upload_receipt.json','supplement_upload_receipt.json','publication_summary.json','publication_holds.json','ARTIFACT_CATALOG.md','local_extra_assets.json','license_addendum.json','supplement_preflight_receipt.json','supplement_preflight_v2_receipt.json','REMOTE_PUBLICATION_COVERAGE.md'];
for(const name of artifactPaths){const file=path.join(out,name);if(!fs.existsSync(file))throw Error('Required publication evidence missing: '+name);add('publication/'+name,fs.readFileSync(file),rel(file));}
add('publication/checkpoint_catalog.json',zlib.gunzipSync(fs.readFileSync(path.join(out,'checkpoint_catalog.json.gz'))),'reports/publication_inventory_20261105/checkpoint_catalog.json.gz (decompressed)');
const coldManifest=path.join(root,'training_space_archive_20261105/completed_logs_and_review_png/v1/manifest.json');
add('publication/completed_logs_and_review_png_manifest.json',fs.readFileSync(coldManifest),rel(coldManifest));
const manifest={schema_version:2,captured_utc:new Date().toISOString(),files:rows,excluded_development_prefixes:exclusions,exclusion_reason:'New repair code is still under separate validation; no formal repaired-model training is claimed by this snapshot.',publication_state:'Draft assets remain incomplete: 137/275 checkpoints recoverable; 138 local archived checkpoints pending.'};
const entries=[...entryMap.values(),{path:prefix+'/SOURCE_MANIFEST.json',mode:'100644',type:'blob',content:JSON.stringify(manifest,null,2)+'\n'},{path:prefix+'/.gitattributes',mode:'100644',type:'blob',content:'* -text\n'}];
const head=api('git/ref/heads/main').object.sha,base=api('git/commits/'+head);let tree=base.tree.sha;
let readme=Buffer.from(api('contents/README.md?ref='+head).content,'base64').toString('utf8');
const section=['<!-- dreamcontrol-v2-20261105:start -->','## DreamControl-v2 29DoF reproduction (2026-11-05)','',
'[Code, results and limitations]('+prefix+'/README.md) · [Source manifest]('+prefix+'/SOURCE_MANIFEST.json) · [Artifact coverage]('+prefix+'/publication/ARTIFACT_CATALOG.md)','',
'Four 500-reference task banks and one shared SuperMimic body tracker are complete. Object-task evaluations are 0/100 under the recorded protocols. Pick PPO (3,000 updates) and Pour PPO (5,000 updates) are complete and each scored 0/100; later skill PPO jobs were cancelled before formal training. Four audit repairs are being prepared and have not begun a new formal training run in this snapshot. The draft release has 137/275 checkpoints recoverable from complete verified groups; the other 138 archived checkpoints are not fully uploaded. This adapted baseline does not reproduce the paper performance.',
'<!-- dreamcontrol-v2-20261105:end -->','',''].join('\n');
readme=readme.replace(/<!-- dreamcontrol-v2-20261105:start -->[\s\S]*?<!-- dreamcontrol-v2-20261105:end -->\s*/g,'');
const firstBreak=readme.indexOf('\n');readme=readme.slice(0,firstBreak+1)+'\n'+section+readme.slice(firstBreak+1);
entries.push({path:'README.md',mode:'100644',type:'blob',content:readme});
if(process.argv.includes('--review-only')){console.log(JSON.stringify({status:'utf8_secret_pattern_license_scope_review_passed',files:rows.length,source_bytes:rows.reduce((s,r)=>s+r.bytes,0),excluded_development_prefixes:exclusions,head,required_evidence:artifactPaths}));process.exit(0);}
const progressPath=path.join(out,'git_tree_progress.json');
const snapshotKey=sha(Buffer.from(JSON.stringify({head,rows,readme})));
let staged=0;
if(fs.existsSync(progressPath)){
 const previous=JSON.parse(fs.readFileSync(progressPath,'utf8'));
 if(previous.snapshotKey===snapshotKey&&previous.head===head&&previous.staged>0&&previous.staged<=entries.length&&previous.capturedUtc){
  tree=api('git/trees/'+previous.tree).sha;
  staged=previous.staged;
  manifest.captured_utc=previous.capturedUtc;
  entries.find(entry=>entry.path===prefix+'/SOURCE_MANIFEST.json').content=JSON.stringify(manifest,null,2)+'\n';
 }
}
for(let i=staged;i<entries.length;i+=100){
 const request={base_tree:tree,tree:entries.slice(i,i+100)};
 for(let attempt=0;;attempt++){
  try{tree=api('git/trees','POST',request).sha;break;}
  catch(error){
   // An identical request creates the same content-addressed Git tree.
   if(attempt>=3||error.status!==255)throw error;
   console.log(JSON.stringify({status:'retrying_git_tree_after_ssh_disconnect',offset:i,attempt:attempt+1}));
  }
 }
 const done=Math.min(i+100,entries.length);
 fs.writeFileSync(progressPath+'.tmp',JSON.stringify({head,snapshotKey,staged:done,tree,capturedUtc:manifest.captured_utc})+'\n');
 fs.renameSync(progressPath+'.tmp',progressPath);
 console.log(JSON.stringify({staged:done,total:entries.length}));
}
const state={head,tree,files:rows.length,entries:entries.length,source_bytes:rows.reduce((s,r)=>s+r.bytes,0),captured_utc:new Date().toISOString(),excluded_development_prefixes:exclusions,source_manifest_sha256:sha(Buffer.from(entries.find(e=>e.path===prefix+'/SOURCE_MANIFEST.json').content))};
fs.writeFileSync(path.join(out,'git_staging_receipt.json'),JSON.stringify(state,null,2)+'\n');
if(process.argv.includes('--publish')){
 if(api('git/ref/heads/main').object.sha!==head)throw Error('Branch changed during staging; restage against current head');
 const commit=api('git/commits','POST',{message:'Archive DreamControl-v2 reproduction code, audit evidence and completed Pour results',tree,parents:[head]});
 api('git/refs/heads/main','PATCH',{sha:commit.sha,force:false});
 if(api('git/ref/heads/main').object.sha!==commit.sha)throw Error('Published head verification failed');
 const resultTree=api('git/trees/'+tree+'?recursive=1');if(resultTree.truncated)throw Error('Tree verification truncated');const byPath=new Map(resultTree.tree.map(e=>[e.path,e]));
 for(const e of entries)if(byPath.get(e.path)?.sha!==blobSha(Buffer.from(e.content)))throw Error('Published blob verification failed: '+e.path);
 const oldTree=api('git/trees/'+base.tree.sha+'?recursive=1');if(oldTree.truncated)throw Error('Existing repository verification truncated');let preserved=0;
 for(const e of oldTree.tree)if(e.type!=='tree'&&e.path!=='README.md'&&!e.path.startsWith(prefix+'/')){if(byPath.get(e.path)?.sha!==e.sha)throw Error('Existing repository blob changed: '+e.path);preserved++;}
 Object.assign(state,{status:'published_all_snapshot_blobs_verified',commit:commit.sha,url:commit.html_url,verified_blob_count:entries.length,existing_outside_snapshot_blobs_preserved:preserved});
 fs.writeFileSync(path.join(out,'git_publication_receipt.json'),JSON.stringify(state,null,2)+'\n');console.log(JSON.stringify(state));
}

// Read-only comparison of fixed upstream listings and captions with the GRAB source registry.
// Exact names and source hashes are evidence of identity, not semantic or physical validation.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const root = path.resolve(__dirname, '..');
const dir = path.join(root, 'reports', 'grab_dex_motionhub_20261103');
fs.mkdirSync(dir); // Fail rather than overwrite any existing result.
const src = path.join(root, 'reports', 'grab_source_registry_20261030.jsonl');
const data = fs.readFileSync(src);
const sha256 = b => crypto.createHash('sha256').update(b).digest('hex');
if (sha256(data) !== '27d74b72fec830afa01cc05f788d921ef3d42fc1115aad08565d26604cdec8e4') throw Error('source registry hash changed');
const rows = data.toString('utf8').trim().split(/\r?\n/).map(JSON.parse);
const ids = new Map(rows.map(x => [x.member.replace(/\.npz$/, ''), x]));
if (ids.size !== rows.length) throw Error('duplicate source identities');
const dRev = '264cde93a2b3f502ded55fe4e5c05234a2bdefa2';
const mRev = 'c3f6c8eb8a4ba9e5ca521cdc0af9264756b66726';
async function get(url) {
  let last;
  for (let i=0;i<4;i++) {
    try { const r=await fetch(url,{signal:AbortSignal.timeout(30000)}); if (!r.ok) throw Error(`HTTP ${r.status}`); return Buffer.from(await r.arrayBuffer()); }
    catch(e) { last=e; await new Promise(resolve=>setTimeout(resolve,500*(i+1))); }
  }
  throw Error(`${url}: ${last}`);
}
async function main() {
  const dex = new Map(), text = new Map(), subjectCounts=[];
  for (let n=1;n<=10;n++) {
    const s=`s${n}`;
    const [d,m]=await Promise.all([
      get(`https://huggingface.co/api/datasets/naichuan04/DexWeave/tree/${dRev}/data/GRAB/${s}?limit=1000`).then(JSON.parse),
      get(`https://huggingface.co/api/datasets/ZeyuLing/MotionHub/tree/${mRev}/GRAB/hierarchical_caption/${s}?limit=1000`).then(JSON.parse)
    ]);
    for (const x of d.filter(x=>x.type==='directory')) dex.set(`${s}/${path.posix.basename(x.path)}`,x.path);
    for (const x of m.filter(x=>x.type==='file' && x.path.endsWith('.json'))) text.set(`${s}/${path.posix.basename(x.path,'.json')}`,x);
    subjectCounts.push({subject:s,source:rows.filter(x=>x.subject===s).length,dex:d.filter(x=>x.type==='directory').length,text:m.filter(x=>x.type==='file'&&x.path.endsWith('.json')).length});
  }
  const missing = (a,b)=>[...a.keys()].filter(k=>!b.has(k)).sort();
  const overlap = [...ids.keys()].filter(k=>dex.has(k)&&text.has(k)).sort();
  const manifest = {source_count:ids.size,dex_directory_count:dex.size,text_file_count:text.size,matched_names:overlap.length,source_missing_dex:missing(ids,dex),source_missing_text:missing(ids,text),dex_missing_source:missing(dex,ids),text_missing_source:missing(text,ids),subject_counts:subjectCounts,source_registry_sha256:sha256(data),dex_revision:dRev,motionhub_revision:mRev,limitations:['Exact names are not proof of time, motion, text semantics or individual source hashes.','Dex entries are sequence directories; each motion file still needs its own integrity check.']};
  fs.writeFileSync(path.join(dir,'inventory.json'),JSON.stringify(manifest,null,2),{flag:'wx'});
  let cursor=0; const results=new Array(overlap.length);
  async function worker(){ while(true){const i=cursor++; if(i>=overlap.length)return; const id=overlap[i],entry=text.get(id); try {
    const buf=await get(`https://huggingface.co/datasets/ZeyuLing/MotionHub/resolve/${mRev}/${entry.path}`);
    const blob=crypto.createHash('sha1').update(Buffer.from(`blob ${buf.length}\0`)).update(buf).digest('hex');
    if(blob!==entry.oid) throw Error('git blob hash mismatch');
    const value=JSON.parse(buf.toString('utf8')); const captions=['macro','meso','micro'].flatMap(k=>Array.isArray(value[k])?value[k]:[]);
    results[i]={id,source_sha256:ids.get(id).member_sha256,text_sha256:sha256(buf),text_git_blob:blob,annotation:value,caption_count:captions.length};
  }catch(e){results[i]={id,error:String(e)}} }}
  await Promise.all(Array.from({length:8},worker));
  fs.writeFileSync(path.join(dir,'captions.jsonl'),results.map(x=>JSON.stringify(x)).join('\n')+'\n',{flag:'wx'});
  const ok=results.filter(x=>!x.error), errors=results.filter(x=>x.error);
  const summary={...manifest,captions_fetched:ok.length,errors:errors.length,error_examples:errors.slice(0,8),captions_total:ok.reduce((n,x)=>n+x.caption_count,0),empty_caption_records:ok.filter(x=>!x.caption_count).length,limitations:[...manifest.limitations,'Captions are text of the source human motions; same name does not prove description matches robot motion.','No visual review or real timestamp verification has been performed.']};
  fs.writeFileSync(path.join(dir,'summary.json'),JSON.stringify(summary,null,2),{flag:'wx'});
  console.log(JSON.stringify({matched_names:manifest.matched_names,source_count:manifest.source_count,dex_count:manifest.dex_directory_count,text_count:manifest.text_file_count,caption_records:ok.length,caption_total:summary.captions_total,errors:errors.length,source_missing_dex:manifest.source_missing_dex.length,source_missing_text:manifest.source_missing_text.length}));
}
main().catch(e=>{console.error(e);process.exitCode=1});

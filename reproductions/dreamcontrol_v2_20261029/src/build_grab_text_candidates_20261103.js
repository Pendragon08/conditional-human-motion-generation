// Assemble traceable text–robot candidate identities. This does not accept semantic quality.
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const root=path.resolve(__dirname,'..'),dir=path.join(root,'reports','grab_dex_motionhub_20261103');
const read=x=>fs.readFileSync(path.join(dir,x));
const inv=JSON.parse(read('inventory.json')),cmp=JSON.parse(read('server_registry_comparison.json'));
const data=read('captions.jsonl');const rows=data.toString('utf8').trim().split(/\r?\n/).map(JSON.parse);
if(inv.matched_names!==1335||cmp.server_members!==1335||cmp.server_not_in_registry.length||cmp.registry_not_in_server.length||cmp.uncompressed_size_mismatch.length||rows.length!==1335||rows.some(x=>x.error||!x.annotation))throw Error('incomplete or inconsistent inventory');
const output=rows.map(x=>{
  const offhand=/_offhand(?:_|$)/i.test(x.id), caption=[x.annotation.action,...(x.annotation.macro||[])].join(' ');
  const risks=[];if(offhand)risks.push('offhand_semantics_needs_review');
  if(offhand&&/turn.{0,8}off|switch.{0,8}off|deactivat/i.test(caption))risks.push('possible_offhand_interpreted_as_switch_off');
  if(offhand&&/toss|throw/i.test(caption))risks.push('possible_throw_interpretation');
  return {id:x.id,source:'GRAB',source_member:`${x.id}.npz`,source_member_sha256:x.source_sha256,
    robot_repository:'naichuan04/DexWeave',robot_revision:inv.dex_revision,robot_relative_path:`data/GRAB/${x.id}/${path.posix.basename(x.id)}_opt.npz`,robot_file_verified:false,
    text_repository:'ZeyuLing/MotionHub',text_revision:inv.motionhub_revision,text_relative_path:`GRAB/hierarchical_caption/${x.id}.json`,text_sha256:x.text_sha256,captions:x.annotation.macro,
    source_to_robot_time_checked:false,semantic_review:'not_reviewed',risk_codes:risks,training_accepted:false};
});
const bytes=Buffer.from(output.map(JSON.stringify).join('\n')+'\n');
fs.writeFileSync(path.join(dir,'candidate_pairs.jsonl'),bytes,{flag:'wx'});
const report={rows:output.length,captions:output.reduce((n,x)=>n+x.captions.length,0),offhand_review:output.filter(x=>x.risk_codes.includes('offhand_semantics_needs_review')).length,possible_switch_off:output.filter(x=>x.risk_codes.includes('possible_offhand_interpreted_as_switch_off')).map(x=>x.id),possible_throw:output.filter(x=>x.risk_codes.includes('possible_throw_interpretation')).map(x=>x.id),training_accepted:0,candidate_sha256:crypto.createHash('sha256').update(bytes).digest('hex'),limitations:['Flags are text heuristics; absence of a flag does not indicate semantic correctness.','No full robot download, identity validation, time alignment, or visual review yet.','No GRAB physical action filtering.']};
fs.writeFileSync(path.join(dir,'candidate_summary.json'),JSON.stringify(report,null,2),{flag:'wx'});console.log(JSON.stringify(report));

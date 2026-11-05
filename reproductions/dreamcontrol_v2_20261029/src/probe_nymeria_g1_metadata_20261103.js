// Read-only metadata/range probe: no Nymeria full trajectory download.
const fs=require('fs'),path=require('path'),{pathToFileURL}=require('url');
const root=path.resolve(__dirname,'..');
const hf='https://huggingface.co';
const rev='fad7f1ff08708761d5d9b7cac28755f302516298';
const repo='sehoonha/nymeria_custom';
const g1=['left_hip_pitch','left_hip_roll','left_hip_yaw','left_knee','left_ankle_pitch','left_ankle_roll','right_hip_pitch','right_hip_roll','right_hip_yaw','right_knee','right_ankle_pitch','right_ankle_roll','waist_yaw','waist_roll','waist_pitch','left_shoulder_pitch','left_shoulder_roll','left_shoulder_yaw','left_elbow','left_wrist_roll','left_wrist_pitch','left_wrist_yaw','right_shoulder_pitch','right_shoulder_roll','right_shoulder_yaw','right_elbow','right_wrist_roll','right_wrist_pitch','right_wrist_yaw'].map(n=>n+'_joint');
async function fetchOk(u){const r=await fetch(u,{signal:AbortSignal.timeout(30000)});if(!r.ok)throw Error(`HTTP ${r.status}`);return r}
async function main(){
 const h=await import(pathToFileURL(path.join(root,'data/nymeria_metadata_probe_20261103/node_modules/hyparquet/src/index.js')).href);
 const info=await(await fetchOk(`${hf}/api/datasets/${repo}?blobs=true`)).json();if(info.sha!==rev)throw Error('dataset revision changed');
 const files=info.siblings.filter(x=>x.rfilename.endsWith('_smplh_retargeted_dex_mj_fps30.parquet'));
 const all=new Set(info.siblings.map(x=>x.rfilename));
 const atom=files.filter(x=>all.has(x.rfilename.split('/')[0]+'/narration/atomic_action.csv')).length;
 const activity=files.filter(x=>all.has(x.rfilename.split('/')[0]+'/narration/activity_summarization.csv')).length;
 let samples=[];
 for(const x of [files[0],files[Math.floor(files.length/2)],files.at(-1)]){
  const u=`${hf}/datasets/${repo}/resolve/${rev}/${x.rfilename}`;const f=await h.asyncBufferFromUrl({url:u,byteLength:x.size});const m=await h.parquetMetadataAsync(f);const kv=Object.fromEntries(m.key_value_metadata.map(z=>[z.key,z.value]));const j=JSON.parse(kv.joint_names);
  const stamp=g=>g.columns.filter(z=>z.meta_data.path_in_schema[0]==='timestamp').map(z=>({min:z.meta_data.statistics.min_value,max:z.meta_data.statistics.max_value}));
  samples.push({path:x.rfilename,bytes:x.size,parquet_rows:Number(m.num_rows),fps:kv.fps,joint_columns:j.length,g1_29_present:g1.filter(n=>j.includes(n)).length,g1_missing:g1.filter(n=>!j.includes(n)),timestamp_first_rowgroup:stamp(m.row_groups[0]),timestamp_last_rowgroup:stamp(m.row_groups.at(-1)),columns:h.parquetSchema(m).children.map(z=>z.element.name)});
 }
 const folder=files[0].rfilename.split('/')[0];const csv=await(await fetchOk(`${hf}/datasets/${repo}/resolve/${rev}/${folder}/narration/atomic_action.csv`)).text();const first=csv.split(/\r?\n/).filter(Boolean).slice(0,2);
 const result={repository:repo,revision:rev,gated:info.gated,trajectory_file_count:files.length,trajectory_bytes:files.reduce((n,x)=>n+x.size,0),atomic_text_in_same_sequence_directory:atom,activity_text_in_same_sequence_directory:activity,inspected_parquet_footer_samples:samples,example_atomic_csv_header:first[0],example_first_annotation_start_seconds:Number(first[1].split(',')[2]),limitations:['Third party contains no dataset card or verified robot URDF provenance.','Only three parquet footers inspected, not all sequences or full data.','Caption CSV source timestamps and robot-relative timestamps are not aligned or semantically validated.','No full robot motion, object geometry, retargeting quality, true clock or usage license validated.']};
 const dest=path.join(root,'reports/nymeria_g1_public_probe_20261103.json');fs.writeFileSync(dest,JSON.stringify(result,null,2),{flag:'wx'});
 console.log(JSON.stringify({trajectories:files.length,bytes:result.trajectory_bytes,atomic:atom,activity,samples:samples.map(x=>({fps:x.fps,g1_29_present:x.g1_29_present,rows:x.parquet_rows,first:x.timestamp_first_rowgroup,last:x.timestamp_last_rowgroup}))}));
}
main().catch(e=>{console.error(e);process.exitCode=1});

const fs=require('fs'),path=require('path');
const base=path.join(__dirname,'paper_protocol_v7');
function edit(name,pairs){let s=fs.readFileSync(path.join(base,name),'utf8');for(const [a,b] of pairs){if(!s.includes(a))throw Error(name+' missing '+a);s=s.split(a).join(b);}fs.writeFileSync(path.join(base,name),s);}
edit('evaluation_controller_v7.py',[
 ['new_references_v6','new_references_paper_v7'],['evaluation_controller_v6','evaluation_controller_paper_v7'],
 ['evaluation_repaired_v1','evaluation_paper_v7'],['caption_tokens_repaired_v1','caption_tokens_paper_v7']]);
edit('supermimic_repaired_controller_v7.py',[
 ['new_references_v6','new_references_paper_v7'],['supermimic_repaired_v6','supermimic_paper_v7'],
 ['evaluation_repaired_v1','evaluation_paper_v7'],['evaluation_controller_v6','evaluation_controller_paper_v7']]);
edit('report_video_sources_v4.py',[
 ['new_references_v6','new_references_paper_v7'],['supermimic_repaired_v6','supermimic_paper_v7'],
 ['report_video_sources_v3','report_video_sources_paper_v7']]);
let t=fs.readFileSync(path.join(__dirname,'../tokenize_repaired_pick_20261105_v2.py'),'utf8');
t=t.replace("legacy = ROOT / 'generated_reference_evaluation_20261105/caption_tokens_v1.json'","legacy = STAGE / 'reports/caption_tokens_repaired_v1.json'")
 .replaceAll('caption_tokens_repaired_v1.json\'\n    receipt','caption_tokens_paper_v7.json\'\n    receipt')
 .replace("receipt = STAGE / 'reports/caption_tokens_repaired_v1_receipt.json'","receipt = STAGE / 'reports/caption_tokens_paper_v7_receipt.json'")
 .replace("caption='a person picks an object with their right hand'","caption='a person picks an object with the right hand and performs a pouring action with the same hand'")
 .replace("print('Official wheel received', hashlib.sha256(body).hexdigest(), flush=True)","assert hashlib.sha256(body).hexdigest() == '1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85'\n        print('Official wheel verified', hashlib.sha256(body).hexdigest(), flush=True)");
fs.writeFileSync(path.join(base,'tokenize_paper_pour_v7.py'),t);
let v=fs.readFileSync(path.join(__dirname,'render_report_videos_v3.cjs'),'utf8')
 .replaceAll('report_video_sources_v3','report_video_sources_paper_v7')
 .replace('D:/DreamControl_v2_repaired_report_v1','D:/DreamControl_v2_paper_protocol_report_v7')
 .replace('18103, debugPort = 18104','18107, debugPort = 18108');
fs.writeFileSync(path.join(__dirname,'render_report_videos_paper_v7.cjs'),v);
console.log('Prepared independent follow-ups; no jobs started.');

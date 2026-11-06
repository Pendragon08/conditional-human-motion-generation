const fs = require('node:fs');
const path = require('node:path');
let body = fs.readFileSync(path.join(__dirname, 'publish_repair_code_20261106.cjs'), 'utf8').replace(/\r\n/g, '\n');
const start = body.indexOf("const files=walk(");
const end = body.indexOf('const entries=[],manifest=[];', start);
if (start < 0 || end < start) throw Error('Publication template markers changed');
body = body.slice(0,start) + `const files=walk(path.join(root,'src/repair_prior29_v2_20261105/paper_protocol_v8')).filter(x=>/\\.py$/.test(x));
for(const name of ['verified_initialization_v8.py','deploy_paper_seed_resume_v8.py','launch_prepared_paper_v8.py',
 'resume_paper_v8_connection.cjs','render_report_videos_paper_v8.cjs','evaluation_controller_paper_v8.py',
 'supermimic_repaired_controller_paper_v8.py','report_video_sources_paper_v8.py',
 'diagnose_paper_pick_seed_20261106_v1.py','diagnose_paper_pick_seed_20261106_v2.py'])
 files.push(path.join(root,'src/repair_prior29_v2_20261105',name));
for(const name of ['PLAN.md','STATUS.md','reports/PAPER_PROTOCOL_INITIALIZATION_V8_20261106.md',
 'src/publish_paper_v8_code_20261106.cjs','src/prepare_v8_code_publication_20261106.cjs'])files.push(path.join(root,name));
files.push(...walk(path.join(root,'reports/paper_pipeline_v8_launch')).filter(x=>/\\.json$/.test(x)));
` + body.slice(end);
body = body.replace('repair_code_20261106_receipt.json','paper_v8_code_20261106_receipt.json')
 .replace('paper_protocol_validated_new_sampling_started','paper_v8_initialization_audited_pilots_in_progress')
 .replace('REPAIR_SOURCE_MANIFEST_20261106.json','PAPER_V8_SOURCE_MANIFEST_20261106.json')
 .replace('Record repaired diffusion and paper-aligned reference validation protocol',
          'Align reference optimizer initialization and resume verified paper-protocol pilots')
 .replace('repaired final model publication is tracked separately.',
          'repaired final model asset715377849 has passed full TAR/member readback. One new Pick reference has passed; other pilots, full-library quality, new tracking training and dynamic evaluation remain pending.');
const target = path.join(__dirname, 'publish_paper_v8_code_20261106.cjs');
if(fs.existsSync(target)){if(fs.readFileSync(target,'utf8')!==body)throw Error('Different v8 publisher exists');}
else fs.writeFileSync(target,body);
console.log('Prepared v8 increment publisher');

const fs=require('fs'),path=require('path');
const {pathToFileURL,fileURLToPath}=require('url');
const crypto=require('crypto');
const {chromium}=require('C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='D:/具身智能/conditional-human-motion-generation/reports/teacher_report_20261002';
(async()=>{
 const manifest=JSON.parse(fs.readFileSync(path.join(root,'bundle_manifest.json'),'utf8'));
 const original=fs.readFileSync(path.join(root,'local_source/实验结果汇报.html'));
 if(crypto.createHash('sha256').update(original).digest('hex')!==manifest.source_report_sha256)throw Error('Exact source snapshot mismatch');
 for(const source of manifest.report_sources||[]){
  const bytes=fs.readFileSync(path.join(root,source.exact_source_copy));
  if(crypto.createHash('sha256').update(bytes).digest('hex')!==source.source_sha256)throw Error('Paper source snapshot mismatch: '+source.file);
 }
 const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
 const page=await browser.newPage({viewport:{width:1450,height:950}});const failures=[],errors=[];let links=0;
 page.on('pageerror',e=>errors.push(String(e)));
 const pages=manifest.files.filter(x=>x.path.endsWith('.html')).map(x=>x.path);
 pages.push('原始资产保存位置.html');
 for(const relative of pages){
  const source=path.join(root,relative);
  await page.goto(pathToFileURL(source).href,{waitUntil:'domcontentloaded'});
  const refs=await page.evaluate(()=>[...document.querySelectorAll('[href],[src],[poster]')].flatMap(x=>['href','src','poster'].map(k=>x.getAttribute(k)).filter(Boolean)).map(x=>new URL(x,document.baseURI).href));
  for(const url of refs){if(!url.startsWith('file:'))continue;links++;const p=fileURLToPath(new URL(url.split('#')[0]));if(!fs.existsSync(p))failures.push({page:relative,url})}
  if(relative==='DreamControl_实验结果汇报.html'){
   const counts=await page.evaluate(()=>({human:document.querySelectorAll('#generation video').length,robot:document.querySelectorAll('#videos video').length,hasPPO:!!document.querySelector('#ppo-six-task-interpretation'),curveTasks:document.querySelectorAll('#training details.training-chart').length,hasGroundTop:!!document.querySelector('#task-video-Ground-Pick-Top'),hasFormat:!!document.querySelector('#frozen-policy-file-formats')}));
   if(counts.human!==18||counts.robot!==32||!counts.hasPPO||!counts.hasFormat||counts.curveTasks!==11||!counts.hasGroundTop)throw Error(JSON.stringify(counts));
  }
  if(relative==='CLoSD_实验结果汇报.html'){
   const scope=await page.evaluate(()=>({tables:document.querySelectorAll('#closd table').length,dreamBaseline:!!document.querySelector('#baseline'),videos:document.querySelectorAll('video').length}));
   if(scope.tables!==2||scope.dreamBaseline||scope.videos!==0)throw Error(JSON.stringify(scope));
  }
 }
 if(failures.length||errors.length)throw Error(JSON.stringify({failures,errors}));
 const result={htmlPages:pages.length,localLinksChecked:links,missing:failures,pageErrors:errors,sourceSha256:manifest.source_report_sha256,bytes:manifest.total_bytes,licensedRawAssetsIncluded:false};
 fs.writeFileSync(path.join(root,'archive_validation.json'),JSON.stringify(result,null,2));
 console.log(JSON.stringify(result));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});

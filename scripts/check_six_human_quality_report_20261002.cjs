const fs=require('fs');
const path=require('path');
const {pathToFileURL}=require('url');
const {chromium}=require('C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='D:/具身智能/通用条件人体运动生成_已完成实验结果_20260929';
const proof=path.join(root,'论文复现汇报资料_20261002/生成与验收');
(async()=>{
 const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
 const page=await browser.newPage({viewport:{width:1500,height:1000}});const errors=[];
 page.on('pageerror',e=>errors.push(String(e)));
 await page.goto(pathToFileURL(path.join(root,'实验结果汇报.html')).href,{waitUntil:'domcontentloaded'});
 await page.addStyleTag({content:'html{scroll-behavior:auto!important}'});
 const check=await page.evaluate(()=>{
  const quality=document.querySelector('#six-task-human-quality-screening');
  const paper=document.querySelector('#dreamcontrol-paper-physical-filtering');
  return {qualityRows:quality.querySelectorAll('table')[0].tBodies[0].rows.length,
   has342:quality.innerText.includes('342'),hasThresholds:quality.innerText.includes('0.20 m/s')&&quality.innerText.includes('自身的时间中位数'),
   paperRows:paper.querySelectorAll('tbody tr').length,hasTableXI:paper.innerText.includes('表XI'),
   generationVideos:document.querySelector('#generation').querySelectorAll('video').length,
   robotVideos:document.querySelector('#videos').querySelectorAll('video').length,
   duplicateIds:[...document.querySelectorAll('[id]')].map(x=>x.id).filter((x,i,a)=>a.indexOf(x)!==i)};
 });
 if(check.qualityRows!==6||!check.has342||!check.hasThresholds||check.paperRows!==4||!check.hasTableXI||check.generationVideos!==18||check.robotVideos!==28||check.duplicateIds.length)throw Error(JSON.stringify(check));
 await page.locator('#six-task-human-quality-screening').evaluate(el=>el.scrollIntoView({behavior:'instant'}));
 await page.screenshot({path:path.join(proof,'六任务人体质量筛查_桌面.png')});
 await page.locator('#dreamcontrol-paper-physical-filtering').evaluate(el=>el.scrollIntoView({behavior:'instant'}));
 await page.screenshot({path:path.join(proof,'DreamControl原文过滤_桌面.png')});
 await page.setViewportSize({width:390,height:844});
 await page.locator('#six-task-human-quality-screening').evaluate(el=>el.scrollIntoView({behavior:'instant'}));
 const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth);
 await page.screenshot({path:path.join(proof,'六任务人体质量筛查_手机.png')});
 if(overflow||errors.length)throw Error(JSON.stringify({overflow,errors}));
 check.mobilePageOverflow=overflow;check.pageErrors=errors;
 fs.writeFileSync(path.join(proof,'六任务人体质量筛查_浏览器验收.json'),JSON.stringify(check,null,2));
 console.log(JSON.stringify(check));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});

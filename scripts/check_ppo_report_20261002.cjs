const fs=require('fs'),path=require('path');
const {pathToFileURL}=require('url');
const {chromium}=require('C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='D:/具身智能/通用条件人体运动生成_已完成实验结果_20260929';
const proof=path.join(root,'论文复现汇报资料_20261002/PPO原理与策略导出');
(async()=>{
 const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
 const page=await browser.newPage({viewport:{width:1500,height:1000}});const errors=[];
 page.on('pageerror',e=>errors.push(String(e)));
 await page.goto(pathToFileURL(path.join(root,'实验结果汇报.html')).href,{waitUntil:'domcontentloaded'});
 await page.addStyleTag({content:'html{scroll-behavior:auto!important}'});
 const sections=['ppo-optimization-flow','ppo-metric-priorities','ppo-six-task-interpretation','frozen-policy-file-formats'];
 const check=await page.evaluate(ids=>({
  sectionCounts:ids.map(id=>document.querySelectorAll('#'+id).length),
  taskRows:document.querySelectorAll('#ppo-six-task-interpretation tbody tr').length,
  formatRows:document.querySelectorAll('#frozen-policy-file-formats table')[1].tBodies[0].rows.length,
  generationVideos:document.querySelectorAll('#generation video').length,
  robotVideos:document.querySelectorAll('#videos video').length,
  duplicateIds:[...document.querySelectorAll('[id]')].map(x=>x.id).filter((x,i,a)=>a.indexOf(x)!==i),
  localLinks:ids.flatMap(id=>[...document.querySelectorAll('#'+id+' a[href]')].map(x=>x.getAttribute('href'))).filter(x=>!/^https?:/.test(x)),
  containsCaveats:document.querySelector('#ppo-six-task-interpretation').innerText.includes('不能统一写成') && document.querySelector('#frozen-policy-file-formats').innerText.includes('没有使用ONNX后端'),
 }),sections);
 if(check.sectionCounts.some(n=>n!==1)||check.taskRows!==6||check.formatRows!==6||check.generationVideos!==18||check.robotVideos!==28||check.duplicateIds.length||!check.containsCaveats)throw Error(JSON.stringify(check));
 for(const link of check.localLinks){const p=path.join(root,decodeURIComponent(link).split('#')[0]);if(!fs.existsSync(p))throw Error('Missing local link: '+p)}
 for(const id of sections){await page.locator('#'+id).evaluate(el=>el.scrollIntoView({behavior:'instant'}));await page.screenshot({path:path.join(proof,id+'_桌面.png')});}
 await page.locator('#training details.training-chart').evaluateAll(xs=>xs.forEach(x=>x.open=true));
 await page.locator('#training img.curve-img').evaluateAll(xs=>xs.forEach(x=>x.loading='eager'));
 await page.waitForFunction(()=>[...document.querySelectorAll('#training img.curve-img')].every(x=>x.complete&&x.naturalWidth>0),{timeout:10000});
 check.loadedCurveImages=await page.locator('#training img.curve-img').evaluateAll(xs=>xs.filter(x=>x.complete&&x.naturalWidth>0).length);
 if(check.loadedCurveImages!==6)throw Error('Curve images missing');
 await page.setViewportSize({width:390,height:844});
 await page.locator('#ppo-six-task-interpretation').evaluate(el=>el.scrollIntoView({behavior:'instant'}));
 check.mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth);
 await page.screenshot({path:path.join(proof,'PPO曲线解读_手机.png')});
 if(check.mobileOverflow||errors.length)throw Error(JSON.stringify({check,errors}));
 check.pageErrors=errors;fs.writeFileSync(path.join(proof,'浏览器验收.json'),JSON.stringify(check,null,2));
 console.log(JSON.stringify(check));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});

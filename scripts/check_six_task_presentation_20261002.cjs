const {chromium}=require('C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),path=require('path'),{pathToFileURL}=require('url');
const root='D:/具身智能/通用条件人体运动生成_已完成实验结果_20260929';
const proof=path.join(root,'论文复现汇报资料_20261002/生成与验收');
(async()=>{
 const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1000}});const errors=[];page.on('pageerror',e=>errors.push(String(e)));
 await page.goto(pathToFileURL(path.join(root,'实验结果汇报.html')).href);
 if(await page.locator('.human-preview').count()!==6||await page.locator('.human-additional').count()!==12)throw Error('Six tasks / eighteen videos');
 const tasks=await page.locator('.human-preview').evaluateAll(v=>v.map(x=>x.dataset.humanTask));
 if(new Set(tasks).size!==6)throw Error('not six distinct tasks');
 await page.locator('#generation').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(proof,'六类人体原始生成展示.png')});
 await page.locator('.human-preview').first().scrollIntoViewIfNeeded();await page.screenshot({path:path.join(proof,'六类人体视频网格.png')});
 const played=[];const vs=page.locator('#generation video');
 for(let i=0;i<await vs.count();i++){
  const v=vs.nth(i);const item=await v.evaluate(async v=>{
   v.preload='auto';v.load();await new Promise((resolve,reject)=>{if(v.readyState>=1)return resolve();v.addEventListener('loadedmetadata',resolve,{once:true});v.addEventListener('error',()=>reject(Error('media load failed '+v.currentSrc)),{once:true})});
   await v.play();await new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('no play')),5000);v.addEventListener('timeupdate',()=>{if(v.currentTime>.05){clearTimeout(timer);resolve()}},{once:true})});v.pause();return {src:v.currentSrc,duration:v.duration,width:v.videoWidth,height:v.videoHeight};
  });if(Math.abs(item.duration-9.8)>.05||item.width!==1280)throw Error('duration / resolution');played.push(item);
 }
 const bad=await page.evaluate(()=>['老师可观察的问题','怎样看视频','怎么样看视频','供老师观察'].filter(t=>document.body.textContent.includes(t)));if(bad.length)throw Error('phrases '+bad);
 await page.locator('.human-preview video').evaluateAll(async videos=>{await Promise.all(videos.map(async v=>{v.pause();v.currentTime=3;await new Promise(resolve=>v.addEventListener('seeked',resolve,{once:true}))}))});
 await page.locator('.human-preview').first().scrollIntoViewIfNeeded();await page.screenshot({path:path.join(proof,'六类人体视频网格.png')});
 await page.locator('#sim-purpose-paper').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(proof,'跨仿真验证论文依据.png')});
 await page.setViewportSize({width:390,height:844});await page.reload();if(await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth+2))throw Error('mobile overflow');
 if(errors.length)throw Error(JSON.stringify(errors));const result={six_distinct_tasks:tasks,all_eighteen_videos_played:played,formal_language_checked:true,paper_basis_visible:true,mobile_no_overflow:true,page_errors:errors};
 fs.writeFileSync(path.join(proof,'六任务人体视频浏览器验收.json'),JSON.stringify(result,null,2));console.log(JSON.stringify({tasks,videos:played.length,errors}));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});

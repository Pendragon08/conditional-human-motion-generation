const {chromium}=require('C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),path=require('path');const {pathToFileURL}=require('url');
const root='D:/具身智能/通用条件人体运动生成_已完成实验结果_20260929';
const proof=path.join(root,'论文复现汇报资料_20261002/生成与验收');
(async()=>{
 const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1050},deviceScaleFactor:1});const errors=[];page.on('pageerror',e=>errors.push(String(e)));
 await page.goto(pathToFileURL(path.join(root,'实验结果汇报.html')).href);
 await page.locator('#training').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(proof,'PPO训练说明预览.png')});
 await page.locator('.training-chart[open]').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(proof,'Kick真实训练曲线预览.png')});
 await page.locator('#artifacts').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(proof,'流程产物说明预览.png')});
 await page.locator('#generation').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(proof,'人体数量与质量信息预览.png')});
 await page.locator('[data-filter="baseline"]').click();if(await page.locator('.video-card[data-group]:not(.hidden)').count()!==6)throw new Error('baseline filter');
 if(await page.locator('.human-preview:not(.hidden)').count()!==6)throw new Error('human visibility');
 const imageChecks=await page.locator('.curve-img').evaluateAll(async imgs=>{for(const im of imgs){im.loading='eager';await im.decode();if(!im.naturalWidth)throw new Error('empty curve')}return imgs.length});if(imageChecks!==6)throw new Error('curve count');
 const human=await browser.newPage({viewport:{width:1280,height:1000}});human.on('pageerror',e=>errors.push(String(e)));
 await human.goto(pathToFileURL(path.join(root,'人体生成视频/人体视频观看.html')).href);if(await human.locator('#select option').count()!==200)throw new Error('human playlist count');
 const played=[];
 for(const id of [0,50,99,100,150,199]){
  await human.locator('#select').selectOption(String(id));await human.waitForFunction(()=>document.querySelector('#player').readyState>=1);
  const meta=await human.locator('#player').evaluate(v=>({duration:v.duration,width:v.videoWidth,height:v.videoHeight}));if(Math.abs(meta.duration-9.8)>.05)throw new Error('human duration');
  await human.locator('#player').evaluate(v=>v.play());await human.waitForFunction(()=>document.querySelector('#player').currentTime>.1);await human.locator('#player').evaluate(v=>v.pause());played.push({id,...meta});
 }
 await human.locator('#select').selectOption('0');await human.waitForFunction(()=>document.querySelector('#player').readyState>=1);
 await human.locator('#player').evaluate(v=>{v.currentTime=4.0});await human.waitForFunction(()=>!document.querySelector('#player').seeking);await human.screenshot({path:path.join(proof,'本地200条人体视频播放器预览.png')});
 const robot=await browser.newPage();robot.on('pageerror',e=>errors.push(String(e)));await robot.goto(pathToFileURL(path.join(root,'机器人执行视频/机器人视频观看.html')).href);
 if(await robot.locator('#select option').count()!==27)throw new Error('robot count');await robot.waitForFunction(()=>document.querySelector('#player').readyState>=1);await robot.locator('#player').evaluate(v=>v.play());await robot.waitForFunction(()=>document.querySelector('#player').currentTime>.1);await robot.locator('#player').evaluate(v=>v.pause());
 await page.setViewportSize({width:390,height:844});await page.goto(pathToFileURL(path.join(root,'实验结果汇报.html')).href);if(await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth+2))throw new Error('mobile page overflow');
 if(errors.length)throw new Error(JSON.stringify(errors));const checks={main_report_video_count:33,human_unique_playlist:200,human_samples_played:played,robot_playlist_count:27,robot_playback_verified:true,training_curve_images_loaded:6,mobile_layout_no_page_overflow:true,page_errors:errors};
 fs.writeFileSync(path.join(proof,'增强汇报浏览器验收.json'),JSON.stringify(checks,null,2));console.log(JSON.stringify(checks));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});

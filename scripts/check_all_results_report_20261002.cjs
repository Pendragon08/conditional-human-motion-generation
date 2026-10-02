const {chromium}=require('C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs');
const path=require('path');
const {pathToFileURL}=require('url');
const root='D:/具身智能/通用条件人体运动生成_已完成实验结果_20260929';
(async()=>{
 const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1050},deviceScaleFactor:1});
 const errors=[];page.on('pageerror',e=>errors.push(String(e)));
 await page.goto(pathToFileURL(path.join(root,'实验结果汇报.html')).href);
 await page.screenshot({path:path.join(root,'汇报资料_20261002/汇报首页预览.png')});
 await page.locator('#sim').scrollIntoViewIfNeeded();
 await page.screenshot({path:path.join(root,'汇报资料_20261002/跨仿真表格预览.png')});
 await page.locator('[data-filter="baseline"]').click();
 if(await page.locator('.video-card:not(.hidden)').count()!==6)throw new Error('baseline filter');
 const video=page.locator('.video-card:not(.hidden) video').first();
 await video.evaluate(v=>{v.preload='auto';v.load()});
 await page.waitForFunction(()=>{let v=document.querySelector('.video-card:not(.hidden) video');return v.readyState>=1&&Number.isFinite(v.duration)&&v.duration>0});
 const duration=await video.evaluate(v=>v.duration);
 await video.evaluate(v=>v.play());
 await page.waitForFunction(()=>document.querySelector('.video-card:not(.hidden) video').currentTime>0.15);
 await video.evaluate(v=>v.pause());
 await page.locator('[data-filter="sim"]').click();
 if(await page.locator('.video-card:not(.hidden)').count()!==9)throw new Error('sim filter');
 await page.locator('#search').fill('CLoSD');
 const matches=await page.locator('.searchable:not(.hidden)').count();
 if(matches===0||errors.length)throw new Error(JSON.stringify({matches,errors}));
 await page.locator('#search').fill('');
 const videoMetadata=await page.locator('video').evaluateAll(async vs=>{
  const rows=[];
  for(const v of vs){
   v.preload='metadata';v.load();
   if(v.readyState<1)await new Promise((resolve,reject)=>{v.onloadedmetadata=resolve;v.onerror=()=>reject(new Error(v.currentSrc));});
   if(!(v.duration>0&&v.videoWidth>0))throw new Error('Invalid video '+v.currentSrc);
   rows.push({file:decodeURI(v.currentSrc),duration_seconds:v.duration,width:v.videoWidth,height:v.videoHeight});
  }
  return rows;
 });
 const checks={page_title:await page.title(),video_count:await page.locator('video').count(),baseline_filter:6,sim_filter:9,video_playback_verified:true,first_video_duration_seconds:duration,all_video_metadata_verified:true,videos:videoMetadata,search_matches:matches,page_errors:errors};
 fs.writeFileSync(path.join(root,'汇报资料_20261002/页面验收.json'),JSON.stringify(checks,null,2));
 console.log(JSON.stringify({video_count:checks.video_count,all_video_metadata_verified:checks.all_video_metadata_verified,video_playback_verified:checks.video_playback_verified,page_errors:errors}));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});

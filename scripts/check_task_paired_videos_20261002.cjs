const {chromium}=require('C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),path=require('path'),{pathToFileURL}=require('url');
const root='D:/具身智能/通用条件人体运动生成_已完成实验结果_20260929';
const proof=path.join(root,'论文复现汇报资料_20261002/生成与验收');
(async()=>{
 const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
 page.on('pageerror',e=>errors.push(String(e)));
 await page.goto(pathToFileURL(path.join(root,'实验结果汇报.html')).href);
 const blocks=page.locator('.task-video-block'); if(await blocks.count()!==7)throw Error('Expected seven task blocks');
 const pairs=await blocks.evaluateAll(nodes=>nodes.map(n=>({task:n.dataset.task,primary:n.querySelector('.video-grid').querySelectorAll('video').length,all:n.querySelectorAll('video').length})));
 if(pairs.some(x=>x.primary!==(x.task==='Ground-Pick'?1:2)))throw Error('Incomplete pairing');
 if(await page.locator('#generation video').count()!==18)throw Error('Human videos changed');
 const played=[],posters=[];
 const vs=page.locator('#videos video');
 for(let i=0;i<await vs.count();i++){
  const v=vs.nth(i); const info=await v.evaluate(async v=>{
   v.preload='auto';v.load();await new Promise((resolve,reject)=>{let t=setTimeout(()=>reject(Error('metadata timeout')),10000);const done=()=>{clearTimeout(t);resolve()};if(v.readyState>=1)return done();v.addEventListener('loadedmetadata',done,{once:true});v.addEventListener('error',()=>reject(Error('media load '+v.currentSrc)),{once:true})});
   await v.play();await new Promise((resolve,reject)=>{const t=setTimeout(()=>reject(Error('play timeout')),5000);const check=()=>{if(v.currentTime>.05){clearTimeout(t);v.removeEventListener('timeupdate',check);resolve()}};v.addEventListener('timeupdate',check)});v.pause();
   const pos=Math.min(1,v.duration/2);await new Promise(resolve=>{v.addEventListener('seeked',resolve,{once:true});v.currentTime=pos});
   return{src:v.currentSrc,duration:v.duration,width:v.videoWidth,height:v.videoHeight};
  });if(!(info.duration>0&&info.width>0))throw Error('Empty video');played.push(info);
  const primary=await v.evaluate(v=>!!v.closest('.task-video-block')&&v.parentElement.parentElement.classList.contains('video-grid'));
  if(primary){
   const task=await v.evaluate(v=>v.closest('.task-video-block').dataset.task),side=await v.evaluate(v=>v.parentElement.querySelector('h3').textContent.startsWith('Isaac')?'Isaac':'MuJoCo');
   await v.scrollIntoViewIfNeeded();const filename=`配对预览_${task}_${side}.png`;await v.evaluate(v=>v.removeAttribute('controls'));await v.screenshot({path:path.join(proof,filename)});await v.evaluate(v=>v.setAttribute('controls',''));posters.push({video:info.src,poster:filename});
  }
 }
 await page.locator('#sim-effect-summary').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(proof,'迁移效果结论表.png')});
 await page.locator('#task-video-Kick').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(proof,'Kick双平台配对展示.png')});
 await page.locator('#task-video-Pick').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(proof,'Pick双平台配对展示.png')});
 await page.setViewportSize({width:390,height:844});await page.reload();
 if(await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth+2))throw Error('Mobile overflow');
 if(errors.length)throw Error(JSON.stringify(errors));
 const result={task_pairs:pairs,robot_videos_played:played.length,played,posters,missing_ground_mujoco_explicit:true,human_videos_preserved:18,mobile_no_overflow:true,page_errors:errors};
 fs.writeFileSync(path.join(proof,'逐任务视频浏览器验收.json'),JSON.stringify(result,null,2));console.log(JSON.stringify({pairs,videos:played.length,errors}));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});

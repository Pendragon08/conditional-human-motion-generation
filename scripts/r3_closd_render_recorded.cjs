// Offline artifact renderer. Private MeshCat scene assets and transforms stay local.
const fs=require('fs'),path=require('path'),zlib=require('zlib');
const {pathToFileURL}=require('url');
const {spawn}=require('child_process');
const {chromium}=require('C:/Users/Pende/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root='D:/具身智能/conditional-human-motion-generation';
const source=path.join(root,'artifacts/closd_recordings_20261006/released_pose_pair_v1');
const output=path.join(root,'media/closd_released_20261006');
fs.mkdirSync(output,{recursive:true});
const preview=process.argv.includes('--preview');
const ffmpeg=process.env.R3_FFMPEG;
async function setup(page,dir,data,index){
 const meta=data.metadata[index];
 await page.goto(pathToFileURL(path.join(dir,meta.task.toLowerCase()+'_private_scene.html')).href);
 await page.waitForFunction(()=>window.viewer&&window.viewer.scene.children.length>2);
 await page.evaluate(async ({data,index})=>{
  // MeshCat static HTML applies embedded asset commands using asynchronous data fetches.
  await Promise.resolve();
  window.recorded=data;window.recordIndex=index;
  document.querySelectorAll('.dg').forEach(x=>x.style.display='none');
  const m=data.metadata[index];
  const points=data.frames.flatMap(f=>{const p=f.environments[index].poses_xyzw;return [p[0],p[p.length-1]];});
  const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
  const center=[(Math.min(...xs)+Math.max(...xs))/2,(Math.min(...ys)+Math.max(...ys))/2,0.9];
  const extent=Math.max(2,Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys));
  // MeshCat rotates its Z-up scene into Three.js Y-up world coordinates.
  // The camera parent cancels that scene rotation, so controls use world coordinates.
  viewer.camera.up.set(0,1,0);
  viewer.camera.fov=50;
  viewer.camera.position.set(center[0]+extent*1.2,3.0,-center[1]+extent*1.4);
  viewer.set_camera_target([center[0],center[2],-center[1]]);
  viewer.scene.traverse(o=>{
   if(!o.isMesh)return;
   let actor=null;for(let a=o;a;a=a.parent)if(a.name==='actor:0'||a.name==='actor:1'){actor=a.name;break;}
   for(const mat of (Array.isArray(o.material)?o.material:[o.material])){
    if(mat&&mat.color&&actor==='actor:0')mat.color.setHex(0xd89145);
    if(mat&&mat.color&&actor==='actor:1')mat.color.setHex(0x4688a8);
   }
  });
  const hud=document.createElement('div');hud.id='r3-hud';
  Object.assign(hud.style,{position:'absolute',left:'12px',top:'12px',padding:'8px 12px',background:'rgba(14,26,46,.85)',color:'white',font:'14px Arial',borderRadius:'6px',zIndex:100});document.body.append(hud);
  window.showFrame=i=>{
   const frame=recorded.frames[i],e=frame.environments[recordIndex],m=recorded.metadata[recordIndex];
   let n=0;
   m.body_names.forEach((names,a)=>names.forEach(name=>{
    const [px,py,pz,x,y,z,w]=e.poses_xyzw[n++];
    // Isaac Gym rigid-body state tensors are already in environment-local coordinates.
    const matrix=[1-2*y*y-2*z*z,2*x*y+2*z*w,2*x*z-2*y*w,0,
      2*x*y-2*z*w,1-2*x*x-2*z*z,2*y*z+2*x*w,0,
      2*x*z+2*y*w,2*y*z-2*x*w,1-2*x*x-2*y*y,0,px,py,pz,1];
    viewer.handle_command({type:'set_transform',path:`/Sim/env:0/actor:${a}/${name}`,matrix});
   }));
   hud.textContent=`${m.task} | ${e.state} | ${frame.simulation_seconds.toFixed(2)} s | Isaac Gym recorded state`;
   viewer.render();
  };
  showFrame(0);
 },{data,index});
}
(async()=>{
 const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true,args:['--enable-webgl','--ignore-gpu-blocklist']});
 try{
  for(const model of ['CLoSD_multitask_finetune','CLoSD_no_finetune']){
   const dir=path.join(source,model),data=JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(dir,'actual_poses.json.gz'))));
   const jobs=data.metadata.map(async (m,index)=>{
    const page=await browser.newPage({viewport:{width:640,height:480}});
    await setup(page,dir,data,index);
    const stem=model+'_'+m.task;
    for(const i of [0,300,600]){await page.evaluate(i=>showFrame(i),i);await page.screenshot({path:path.join(output,stem+'_'+i+'.jpg'),type:'jpeg',quality:85});}
    if(!preview){
     if(!ffmpeg)throw Error('Set R3_FFMPEG to the encoder executable');
     const outfile=path.join(output,stem+'.mp4');if(fs.existsSync(outfile))throw Error('Refusing to overwrite '+outfile);
     const enc=spawn(ffmpeg,['-hide_banner','-loglevel','error','-f','image2pipe','-vcodec','mjpeg','-framerate','30','-i','pipe:0','-c:v','libx264','-pix_fmt','yuv420p','-crf','21','-movflags','+faststart',outfile]);
     let err='';enc.stderr.on('data',b=>err+=b);
     const ended=new Promise((resolve,reject)=>{enc.on('error',reject);enc.on('exit',code=>code===0?resolve():reject(Error(err)));});
     for(let i=0;i<data.frames.length;i++){
      await page.evaluate(i=>showFrame(i),i);
      const jpeg=await page.screenshot({type:'jpeg',quality:85});
      if(!enc.stdin.write(jpeg))await new Promise(r=>enc.stdin.once('drain',r));
     }
     enc.stdin.end();await ended;console.log('VIDEO_COMPLETE '+outfile);
    }
    await page.close();
   });
   await Promise.all(jobs);
  }
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});

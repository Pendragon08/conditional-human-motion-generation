// Transcode the original Viser browser view into report videos.
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const { createHash } = require('node:crypto');
const { spawn, spawnSync } = require('node:child_process');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const root = 'D:/DreamControl_v2_paper_protocol_report_v10';
const assets = path.resolve('D:/DreamControl_v2_reference_videos_upstream_20261105/build');
const browser = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const remote = '/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/repair_prior29_v2_20261105';
const sshArgs = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '10.147.17.32'];
const port = 18121, debugPort = 18122;
const minimum = 1073741824 + 100 * 1048576;
const digest = data => createHash('sha256').update(data).digest('hex');
const names = {pick:'Pick', pour:'Pour', bimanual_box_lift:'Bimanual box lift', box_push_pull:'Box push / pull'};
const mime = {'.html':'text/html','.js':'text/javascript','.css':'text/css','.svg':'image/svg+xml','.wasm':'application/wasm','.json':'application/json','.ttf':'font/ttf'};
let current;
function ssh(command, input) {
    return spawnSync('ssh', [...sshArgs, command], {input, windowsHide:true, timeout:120000, maxBuffer:20*1048576});
}
function checkDisk(extra) {
    const stats = fs.statfsSync('D:/');
    if (stats.bsize * stats.bavail < minimum + extra) throw Error('D: disk reserve reached');
}
function page(meta) {
    return `<!doctype html><meta charset="utf-8"><style>body{background:#18202a;margin:0}iframe{width:640px;height:520px;border:0}#views{display:flex}</style>
<div id="views"><iframe id="full"></iframe><iframe id="close"></iframe></div><canvas id="output" width="1280" height="604"></canvas><script>
const meta=${JSON.stringify(meta)}, views=[document.getElementById('full'),document.getElementById('close')];
const output=document.getElementById('output'), ctx=output.getContext('2d');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const slider=f=>f.contentDocument.querySelector('input');
const canvas=f=>f.contentDocument.querySelector('canvas[data-engine]');
const time=f=>Number(slider(f)?.value || 0);
function seek(f,t){const el=slider(f);Object.getOwnPropertyDescriptor(f.contentWindow.HTMLInputElement.prototype,'value').set.call(el,String(t));el.dispatchEvent(new f.contentWindow.Event('input',{bubbles:true}));el.dispatchEvent(new f.contentWindow.Event('change',{bubbles:true}));}
function play(f){f.contentWindow.dispatchEvent(new f.contentWindow.KeyboardEvent('keydown',{code:'Space',bubbles:true}));}
function draw(){ctx.fillStyle='#18202a';ctx.fillRect(0,0,1280,604);for(let i=0;i<2;i++){const source=canvas(views[i]);if(source)ctx.drawImage(source,640*i,42,640,520);}
ctx.fillStyle='white';ctx.font='20px sans-serif';ctx.fillText(meta.title+' | '+time(views[0]).toFixed(2)+' s',14,28);
ctx.font='16px sans-serif';ctx.fillText(meta.footer,14,587);requestAnimationFrame(draw);}
async function run(){
  views[0].src='/index.html?playbackPath=/clip.viser&initialCameraPosition=2.15,-2.3,1.55&initialCameraLookAt=.10,0,.75&initialCameraUp=0,0,1';
  views[1].src='/index.html?playbackPath=/clip.viser&initialCameraPosition=1.28,-.85,1.25&initialCameraLookAt=.22,0,1&initialCameraUp=0,0,1';
  const ready=Date.now()+45000;
  while(views.some(f=>!slider(f)||!canvas(f))){if(Date.now()>ready)throw Error('Viewer failed to load: '+views.map(f=>[!!slider(f),!!canvas(f),f.contentDocument.body.innerText.slice(0,300)].join(' / ')).join(' || '));await sleep(100);}
  views.forEach(f=>seek(f,0));draw();await sleep(650);
  const stream=output.captureStream(25), chunks=[];
  const type=['video/mp4;codecs=avc1.42001E','video/webm;codecs=vp8'].find(t=>MediaRecorder.isTypeSupported(t));
  if(!type)throw Error('No supported recorder');
  const rec=new MediaRecorder(stream,{mimeType:type,videoBitsPerSecond:1800000});
  const viewFingerprints=[];
  const stop=new Promise(r=>{rec.onstop=r;});rec.ondataavailable=e=>{if(e.data.size)chunks.push(e.data);};
  rec.start(250);await sleep(250);views.forEach(play);
  const end=performance.now()+(meta.duration_seconds+12)*1000;let reached=0, wanted=meta.duration_seconds;
  while(reached<wanted-.14){if(performance.now()>end)throw Error('Playback incomplete: '+reached+' / '+wanted);reached=Math.max(reached,time(views[0]));if(viewFingerprints.length<2&&reached>1+2*viewFingerprints.length){const pixels=ctx.getImageData(20,60,1240,480).data;let h=0;for(let j=0;j<pixels.length;j+=4)h=(Math.imul(h,31)^pixels[j]^pixels[j+1])|0;viewFingerprints.push(h);}await sleep(20);}
  views.forEach(f=>seek(f,wanted-.06));await sleep(350);
  if(viewFingerprints.length===2&&viewFingerprints[0]===viewFingerprints[1])throw Error('Rendered scene pixels did not change');
  rec.stop();await stop;stream.getTracks().forEach(t=>t.stop());
  const result=await fetch('/save',{method:'POST',body:new Blob(chunks,{type}),headers:{'x-video-type':type,'x-viewer-seconds':String(reached)}});
  if(!result.ok)throw Error(await result.text());
}
run().catch(async e=>{await fetch('/error',{method:'POST',body:String(e)});});
</script>`;
}
function serve(req, res) {
    if(process.argv[2]==='--smoke') console.log(req.method,req.url);
    const target = new URL(req.url, 'http://localhost').pathname;
    if (req.method === 'POST' && (target === '/save' || target === '/error')) {
        const chunks=[];let count=0;
        req.on('data', data => {count+=data.length;if(count>12*1048576)req.destroy();else chunks.push(data);});
        req.on('end', () => {
            const data=Buffer.concat(chunks);
            if(target==='/error') current.reject(Error(data.toString()));
            else current.resolve({data, type:req.headers['x-video-type'], reached:Number(req.headers['x-viewer-seconds'])});
            res.end('received');
        });
        return;
    }
    if(target==='/record'){res.setHeader('Content-Type','text/html');res.end(page(current.meta));return;}
    if(target==='/clip.viser'){res.setHeader('Content-Length',current.scene.length);res.end(current.scene);return;}
    const file=path.resolve(assets,'.'+target);
    if (!file.startsWith(assets+path.sep)||!fs.existsSync(file)||!fs.statSync(file).isFile()) {res.writeHead(404);res.end();return;}
    res.setHeader('Content-Type',mime[path.extname(file)]||'application/octet-stream');
    fs.createReadStream(file).pipe(res);
}
async function render(meta, scene) {
    checkDisk(28*1048576);
    const profile=path.resolve(root,'browser_'+process.pid);
    const server=http.createServer(serve);
    await new Promise((ok,bad)=>{server.once('error',bad);server.listen(port,'127.0.0.1',ok);});
    let timer, child;
    try {
        const video=new Promise((resolve,reject)=>{
            current={meta,scene,resolve,reject};
            timer=setTimeout(()=>reject(Error('Browser export timed out')),90000);
        });
        child=spawn(browser,['--headless=new','--no-first-run','--no-default-browser-check','--disable-background-networking',
            '--disable-component-update','--disk-cache-size=1','--use-angle=swiftshader','--enable-unsafe-swiftshader',
            '--window-size=1310,1250','--user-data-dir='+profile,'--remote-debugging-port='+debugPort,
            'http://127.0.0.1:'+port+'/record'],{windowsHide:true,stdio:'ignore'});
        child.once('error',error=>current.reject(error));
        return await video;
    } finally {
        clearTimeout(timer);
        if(child && child.exitCode===null){
            const stopped=new Promise(resolve=>child.once('exit',resolve));child.kill();await Promise.race([stopped,sleep(5000)]);
        }
        server.closeAllConnections();await new Promise(ok=>server.close(ok));
        if(profile.startsWith(path.resolve(root)+path.sep+'browser_') && fs.existsSync(profile)
            && fs.realpathSync(profile).toLowerCase()===profile.toLowerCase()){
            try {fs.rmSync(profile,{recursive:true,maxRetries:3,retryDelay:500});}catch(error){console.warn(String(error));}
        }
    }
}
async function main() {
    if(!fs.existsSync(path.join(assets,'index.html'))||!fs.existsSync(browser))throw Error('Verified original viewer assets or Chrome missing');
    if (process.argv[2] === '--smoke') {
        fs.mkdirSync(root, {recursive:true});
        const scene=fs.readFileSync(path.join(assets,'01_pick.viser'));
        const result=await render({title:'Viewer smoke',footer:'Existing reference for export preflight',duration_seconds:6},scene);
        if(result.data.length<1024||result.reached<5.7)throw Error('Capture smoke did not cover six seconds');
        console.log(JSON.stringify({video_type:result.type,bytes:result.data.length,timeline_seconds:result.reached}));
        return;
    }
    const keys=['reference','physics'].flatMap(kind=>Object.keys(names).map(task=>kind+'_'+task));
    fs.mkdirSync(root,{recursive:true});
    const lock=path.join(root,'controller.lock');
    fs.writeFileSync(lock,String(process.pid),{flag:'wx'});
    try{
        while(true){
            const todo=keys.filter(key=>!fs.existsSync(path.join(root,key+'.json')));
            if(!todo.length)break;
            const query=ssh('cat '+remote+'/report_video_sources_paper_v10/status.json');
            if(query.status!==0){await sleep(60000);continue;}
            const status=JSON.parse(query.stdout.toString());
            for(const key of todo){
                const manifest=status.completed?.[key];
                if(!manifest)continue;
                const folder=remote+'/report_video_sources_paper_v10/'+key;
                const metadata=ssh('cat '+folder+'/receipt.json'), artifact=ssh('cat '+folder+'/clip.viser');
                if(metadata.status!==0 || artifact.status!==0)continue;
                if(digest(metadata.stdout)!==manifest.sha256)throw Error('Scene receipt changed');
                const meta=JSON.parse(metadata.stdout);
                if(artifact.stdout.length!==meta.viser_bytes||digest(artifact.stdout)!==meta.viser_sha256)throw Error('Scene bytes changed');
                const reference=meta.kind==='reference';
                meta.title=names[meta.task]+' | '+(reference?'reference ':'evaluation episode 0 / ref ')+meta.candidate_id;
                meta.footer=reference?'DreamControl / Viser | Generated kinematic reference with hand plan':
                    'Recorded PhysX poses | Episode: '+(meta.episode_success?'success':'failure')+' | Task successes: '+meta.success_count+'/100';
                const clip=await render(meta,artifact.stdout);
                const ext=clip.type.startsWith('video/mp4')?'mp4':'webm';
                if(clip.data.length<1024 || !Number.isFinite(clip.reached))throw Error('Invalid media capture');
                const output=path.join(root,key+'.'+ext);
                fs.writeFileSync(output,clip.data,{flag:'wx'});
                fs.writeFileSync(path.join(root,key+'.json'),JSON.stringify({...meta, video:output, video_sha256:digest(clip.data),
                    video_bytes:clip.data.length, video_type:clip.type, viewer_seconds_reached:clip.reached},null,2),{flag:'wx'});
                console.log('completed '+output);
            }
            if(status.status==='failed_preserved')throw Error(status.traceback||'Video source preparation failed');
            await sleep(60000);
        }
    } finally {fs.unlinkSync(lock);}
}
main().catch(e=>{console.error(e);process.exitCode=1;});

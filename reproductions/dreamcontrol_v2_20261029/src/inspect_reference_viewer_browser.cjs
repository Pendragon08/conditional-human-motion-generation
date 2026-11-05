const fs = require('node:fs');
(async () => {
  const targets = await (await fetch('http://127.0.0.1:18086/json/list')).json();
  const target = targets.find(t => t.type === 'page' && /^http:\/\/127\.0\.0\.1:(18085|18087)/.test(t.url));
  if (!target) throw Error('Task viewer tab missing');
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { ws.onopen=resolve; ws.onerror=reject; });
  let id=0; const pending=new Map();
  ws.onmessage = e => { const r=JSON.parse(e.data); if(pending.has(r.id)){const [ok,bad]=pending.get(r.id);pending.delete(r.id);r.error?bad(r.error):ok(r.result);} else if(process.argv[2]==='diagnose' && /exception|console|entryAdded/i.test(r.method||'')) console.log(JSON.stringify(r)); };
  const call = (method,params={}) => new Promise((ok,bad)=>{const n=++id;pending.set(n,[ok,bad]);ws.send(JSON.stringify({id:n,method,params}));});
  if (process.argv[2] === 'local') { await call('Network.enable');await call('Network.setBlockedURLs',{urls:['*google-analytics.com*','*googletagmanager.com*']});await call('Page.navigate',{url:'http://127.0.0.1:18087/?websocket=ws%3A%2F%2F127.0.0.1%3A18085'}); }
  else if (process.argv[2] === 'diagnose') { await call('Runtime.enable');await call('Log.enable');await new Promise(r=>setTimeout(r,3000)); }
  else if (process.argv[2] === 'reload') await call('Page.reload',{ignoreCache:true});
  else {
    const state=await call('Runtime.evaluate',{expression:'JSON.stringify({title:document.title,ready:document.readyState,text:document.body.innerText,canvases:document.querySelectorAll("canvas").length,resources:performance.getEntriesByType("resource").map(r=>({name:r.name,bytes:r.encodedBodySize,duration:r.duration}))})',returnByValue:true});
    console.log(state.result.value);
    const shot=await call('Page.captureScreenshot',{format:'jpeg',quality:85});
    fs.writeFileSync(process.argv[2],Buffer.from(shot.data,'base64'));
  }
  ws.close();
})().catch(e=>{console.error(e);process.exit(1);});

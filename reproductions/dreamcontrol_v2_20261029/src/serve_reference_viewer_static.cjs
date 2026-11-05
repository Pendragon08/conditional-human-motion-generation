// Serve an exact local copy of Viser's shipped frontend. No rendering code here.
const fs=require('node:fs'),http=require('node:http'),path=require('node:path');
const root=path.resolve('D:/DreamControl_v2_reference_videos_upstream_20261105/build');
const mime={'.html':'text/html','.js':'text/javascript','.css':'text/css','.svg':'image/svg+xml','.wasm':'application/wasm','.json':'application/json','.hdr':'application/octet-stream','.ttf':'font/ttf'};
http.createServer((req,res)=>{
  const pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
  const file=path.resolve(root,'.'+(pathname==='/'?'/index.html':pathname));
  if(!file.startsWith(root+path.sep)||!fs.existsSync(file)||!fs.statSync(file).isFile()){res.writeHead(404);res.end();return;}
  res.writeHead(200,{'Content-Type':mime[path.extname(file)]||'application/octet-stream','Content-Length':fs.statSync(file).size});fs.createReadStream(file).pipe(res);
}).listen(18087,'127.0.0.1',()=>console.log('Original viewer static files ready on localhost:18087'));

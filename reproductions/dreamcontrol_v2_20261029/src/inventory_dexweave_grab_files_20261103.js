// Metadata only: no dataset downloads. Count file sizes at a pinned release.
async function main() {
  let url='https://huggingface.co/api/datasets/naichuan04/DexWeave/tree/264cde93a2b3f502ded55fe4e5c05234a2bdefa2/data/GRAB?recursive=true&limit=1000';
  const totals={pages:0,motion_files:0,motion_bytes:0,evaluation_files:0,evaluation_bytes:0};
  while(url){
    const r=await fetch(url,{signal:AbortSignal.timeout(30000)});
    if(!r.ok)throw Error(`HTTP ${r.status}`);
    for(const x of await r.json()){
      if(x.type==='file'&&x.path.endsWith('_opt.npz')){totals.motion_files++;totals.motion_bytes+=x.size;}
      if(x.type==='file'&&x.path.endsWith('_val.npz')){totals.evaluation_files++;totals.evaluation_bytes+=x.size;}
    }
    totals.pages++;
    const h=r.headers.get('link');const match=h&&h.match(/<([^>]+)>;\s*rel="next"/);url=match?match[1]:null;
    if(totals.pages>20)throw Error('Unexpected pagination');
  }
  console.log(JSON.stringify(totals));
}
main().catch(e=>{console.error(e);process.exitCode=1});

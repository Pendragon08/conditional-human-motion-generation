"""Fetch a pinned public DistilBERT candidate, not an author-supplied PhyGile model."""
from pathlib import Path
import hashlib,json,os
from huggingface_hub import HfApi,hf_hub_download

root=Path(__file__).resolve().parents[1]
destination=root/'cache/text/distilbert-base-uncased'
out=root/'results/reimplementation_text';out.mkdir(parents=True,exist_ok=True)
repository='distilbert/distilbert-base-uncased'
info=HfApi().model_info(repository,files_metadata=True)
revision=info.sha
available={entry.rfilename:entry for entry in info.siblings}
required=['config.json','tokenizer_config.json','tokenizer.json','vocab.txt','model.safetensors']
names=required+[name for name in ['README.md','LICENSE'] if name in available]
record={'status':'DOWNLOADING','repository':repository,'revision':revision,
        'model_selection':'Explicit implementation choice; paper does not identify this exact DistilBERT checkpoint',
        'files':[]}
path=out/'download.json';path.write_text(json.dumps(record,indent=2)+'\n')
try:
    for name in names:
        file=Path(hf_hub_download(repository,name,revision=revision,local_dir=destination,cache_dir=root/'cache/huggingface'))
        digest=hashlib.sha256(file.read_bytes()).hexdigest()
        metadata=available[name]
        expected=getattr(metadata.lfs,'sha256',None) if metadata.lfs else None
        if expected:assert digest==expected
        if metadata.size is not None:assert file.stat().st_size==metadata.size
        record['files'].append({'name':name,'sha256':digest,'bytes':file.stat().st_size,
                                'publisher_lfs_sha256':expected,'url':f'https://huggingface.co/{repository}/resolve/{revision}/{name}'})
        path.write_text(json.dumps(record,indent=2)+'\n')
        print('Verified',name,file.stat().st_size,flush=True)
    record['status']='COMPLETED'
except BaseException as error:
    record['status']='FAILED';record['error']=type(error).__name__+': '+str(error);raise
finally:path.write_text(json.dumps(record,indent=2)+'\n')

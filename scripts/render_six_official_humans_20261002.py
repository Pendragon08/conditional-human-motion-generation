from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import json,subprocess,hashlib,textwrap
ROOT=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/artifacts/report_all_human_20261002')
OUT=ROOT/'six_official_tasks';OUT.mkdir(exist_ok=True)
CHAINS=[[0,2,5,8,11],[0,1,4,7,10],[0,3,6,9,12,15],[9,14,17,19,21],[9,13,16,18,20]]
FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
SMALL=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
rows=[];tasks=[]
for task in ['Kick','Punch','Jump','Sit','Pick','Bimanual_Pick']:
    source=ROOT/f'{task}_official_results.npy';data=np.load(source,allow_pickle=True).item();count=len(data['motion'])
    assert np.isfinite(data['motion']).all()
    tasks.append({'task':task,'source_count':count,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'motion_shape':list(data['motion'].shape),'selection_ids':[0,1,2],'selection':'fixed first three IDs, no outcome-dependent selection'})
    for sample in range(3):
        length=int(data['lengths'][sample]);m=data['motion'][sample,:,:,:length].transpose(2,0,1);hints=data['hint'][sample,:length]
        assert m.shape==(length,22,3) and hints.shape==m.shape
        target=hints[np.any(hints!=0,axis=-1)];bounds=np.concatenate([m.reshape(-1,3),target])
        lo=bounds.min(0);hi=bounds.max(0);top=max(2.,float(hi[1])+.15);bottom=min(-.15,float(lo[1])-.05)
        base=Image.new('RGB',(1280,720),'#f0f5f9');d=ImageDraw.Draw(base);views=[]
        d.text((24,16),f'Official released OmniControl HUMAN | {task.replace("_","-")} | sample {sample:03d}',font=FONT,fill='#17394e')
        for line,t in enumerate(textwrap.wrap(str(data['text'][sample]),100)):d.text((24,45+line*19),'Text: '+t if line==0 else t,font=SMALL,fill='#536c7d')
        for x0,axis,title in [(32,0,'Front view: lateral / height'),(672,2,'Side view: forward / height')]:
            span=max(2.,float(hi[axis]-lo[axis])+.4);mid=(hi[axis]+lo[axis])/2
            def xy(points,x0=x0,axis=axis,span=span,mid=mid):return np.column_stack([x0+(points[:,axis]-mid+span/2)/span*560,615-(points[:,1]-bottom)/(top-bottom)*490])
            d.text((x0,95),title,font=SMALL,fill='#315269')
            for y in np.arange(0,top+.001,.5):
                yy=615-(y-bottom)/(top-bottom)*490;d.line([(x0,yy),(x0+560,yy)],fill='#cedbe2');d.text((x0,yy-17),f'{y:.1f} m',font=SMALL,fill='#718894')
            yy=615-(0-bottom)/(top-bottom)*490;d.line([(x0,yy),(x0+560,yy)],fill='#526f7c',width=2);views.append(xy)
        d.text((24,688),'Blue: body | orange: right side | red X: active spatial hint | original y=0 ground | no physics rollout',font=SMALL,fill='#4e6879')
        p=OUT/f'{task}_official_sample{sample:03d}.mp4'
        proc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','20','-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-threads','1','-movflags','+faststart',str(p)],stdin=subprocess.PIPE)
        for f in range(length):
            img=base.copy();d=ImageDraw.Draw(img)
            for project in views:
                points=project(m[f])
                for ci,chain in enumerate(CHAINS):d.line([tuple(points[j]) for j in chain],fill='#df8733' if ci in [0,3] else '#217d9b',width=6)
                x,y=points[15];d.ellipse([x-8,y-8,x+8,y+8],fill='#217d9b')
                for goal in project(hints[f,np.any(hints[f]!=0,axis=-1)]):
                    x,y=goal;d.line([(x-8,y-8),(x+8,y+8)],fill='#c84f53',width=3);d.line([(x-8,y+8),(x+8,y-8)],fill='#c84f53',width=3)
            d.text((24,650),f'{f/20:.2f} / {length/20:.2f} s | original 20 fps | 1x speed | HUMAN SKELETON',font=FONT,fill='#17394e')
            if f==60:img.save(p.with_suffix('.jpg'))
            proc.stdin.write(img.tobytes())
        proc.stdin.close();assert proc.wait()==0
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,r_frame_rate:format=duration','-of','json',str(p)],text=True))
        assert int(probe['streams'][0]['nb_frames'])==length and float(probe['format']['duration'])==length/20
        rows.append({'task':task,'sample_id':sample,'text':str(data['text'][sample]),'file':p.name,'poster':p.with_suffix('.jpg').name,'frames':length,'fps':20,'duration_seconds':length/20,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_sha256':tasks[-1]['source_sha256'],'source_count':count,'probe':probe,'display_only':'original world-coordinate skeleton; no smoothing, extra IK, root repositioning or physics'})
        print('RENDERED',p.name,flush=True)
(OUT/'manifest.json').write_text(json.dumps({'status':'COMPLETE','official_released_source_sequences':sum(t['source_count'] for t in tasks),'new_diffusion_inference':False,'tasks':tasks,'videos':rows},ensure_ascii=False,indent=2))

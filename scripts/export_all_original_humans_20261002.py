"""CPU-only original-human video export and declared geometric quality proxy audit.
Never changes generation, robot references, policies, or running experiments.
"""
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import json,hashlib,subprocess,datetime,csv
R=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
OUT=R/'artifacts/report_all_human_20261002';OUT.mkdir(exist_ok=True)
VID=OUT/'videos';VID.mkdir(exist_ok=True)
CHAINS=[[0,2,5,8,11],[0,1,4,7,10],[0,3,6,9,12,15],[9,14,17,19,21],[9,13,16,18,20]]
EDGES=[(a,b) for c in CHAINS for a,b in zip(c[:-1],c[1:])]
FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
SMALL=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
W,H=960,540;FPS=20
THRESHOLDS={'foot_penetration_m':-.03,'ground_proximity_m':.05,'ground_slide_speed_m_s':.2,'minimum_slide_consecutive_frames':3,'both_feet_height_m':.1,'minimum_air_consecutive_frames':6,'joint_displacement_per_frame_m':.15,'bone_relative_length_deviation':.1,'minimum_penetration_consecutive_frames':3,'minimum_bone_deviation_consecutive_frames':3}
def longest(mask):
    best=cur=0
    for b in mask:
        cur=cur+1 if b else 0;best=max(best,cur)
    return best
def audit(motion):
    feet=motion[:,[10,11],:];foot_y=feet[:,:,1]
    penetration=longest((foot_y<THRESHOLDS['foot_penetration_m']).any(1))
    speed=np.linalg.norm(np.diff(feet[:,:,[0,2]],axis=0),axis=-1)*FPS
    near=(foot_y[:-1]<=.05)&(foot_y[1:]<=.05)&(foot_y[:-1]>=-.03)&(foot_y[1:]>=-.03)
    sliding=(near&(speed>.2)).any(1);sliding_run=longest(sliding)
    airborne_run=longest((foot_y>.1).all(1))
    step=np.linalg.norm(np.diff(motion,axis=0),axis=-1);jump=float(step.max())
    lengths=np.stack([np.linalg.norm(motion[:,a]-motion[:,b],axis=1) for a,b in EDGES],1)
    med=np.median(lengths,axis=0);dev=np.abs(lengths-med)/np.maximum(med,1e-8)
    bone_run=longest((dev>.1).any(1))
    flags={'foot_below_nominal_ground':penetration>=3,'near_ground_slide_proxy':sliding_run>=3,'both_feet_airborne_proxy':airborne_run>=6,'large_per_frame_joint_change':jump>.15,'bone_length_variation_proxy':bone_run>=3}
    return {'flags':flags,'any_proxy_flag':any(flags.values()),'foot_min_y_m':float(foot_y.min()),'penetration_longest_frames':penetration,'sliding_longest_frames':sliding_run,'sliding_frames':int(sliding.sum()),'both_feet_airborne_longest_frames':airborne_run,'max_joint_change_m_per_frame':jump,'max_bone_relative_length_deviation':float(dev.max()),'bone_variation_longest_frames':bone_run}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
all_rows=[];batches=[]
for seed,folder in [(10,'sample_r3_n100'),(42,'sample_r3_n100_seed42')]:
    source=R/'third_party/DreamControl/TrajGen'/folder/'Kick_sim/results.npy'
    data=np.load(source,allow_pickle=True).item();assert data['motion'].shape==(100,22,3,196)
    assert np.isfinite(data['motion']).all();source_hash=sha(source)
    np.savez_compressed(OUT/f'Kick_seed{seed}_original_arrays.npz',motion=data['motion'],hint=data['hint'],lengths=data['lengths'],text=np.asarray(data['text'],dtype=str))
    rows=[]
    for sample in range(100):
        length=int(data['lengths'][sample]);assert length==196
        m=data['motion'][sample,:,:,:length].transpose(2,0,1);hints=data['hint'][sample,:length]
        assert m.shape==(196,22,3) and hints.shape==(196,22,3)
        qa=audit(m);p=VID/f'Kick_seed{seed}_sample{sample:03d}.mp4'
        # Front and side orthographic views, fixed throughout a sequence.
        target=hints[np.any(hints!=0,axis=-1)];bounds=np.concatenate([m.reshape(-1,3),target])
        lo=bounds.min(0);hi=bounds.max(0);top=max(2.0,float(hi[1])+.15);bottom=min(-.15,float(lo[1])-.05)
        views=[];base=Image.new('RGB',(W,H),'#f0f5f9');draw=ImageDraw.Draw(base)
        draw.text((20,12),f'Original OmniControl HUMAN | Kick | seed {seed} | sample {sample:03d}',font=FONT,fill='#17394e')
        draw.text((20,37),'Text: A person stands and kicks with his right leg.',font=SMALL,fill='#536c7d')
        for x0,axis,title in [(28,0,'Front view: lateral / height'),(500,2,'Side view: forward / height')]:
            width=410;right=x0+width;upper=100;lower=450;span=max(2.,float(hi[axis]-lo[axis])+.4);mid=(hi[axis]+lo[axis])/2
            def xy(points,x0=x0,axis=axis,span=span,mid=mid):
                return np.column_stack([x0+(points[:,axis]-mid+span/2)/span*410,450-(points[:,1]-bottom)/(top-bottom)*350])
            draw.text((x0,72),title,font=SMALL,fill='#315269')
            for yval in np.arange(0,top+.001,.5):
                yy=450-(yval-bottom)/(top-bottom)*350;draw.line([(x0,yy),(right,yy)],fill='#cedbe2',width=1);draw.text((x0,yy-14),f'{yval:.1f} m',font=SMALL,fill='#718894')
            ground=450-(0-bottom)/(top-bottom)*350;draw.line([(x0,ground),(right,ground)],fill='#526f7c',width=2)
            views.append(xy)
        draw.text((20,515),'Blue: body | orange: right leg | red X: active hint | green: left-foot trace | original y=0 ground',font=SMALL,fill='#4e6879')
        cmd=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r','20','-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','25','-pix_fmt','yuv420p','-threads','1','-movflags','+faststart',str(p)]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        for f in range(length):
            img=base.copy();d=ImageDraw.Draw(img)
            for project in views:
                points=project(m[f])
                for ci,chain in enumerate(CHAINS):d.line([tuple(points[j]) for j in chain],fill='#df8733' if ci==0 else '#217d9b',width=5)
                head=points[15];d.ellipse([head[0]-6,head[1]-6,head[0]+6,head[1]+6],fill='#217d9b')
                for joint,color in [(10,'#579e8d'),(11,'#df8733')]:
                    pts=project(m[max(0,f-30):f+1,joint]);
                    if len(pts)>1:d.line([tuple(z) for z in pts],fill=color,width=1)
                mask=np.any(hints[f]!=0,axis=-1)
                for g in project(hints[f,mask]):
                    x,y=g;d.line([(x-6,y-6),(x+6,y+6)],fill='#c84f53',width=3);d.line([(x-6,y+6),(x+6,y-6)],fill='#c84f53',width=3)
            d.text((20,481),f'{f/FPS:.2f} / 9.80 s | original 20 fps | 1x speed | SKELETON ONLY, not physics',font=FONT,fill='#17394e')
            proc.stdin.write(img.tobytes())
        proc.stdin.close();assert proc.wait()==0
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,r_frame_rate:format=duration','-of','json',str(p)],text=True))
        assert int(probe['streams'][0]['nb_frames'])==196 and float(probe['format']['duration'])==9.8
        row={'seed':seed,'sample_id':sample,'text':str(data['text'][sample]),'frames':196,'fps':20,'duration_seconds':9.8,'file':'videos/'+p.name,'bytes':p.stat().st_size,'sha256':sha(p),'probe':probe,'quality_proxy':qa};rows.append(row);all_rows.append(row)
        if sample%20==19:print('HUMAN_EXPORT_PROGRESS',seed,sample+1,flush=True)
    counts={k:sum(r['quality_proxy']['flags'][k] for r in rows) for k in rows[0]['quality_proxy']['flags']}
    batches.append({'seed':seed,'source':str(source),'source_sha256':source_hash,'count':100,'video_count':100,'duration_seconds_each':9.8,'proxy_flagged_count':sum(r['quality_proxy']['any_proxy_flag'] for r in rows),'proxy_unflagged_count':sum(not r['quality_proxy']['any_proxy_flag'] for r in rows),'flag_counts_overlapping':counts,'original_compressed_arrays':f'Kick_seed{seed}_original_arrays.npz'})
summary={'status':'COMPLETE','generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'render_only':True,'motion_sequences':200,'exported_videos':200,'single_text_class':'Kick','duration_seconds_each':9.8,'fps':20,'frames_each':196,'video_total_seconds':1960,'batches':batches,'thresholds':THRESHOLDS,'proxy_flagged_count':sum(r['quality_proxy']['any_proxy_flag'] for r in all_rows),'physics_validity_certified_count':None,'not_dynamically_validated_count':200,'interpretation':'Counts are declared geometric warning proxies, NOT a physics failure rate or natural-motion quality pass. Foot joint is not a foot sole. Airborne can be a legitimate kicking style. No collision, contact force, torque limit, stability, semantic or human rater certification is performed. No new diffusion generation, IK, smoothing, root repositioning, or physics simulation. All 200 samples retained; categories overlap.','rows':all_rows}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
with (OUT/'quality_proxy_per_sample.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['seed','sample_id','any_proxy_flag']+list(THRESHOLDS)+['foot_min_y_m','max_joint_change_m_per_frame','file'])
    for r in all_rows:w.writerow([r['seed'],r['sample_id'],r['quality_proxy']['any_proxy_flag']]+[THRESHOLDS[t] for t in THRESHOLDS]+[r['quality_proxy']['foot_min_y_m'],r['quality_proxy']['max_joint_change_m_per_frame'],r['file']])
print('ALL_ORIGINAL_HUMAN_EXPORT_COMPLETE',200,'proxy_flagged',summary['proxy_flagged_count'],flush=True)

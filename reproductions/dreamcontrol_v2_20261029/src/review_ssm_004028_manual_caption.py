"""A bounded, gender-neutral manual caption for a reviewed right-leg clip."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
DATA=ROOT/'ssm_new6_20hz_resume_20261101';CURVES=ROOT/'ssm_new5_caption_curves_20261101';HUMAN=ROOT/'ssm_new6_human_20261101';HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    sid='004028';datareport=json.loads((DATA/'report.json').read_text());audit=json.loads((DATA/'audit.json').read_text());chart=json.loads((CURVES/'report.json').read_text())
    if (sha(DATA/'report.json')!=audit['source_report_sha256'] or sha(DATA/'records.jsonl')!=datareport['records_sha256'] or
        chart['input_report_sha256']!=sha(DATA/'report.json') or chart['input_audit_sha256']!=sha(DATA/'audit.json')):raise ValueError('review inputs changed')
    rec=next(x for x in map(json.loads,(DATA/'records.jsonl').read_text().splitlines()) if x['id']==sid)
    ch=next(x for x in chart['cases'] if x['id']==sid);artifact=DATA/rec['artifact'];curve=CURVES/(sid+'_curves.npz');figure=CURVES/(sid+'.png')
    if sha(artifact)!=rec['artifact_sha256'] or sha(curve)!=ch['curves_sha256'] or sha(figure)!=ch['image_sha256']:raise ValueError('motion or curves changed')
    with np.load(artifact,allow_pickle=False) as f:
        times=f['sample_times'];features=f['features'];rawframes=f['raw_source_frames']
    with np.load(curve,allow_pickle=False) as f:
        ct=f['sample_times'];human=f['human'];robot=f['robot'];labels=[str(x) for x in f['labels']]
    hr=json.loads((HUMAN/sid/'human_report.json').read_text());rate=float(hr['reference_fps'])
    if (not np.array_equal(ct,times) or len(features)+1!=len(times) or labels[5]!='right ankle lateral' or
        labels[9]!='right ankle height' or np.any(np.diff(rawframes)!=1)):raise ValueError('curve or source support mismatch')
    frames=times*rate
    if frames[0]<rawframes[0]-1e-7 or frames[-1]>rawframes[-1]+1e-7:raise ValueError('source support exceeded')
    events={}
    for name,data in [('human',human),('robot',robot)]:
        lat=data[:,5];height=data[:,9];peak=int(np.argmax(lat))
        events[name]={'lateral_peak_index':peak,'lateral_from_start_m':float(lat[peak]-lat[0]),
            'lateral_end_minus_start_m':float(lat[-1]-lat[0]),'height_rise_m':float(np.max(height)-height[0]),
            'height_end_minus_start_m':float(height[-1]-height[0])}
        if not (events[name]['lateral_from_start_m']>.15 and events[name]['height_rise_m']>.15 and
                abs(events[name]['lateral_end_minus_start_m'])<.1 and abs(events[name]['height_end_minus_start_m'])<.1):
            raise ValueError('right-leg description unsupported by observed curves')
    caption='a person raises their right leg to the side and lowers it.'
    annotation={'sample_id':sid,'caption':caption,'caption_origin':'manual visual annotation from all-sample kinematic curves; not provided by the source dataset',
        'derived_from_source_caption':rec['captions'][1]['caption'],'reviewer':'Codex: inspected human/robot all-sample right-ankle curves',
        'feature_sha256':sha(artifact),'curve_sha256':sha(curve),'figure_sha256':sha(figure),
        'feature_rows':len(features),'nominal_time_s':[float(times[0]),float(times[-1])],
        'timeline_mapping':'linear_source_frame_positions','sample_source_frame_positions':frames.tolist(),
        'source_frame_bounds':[int(rawframes[0]),int(rawframes[-1])],'declared_source_fps':rate,
        'motion_events':events,'semantic_supported_for_exported_window':True,'training_ready':False,
        'limitations':['Manual self-label, not original HumanML text; body lateral axis estimated from hips and shoulders.',
          'Project distances are review aids, not paper thresholds; no claim of contact or ground truth.',
          'Fitted human, timing, source identity and physical quality remain unaccepted.']}
    with (HERE/'004028_annotation.json').open('x') as f:json.dump(annotation,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'sample_id':sid,'caption':caption,'events':events,'training_ready':False}))
if __name__=='__main__':main()

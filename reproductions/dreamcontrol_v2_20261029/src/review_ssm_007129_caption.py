"""Record one limited source-caption review from all-sample movement curves.

The selected source sentence is gender-neutral and only describes hand motion;
human/robot wrist keypoints are proxies for hands, not hand-surface evidence.
"""
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
DATA=ROOT/'ssm_new6_20hz_resume_20261101';CURVES=ROOT/'ssm_new5_caption_curves_20261101';HUMAN=ROOT/'ssm_new6_human_20261101'
ZIP=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D/texts.zip')
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    sid='007129';datareport=json.loads((DATA/'report.json').read_text());audit=json.loads((DATA/'audit.json').read_text())
    chart=json.loads((CURVES/'report.json').read_text())
    if (sha(DATA/'report.json')!=audit['source_report_sha256'] or sha(DATA/'records.jsonl')!=datareport['records_sha256'] or
        chart['input_report_sha256']!=sha(DATA/'report.json') or chart['input_audit_sha256']!=sha(DATA/'audit.json')):
        raise ValueError('review inputs changed')
    rec=next(x for x in map(json.loads,(DATA/'records.jsonl').read_text().splitlines()) if x['id']==sid)
    ch=next(x for x in chart['cases'] if x['id']==sid);artifact=DATA/rec['artifact'];curve=CURVES/(sid+'_curves.npz');figure=CURVES/(sid+'.png')
    if sha(artifact)!=rec['artifact_sha256'] or sha(curve)!=ch['curves_sha256'] or sha(figure)!=ch['image_sha256']:
        raise ValueError('motion or curves changed')
    with zipfile.ZipFile(ZIP) as z:raw=z.read('texts/'+sid+'.txt')
    with np.load(artifact,allow_pickle=False) as f:times=f['sample_times'];features=f['features']
    with np.load(curve,allow_pickle=False) as f:
        ct=f['sample_times'];human=f['human'];robot=f['robot'];labels=[str(x) for x in f['labels']]
    if not np.array_equal(ct,times) or labels[:2]!=['left wrist height','right wrist height'] or len(features)+1!=len(times):
        raise ValueError('curve coverage mismatch')
    line=raw.decode('utf8').splitlines()[2].split('#');caption=rec['captions'][2]
    if len(line)!=4 or line[0]!='a person moves hand up and down' or line[0]!=caption['caption'] or line[2:]!=['0.0','0.0']:
        raise ValueError('original selected caption differs')
    assoc=caption['association']
    if assoc['status']!='associated' or assoc['matching_feature_intervals']!=list(range(len(features))):
        raise ValueError('caption not paired to full exported range')
    peaks={};ranges={}
    for label,arr in (('human',human),('robot',robot)):
        for side,col in (('left',0),('right',1)):
            key=label+'_'+side;values=arr[:,col];ranges[key]=float(np.ptp(values))
            peaks[key]=find_peaks(values,prominence=.1)[0].tolist()
            if ranges[key]<.4 or len(peaks[key])<2:raise ValueError('expected up-and-down movement unsupported')
    sample=[float(times[0]),float(times[-1])];source=[assoc['annotation_start_time'],assoc['annotation_end_time']]
    if not source[0]<=sample[0]<sample[1]<=source[1]+1e-8:raise ValueError('sample outside nominal caption range')
    receipt={'sample_id':sid,'caption':line[0],'selected_line':2,'tags':line[2:],
        'caption_archive_sha256':sha(ZIP),'caption_member_sha256':hashlib.sha256(raw).hexdigest(),
        'candidate_sha256':sha(artifact),'curve_sha256':sha(curve),'curve_report_sha256':sha(CURVES/'report.json'),
        'figure_sha256':sha(figure),'sample_range_s':sample,'source_range_s':source,
        'wrist_height_ranges_m':ranges,'prominent_wrist_height_peak_indices':peaks,'prominence_m':.1,
        'manual_semantic_assessment':{'status':'supported_for_this_exported_motion',
            'reviewer':'Codex: inspected all-sample human/robot wrist-height curves',
            'reason':'Both wrist keypoints repeatedly move vertically relative to their shoulders in fitted human and G1; wrist points are hand proxies.',
            'semantic_accepted_for_exported_samples':True},'training_ready':False,
        'limitations':['Kinematic-curve review, not full hand geometry or physical interaction.',
            'Prominence is an explanatory project threshold, not a paper filter.',
            'Nominal text/time, true clock, source identity, physical quality and license are not established.']}
    with (HERE/'007129_review.json').open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'sample_id':sid,'caption':line[0],'wrist_ranges_m':ranges,'peaks':peaks,'training_ready':False}))
if __name__=='__main__':main()

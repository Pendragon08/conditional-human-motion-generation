"""Audit hard/smooth sole-min variants on identical S1 sample windows."""
import hashlib,json,math,re
from pathlib import Path
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HARD=ROOT/'pyroki_s1_sole_w3_pilot_20261030';SMOOTH=ROOT/'pyroki_s1_sole_smooth_pilot_20261030'
HVAR=ROOT/'pyroki_sole_w3_variant_20261030';SVAR=ROOT/'pyroki_sole_smooth_variant_20261030'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def change(s,old,new):
    if s.count(old)!=1:raise ValueError('unexpected edit occurrence')
    return s.replace(old,new)
def stops(path):
    segments=path.read_text().split('Terminated @ iteration #')
    if len(segments)!=6:raise ValueError('missing termination records')
    out=[]
    for i in range(5):
        before=segments[i];after=segments[i+1]
        header=re.match(r'(\d+): cost=([0-9.eE+-]+) criteria=\[([01]) ([01]) ([01])\]',after)
        steps=re.findall(r'step #(\d+): cost=[0-9.eE+-]+ lambd=([0-9.eE+-]+)',before)
        accepted=re.findall(r'accepted=(True|False)',before)
        if not header or not steps or not accepted:raise ValueError('unparseable solver log')
        flags=[int(header.group(k)) for k in (3,4,5)];damping=float(steps[-1][1]);last=accepted[-1]=='True'
        out.append({'window':i,'iterations':int(header.group(1)),'flags':flags,'last_damping':damping,'last_accepted':last,
                    'all_flags_false':not any(flags),'last_rejected_at_max_damping':bool(not last and damping>=1e6)})
    return out

def main():
    hm=json.loads((HARD/'manifest.json').read_text());sm=json.loads((SMOOTH/'manifest.json').read_text())
    hr=json.loads((HVAR/'report.json').read_text());sr=json.loads((SVAR/'report.json').read_text())
    hh=HVAR/'retarget_helpers/_utils.py';sh=SVAR/'retarget_helpers/_utils.py'
    if sha(hh)!=hm['helper_sha256'] or sha(hh)!=sm['old_helper_sha256'] or sha(sh)!=sm['new_helper_sha256'] or sha(SVAR/'report.json')!=sm['new_variant_sha256']:raise ValueError('helper manifest mismatch')
    if hr['weights']!=sr['weights'] or hr['weights']['sole_contact']!=3. or hr['urdf_sha256']!=sr['urdf_sha256']:raise ValueError('weights or URDF differ')
    expected=change(hh.read_text(),'import jax\n','import jax\nimport jax.scipy.special as jsp\n')
    for side,idx in (('left',0),('right',1)):
        expected=change(expected,f'{side}_clearance = jnp.min({side}_centers[:, 2] - sole_radii[{idx}]) - {side}_foot_keypoints[2]',f'{side}_clearance = -0.002 * jsp.logsumexp(-({side}_centers[:, 2] - sole_radii[{idx}]) / 0.002) - {side}_foot_keypoints[2]')
    if expected!=sh.read_text() or abs(sr['smooth_bias_upper_bound_m']-.002*math.log(4))>1e-15:raise ValueError('unexpected helper change or bound')
    comparisons={};stopping={};inputs={}
    for label,path in (('hard',HARD),('smooth',SMOOTH)):
        report=json.loads((path/'report.json').read_text());audit=json.loads((path/'audit_report.json').read_text());comp=json.loads((path/'comparison.json').read_text())
        if sha(path/'report.json')!=audit['report_sha256'] or sha(path/'windows.jsonl')!=report['rows_sha256'] or sha(path/'audit_report.json')!=comp['variant_audit_sha256'] or sha(path/'report.json')!=comp['variant_report_sha256'] or audit['windows_verified']!=5:raise ValueError('trial provenance mismatch')
        comparisons[label]=comp;stopping[label]=stops(path/'solver.log');inputs[label]={'report_sha256':sha(path/'report.json'),'audit_sha256':sha(path/'audit_report.json'),'comparison_sha256':sha(path/'comparison.json'),'solver_log_sha256':sha(path/'solver.log')}
    if comparisons['hard']['source_sha256']!=comparisons['smooth']['source_sha256'] or comparisons['hard']['mean_of_window_metrics']['baseline']!=comparisons['smooth']['mean_of_window_metrics']['baseline']:raise ValueError('different input or baseline')
    if [x['frames'] for x in comparisons['hard']['windows']]!=[x['frames'] for x in comparisons['smooth']['windows']]:raise ValueError('different frame selections')
    out={'script_sha256':sha(Path(__file__)),'source_sha256':comparisons['hard']['source_sha256'],'inputs':inputs,
         'temperature_m':.002,'smooth_bias_upper_bound_m':.002*math.log(4),'sole_weight':3.,'helper_only_uses_expected_changes':True,
         'mean_of_window_metrics':{n:c['mean_of_window_metrics']['sole_variant'] for n,c in comparisons.items()},
         'max_overlap_disagreement':{n:c['max_overlap_disagreement']['sole_variant'] for n,c in comparisons.items()},'solver_stops':stopping,
         'warning':'5 overlapping S1 tuning windows, assumed z=0 and low-speed mask, different nonsmooth/smooth objectives, no independent sources, real contact or dynamics. Solver flags do not prove global convergence; smooth-min bias bound addresses only four sphere clearances.'}
    with Path(__file__).with_name('report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({'metrics':out['mean_of_window_metrics'],'seams':out['max_overlap_disagreement'],
                      'flags_false':{k:sum(x['all_flags_false'] for x in v) for k,v in stopping.items()},
                      'rejected_at_max':{k:sum(x['last_rejected_at_max_damping'] for x in v) for k,v in stopping.items()}}),flush=True)
if __name__=='__main__':main()

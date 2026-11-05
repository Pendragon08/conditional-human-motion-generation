"""Extract solver stopping evidence; reported flags are not physical validation."""
import hashlib,json,re
from pathlib import Path
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
CASES={'baseline':'pyroki_grab_s1_pour_windowed_20261030','sole10':'pyroki_s1_sole_pilot_20261030','sole3':'pyroki_s1_sole_w3_pilot_20261030'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    rows={};hashes={}
    for label,dirname in CASES.items():
        path=ROOT/dirname/'solver.log';hashes[label]=sha(path);text=path.read_text();segments=text.split('Terminated @ iteration #')
        expected=29 if label=='baseline' else 5
        if len(segments)-1!=expected:raise ValueError('missing log termination records')
        observations=[]
        for i in range(5):
            before=segments[i];after=segments[i+1]
            head=re.match(r'(\d+): cost=([0-9.eE+-]+) criteria=\[([01]) ([01]) ([01])\]',after)
            steps=re.findall(r'step #(\d+): cost=[0-9.eE+-]+ lambd=([0-9.eE+-]+)',before)
            accepted=re.findall(r'accepted=(True|False)',before)
            if not head or not steps or not accepted:raise ValueError('unparseable termination')
            flags=[int(head.group(i)) for i in (3,4,5)];damping=float(steps[-1][1]);last_accepted=accepted[-1]=='True'
            observations.append({'window':i,'iterations':int(head.group(1)),'criterion_flags':flags,'last_recorded_damping':damping,
                                 'last_step_accepted':last_accepted,'all_criteria_false':not any(flags),
                                 'last_step_rejected_at_1e6_damping':bool(not last_accepted and damping>=1e6)})
        rows[label]=observations
    solver=Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/jaxls/src/jaxls/_py310/_solvers.py')
    out={'script_sha256':sha(Path(__file__)),'jaxls_solver_sha256':sha(solver),'solver_log_sha256':hashes,'results':rows,
         'warning':'Extracted last logged damping/acceptance and termination flags. According to this JAXLS source a rejected step at max damping can terminate the loop; logs alone are not a complete optimizer proof. One S1 clip, 5 overlapping windows per variant; no ground truth or training acceptance.'}
    with Path(__file__).with_name('report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({k:{'all_flags_false':sum(x['all_criteria_false'] for x in v),'last_rejected_at_max':sum(x['last_step_rejected_at_1e6_damping'] for x in v)} for k,v in rows.items()}),flush=True)
if __name__=='__main__':main()

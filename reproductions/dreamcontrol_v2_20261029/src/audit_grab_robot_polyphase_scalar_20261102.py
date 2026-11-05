"""Direct-convolution witness for two saved GRAB robot resampling previews.

Checks numerical filtering only; it does not validate the filter choice,
underlying solver, true clock, physical interaction or model behavior.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import firwin
from blend_robot_windows import blend_windows

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
WORK=ROOT/'grab_full_robot_work_20261102'
PREVIEW=ROOT/'grab_full_robot_263_preview_20261102'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def direct(state):
    x=np.asarray(state,dtype=np.float64).copy()
    if x.ndim!=2 or x.shape[1]!=34 or len(x)<7:raise ValueError('bad state')
    # Sign-continuity is applied sequentially, independently of the vectorized implementation.
    for i in range(1,len(x)):
        if np.dot(x[i-1,:4],x[i,:4])<0:x[i,:4]*=-1
    slope=(x[-1]-x[0])/(len(x)-1)
    residual=x-(x[0]+np.arange(len(x))[:,None]*slope)
    weights=firwin(121,1/6,window=('kaiser',5.0))
    count=(len(x)-1)//6+1
    positions=60+np.arange(count)*6
    result=np.empty((count,34),dtype=np.float64)
    for column in range(34):
        result[:,column]=np.convolve(residual[:,column],weights,mode='full')[positions]
    result+=x[0]+(np.arange(count)*6)[:,None]*slope
    result[0]=x[0]
    if (count-1)*6==len(x)-1:result[-1]=x[-1]
    norms=np.linalg.norm(result[:,:4],axis=1)
    if np.any(norms<1e-8):raise ValueError('degenerate quaternion')
    result[:,:4]/=norms[:,None]
    return result


def main():
    report=json.loads((PREVIEW/'report.json').read_text())
    audit_path=WORK/'robot_windows_audit_20261102.json'
    if sha(audit_path)!=report['window_audit_sha256']:raise ValueError('audit binding mismatch')
    results=[]
    for case in report['cases']:
        if case['status']!='technical_preview':continue
        sid=case['id'];directory=WORK/'robot_windows'/sid;record=json.loads((directory/'report.json').read_text())
        if sha(directory/'report.json')!=case['solver_report_sha256'] or sha(directory/'windows.jsonl')!=record['rows_sha256']:
            raise ValueError('solver bindings differ')
        windows=[]
        for row in map(json.loads,(directory/'windows.jsonl').read_text().splitlines()):
            file=directory/row['artifact']
            if sha(file)!=row['sha256']:raise ValueError('window hash mismatch')
            with np.load(file,allow_pickle=False) as z:
                windows.append({'frames':z['sample_indices'].copy(),'root_wxyz_xyz':z['root_wxyz_xyz'].copy(),
                                'joint_pos':z['joint_pos'].copy(),'joint_names':z['joint_names'].copy()})
        blended=blend_windows(windows)['blended'];expected=direct(blended)
        preview=PREVIEW/case['artifact']
        if sha(preview)!=case['output_sha256']:raise ValueError('preview hash mismatch')
        with np.load(preview,allow_pickle=False) as z:actual=np.concatenate((z['root_wxyz_xyz'],z['joint_pos']),axis=1)
        if actual.shape!=expected.shape:raise ValueError('resampled shape mismatch')
        error=float(np.max(np.abs(actual-expected)))
        if not np.isfinite(error) or error>1e-9:raise ValueError(f'direct convolution mismatch: {error}')
        results.append({'id':sid,'source_states':len(blended),'output_states':len(actual),
                        'max_abs_state_difference':error,'output_sha256':sha(preview)})
    out={'cases':results,'preview_report_sha256':sha(PREVIEW/'report.json'),'window_audit_sha256':sha(audit_path),
         'script_sha256':sha(Path(__file__)),'training_ready':False,
         'limitations':['Shares the blended source windows and FIR coefficients; does not independently reproduce solver or blending.',
                         'Component-wise quaternion FIR is an approximation, not a validated SO(3) filter.',
                         'Object resampling, actual source clock, physical quality and text are not accepted.']}
    with (PREVIEW/'polyphase_direct_audit.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({'cases':len(results),'max_abs_state_difference':max(x['max_abs_state_difference'] for x in results),
                      'training_ready':False}))


if __name__=='__main__':main()

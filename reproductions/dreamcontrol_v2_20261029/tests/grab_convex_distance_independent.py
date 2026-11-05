"""Independent convex-set distance brackets using support points and feasible witnesses.

No MuJoCo query is used in the independent calculation. A feasible convex
combination supplies an upper bound; a separating support plane supplies a
lower bound. The result remains a numerical diagnostic, not a physics test.
"""
import json, sys, zipfile
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
sys.path.insert(0,str(ROOT/'grab_s3_scene_timeline_v2_20261030'))
from grab_s3_scene_timeline import sha,load_table,mesh_for
sys.path.insert(0,str(ROOT/'grab_s3_human_source_20261030'))
from grab_restricted_reader import read_grab_field

def support(a,b,x):
    ia=int(np.argmin(a@x));ib=int(np.argmax(b@x))
    return a[ia]-b[ib],(ia,ib)

def bracket(a,b,tol=1e-5,max_iter=160):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=3 or b.shape[1]!=3:raise ValueError('invalid vertices')
    x=a[0]-b[0];points=[x.copy()];weights=np.array([1.]);keys=set();status='iteration_limit';last=None
    for it in range(max_iter):
        upper=float(np.linalg.norm(x))
        if upper<tol:
            return {'lower_m':0.,'upper_m':upper,'gap_m':upper,'iterations':it,'status':'bounded'}
        w,key=support(a,b,x)
        lower=max(0.,float(np.dot(x,w)/upper));gap=max(0.,upper-lower)
        if lower>upper+1e-9:raise ValueError('invalid support bounds')
        last={'lower_m':lower,'upper_m':upper,'gap_m':gap,'iterations':it,'status':'bounded' if gap<=tol else 'running'}
        if gap<=tol:return last
        if key in keys:
            status='repeated_support';break
        keys.add(key);points.append(w);weights=np.r_[weights,0.]
        mat=np.array(points); gram=mat@mat.T
        result=minimize(lambda q:float(q@gram@q),weights,jac=lambda q:2*gram@q,
                        method='SLSQP',bounds=[(0.,1.)]*len(weights),
                        constraints={'type':'eq','fun':lambda q:float(q.sum()-1.),'jac':lambda q:np.ones_like(q)},
                        options={'ftol':1e-14,'maxiter':150,'disp':False})
        q=np.clip(result.x,0.,1.);q/=q.sum();keep=q>1e-10
        q=q[keep];q/=q.sum();mat=mat[keep];points=list(mat);weights=q;x=q@mat
        if not np.isfinite(x).all():raise ValueError('nonfinite optimizer')
    if last is None:raise ValueError('empty solve')
    last['status']=status
    return last

def main():
    # Axis-aligned boxes separated by exactly 0.5 m.
    box=np.array([[x,y,z] for x in (-.1,.1) for y in (-.1,.1) for z in (-.1,.1)])
    synthetic=bracket(box,box+np.array([.7,0.,0.]))
    if not (synthetic['lower_m']-1e-8<=.5<=synthetic['upper_m']+1e-8 and synthetic['gap_m']<=1e-5):
        raise ValueError('synthetic exact box test failed')
    meta=json.loads((ROOT/'grab_s3_human_source_20261030/report.json').read_text());archive=ROOT/'data/GRAB/grab__s3.zip'
    if sha(archive)!=meta['archive_sha256']:raise ValueError('archive changed')
    with zipfile.ZipFile(archive) as z:obj=read_grab_field(z,meta['source_member'],'object')
    table=load_table(archive,meta['source_member']);ap,ao,ah=mesh_for(obj['object_mesh']);tp,to,th=mesh_for(table['table_mesh'])
    v=np.asarray(ah.vertices);count=512;i=np.arange(count);z=1-2*(i+.5)/count;theta=i*np.pi*(3-np.sqrt(5))
    dirs=np.column_stack((np.sqrt(1-z*z)*np.cos(theta),np.sqrt(1-z*z)*np.sin(theta),z))
    chosen=[int(np.argmax(v@d)) for d in dirs]
    small=trimesh.Trimesh(vertices=v[np.unique(chosen)],process=False).convex_hull;sv=np.asarray(small.vertices)
    # Every vertex of the inner hull comes from the original convex hull.
    # For every full-hull vertex choose a nearest inner-hull vertex; the
    # maximum such distance is a conservative upper bound on one-sided
    # Hausdorff distance for the two convex polytopes.
    if float(cKDTree(v).query(sv)[0].max())>1e-10:raise ValueError('approximation not inscribed')
    hausdorff_upper=float(cKDTree(sv).query(v)[0].max())
    previous=json.loads((ROOT/'grab_s3_distance_solver_audit_v2_20261030/report.json').read_text())
    prior={(r['frame'],r['mesh'],r['solver']):r['distance_m'] for r in previous['results']}
    frames=sorted(set(r['frame'] for r in previous['results']));results=[]
    for frame in frames:
        rot_obj=Rotation.from_rotvec(np.asarray(obj['params']['global_orient'])[frame]).inv()
        trans_obj=np.asarray(obj['params']['transl'])[frame]
        rot_table=Rotation.from_rotvec(np.asarray(table['params']['global_orient'])[frame]).inv()
        trans_table=np.asarray(table['params']['transl'])[frame]
        bw=rot_table.apply(th.vertices)+trans_table
        for label,av in (('full',v),('inscribed_support_512',sv)):
            aw=rot_obj.apply(av)+trans_obj;answer=bracket(aw,bw)
            row={'frame':frame,'mesh':label,'independent':answer}
            for mode in ('native','libccd'):
                d=prior[(frame,label,mode)];row['mujoco_'+mode+'_m']=d
                row[mode+'_outside_bracket']=bool(d<answer['lower_m']-1e-5 or d>answer['upper_m']+1e-5)
            results.append(row)
    report={'script_sha256':sha(Path(__file__)),'archive_sha256':sha(archive),'apple_sha256':sha(ap),'table_sha256':sha(tp),
            'prior_report_sha256':sha(ROOT/'grab_s3_distance_solver_audit_v2_20261030/report.json'),
            'synthetic_box_exact_m':.5,'synthetic_box_bracket':synthetic,'full_vertices':len(v),'inner_vertices':len(sv),
            'hausdorff_conservative_upper_bound_m':hausdorff_upper,'results':results,
            'warning':'Independent convex hull distances bracketed numerically for 15 selected static frames; not true nonconvex mesh distances, contact forces, dynamics or grasp success. Hausdorff bound based on nearest inner vertices is conservative; numerical tolerance and optimizer failures must be inspected.'}
    with Path(__file__).with_name('report.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({'rows':len(results),'not_bounded':sum(r['independent']['status']!='bounded' for r in results),
                      'max_gap':max(r['independent']['gap_m'] for r in results),'hausdorff_upper_m':hausdorff_upper,
                      'native_outside':sum(r['native_outside_bracket'] for r in results),'libccd_outside':sum(r['libccd_outside_bracket'] for r in results)}),flush=True)
if __name__=='__main__':main()

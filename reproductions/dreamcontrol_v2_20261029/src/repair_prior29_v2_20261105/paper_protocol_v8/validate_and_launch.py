"""Validate preserved Pour raw under explicit paper rules before new sampling."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

S = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/repair_prior29_v2_20261105')
HERE = Path(__file__).resolve().parent
OUT = S/'paper_protocol_validation_v7'
PLAN = S/'paper_pipeline_v7_launch/config.json'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())

def write(p,x):
    p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)

def prepare():
    if (OUT/'launch.json').exists():
        print((OUT/'launch.json').read_text());return
    assert not OUT.exists(), 'preserve partial setup; inspect before resuming'
    plan=read(S/'generation_pipeline_v6_launch/config.json')
    for spec in plan['sources'].values():assert sha(spec['path'])==spec['sha256']
    for key in ['controller','sample','postopt','runtime','guidance','contracts','validation','conditions','train']:
        name='pipeline_controller_v2.py' if key=='controller' else ('postopt_v2.py' if key=='postopt' else key+'.py')
        file=HERE/name
        plan['sources'][key]={'path':str(file),'sha256':sha(file)}
    for key in ['native_pick','native_supported','paper_conditions']:
        file=HERE/(key+'.py');plan['sources'][key]={'path':str(file),'sha256':sha(file)}
    plan['sources'].pop('postopt_pick',None)
    plan.pop('pick_seed_audit',None);plan.pop('resume_pick_cohorts',None)
    plan['reused_pick_raw']={}
    plan['output_dir']=str(S/'new_references_paper_v7')
    plan['seed']=20261106
    plan['user_authorization']='paper documented filters; no added raw-fidelity/correction windows; existing validation then fresh generation'
    # Native entrypoints depend on the existing, hash-bound public adapter modules.
    native=read(plan['native_runtime']['path'])
    plan['native_import_dir']=str(Path(plan['sources']['native_pick_scene_env']['path']).parent)
    OUT.mkdir();PLAN.parent.mkdir(exist_ok=False)
    write(PLAN,plan)
    env={**os.environ,**plan['postopt_env'],'PYTHONDONTWRITEBYTECODE':'1'}
    with (OUT/'controller.log').open('xb') as f:
        proc=subprocess.Popen([plan['postopt_python'],'-B','-u',str(Path(__file__).resolve()),'--worker'],
             cwd=S,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    launch={'pid':proc.pid,'plan_sha256':sha(PLAN),'source_sha256':sha(__file__)}
    write(OUT/'launch.json',launch);print(json.dumps(launch))

def cpu_checks(plan):
    for spec in plan['sources'].values():assert sha(spec['path'])==spec['sha256']
    sys.path.insert(1,str(Path(plan['sources']['task_conditions']['path']).parent))
    sys.path.insert(1,plan['native_import_dir'])
    sys.path.insert(1,str(Path(plan['fk']['path']).parent))
    import paper_conditions as c
    from contracts import coordinate_contract, world_to_canonical, canonical_to_world
    from postopt_v2 import load_sample,acceptance_checks
    from sample import standing_template
    from g1_fk import URDFFK
    from stage3_pick29_core import BODY_JOINTS,SCENE,SCENE_SHA
    initial,rotations=standing_template(URDFFK(plan['urdf']['path']),BODY_JOINTS)
    transform=coordinate_contract(initial,BODY_JOINTS,plan['urdf']['sha256'],SCENE['ground_z'])
    checks={}
    for task in ['pick','pour','bimanual_box_lift','box_push_pull']:
        result=c.build(task,list(range(64)),20261106,initial,rotations,SCENE,SCENE_SHA)
        world=result['world'];mask=result['mask']
        assert world.shape[:2]==mask.shape[:2] and np.isfinite(world).all()
        assert np.max(np.abs(canonical_to_world(world_to_canonical(world,transform),transform)-world))<1e-6
        assert np.allclose(np.linalg.norm(result['desired_wrist_quat_wxyz'],axis=-1),1.,atol=1e-6)
        if task in ['pick','pour']:
            assert mask[:,:,19].all() and mask[:,:,21].all() and mask[:,::5,4].all() and mask[:,::5,5].all()
            for b,frame in enumerate(result['grasp_frames']):
                assert 40<=frame<=(50 if task=='pick' else 42)
                x,y,z=result['plans'][b]['paper_control_position']
                assert (-.22<=x<=.12 if task=='pick' else -.1<=x<=.1) and .2<=y<=.4 and z==.94
                assert np.allclose(world[b,frame,21], [y,-x,.94])
                elbow=world[b,frame,19]-world[b,frame,21]
                length=np.linalg.norm(initial[21]-initial[19])
                assert np.allclose(elbow,[0,length/np.sqrt(2),-length/np.sqrt(2)],atol=1e-6)
        checks[task]={'finite':True,'coordinate_roundtrip':True,'planned_quaternion_unit':True,'candidates':64,
                      'constraint_points':np.flatnonzero(mask[0].any(0)).tolist()}
    # Regression: prior extra limits are measurements only; exact paper limits reject.
    m=dict(finite=True,joint_violation_rad=0.,wrist_max_l2_m=.1,
           minimum_active_wrist_table_sdf_m=.04,minimum_mesh_height_above_ground_m=0.,
           minimum_table_vertex_sdf_m=0.,decoded_point_vector_rmse_m=.2,
           stage_b_joint_max_shift_rad=1.,wrist_max_angle_rad=.3)
    assert all(acceptance_checks(m).values())
    assert not all(acceptance_checks({**m,'wrist_max_l2_m':.15}).values())
    assert not all(acceptance_checks({**m,'minimum_active_wrist_table_sdf_m':.029}).values())
    write(OUT/'cpu_protocol_checks.json',{'status':'passed','tasks':checks,'paper_filter_regression':True,
          'paper_frame_adapter':'explicit proper rotation X=paper_y,Y=-paper_x; same rotation applied to elbow offsets',
          'not_claimed':'exact paper scene transform, undisclosed alpha_l, angle schedule, or Full-Mix reproduction'})

def work():
    import pipeline_controller_v2 as p
    import paper_conditions as c
    from postopt_v2 import load_sample
    plan,cfg,_=p.setup_plan(PLAN)
    try:
        write(OUT/'status.json',{'status':'cpu_protocol_validation','new_samples':0})
        cpu_checks(plan)
        prior_path=S/'pour_existing_contrast_v3/postopt/candidate_10100005/report.json'
        assert sha(prior_path)=='905639e8e1aa7e51d80cc49860299f9b85ad6874db84d12db9461966e28f2687'
        prior=read(prior_path);original=Path(prior['source_sample'])
        assert sha(original)==prior['source_sample_sha256']
        with np.load(original,allow_pickle=False) as z: values=c.augment_existing_sample(dict(z))
        meta=json.loads(values['metadata_utf8'].tobytes())
        state=prior['stages'][2]
        assert sha(state['path'])==state['sha256']
        meta.update(original_sample_path=str(original),original_sample_sha256=sha(original),
                    validation_initial_state={'path':state['path'],'sha256':state['sha256']},
                    actual_generation_task_conditions_sha256=meta['task_conditions_source_sha256'],
                    task_conditions_source_sha256=sha(HERE/'paper_conditions.py'))
        values['metadata_utf8']=np.frombuffer(json.dumps(meta).encode(),dtype=np.uint8)
        derived=OUT/'existing_10100005_elbow_linked.npz'
        with derived.open('xb') as f:np.savez_compressed(f,**values)
        sample=load_sample(derived,sha(derived))
        with np.load(original,allow_pickle=False) as z:
            assert np.array_equal(values['features'],z['features']) and np.array_equal(values['canonical_points'],z['canonical_points'])
        write(OUT/'existing_source_receipt.json',{'original_path':str(original),'original_sha256':sha(original),
              'derived_path':str(derived),'derived_sha256':sha(derived),'actual_diffusion_elbow_hint':False,
              'new_generated_references':0,'feature_and_decoded_arrays_unchanged':True})
        out=OUT/'postopt/candidate_10100005'
        argv=[plan['postopt_python'],'-B','-u',str(p.bound(plan['sources']['postopt'])),
              '--sample',str(derived),'--sample-sha256',sha(derived),'--output-dir',str(out),
              '--urdf',plan['urdf']['path'],'--fk-module',plan['fk']['path']]
        try:p.phase(PLAN,sha(PLAN),plan,cfg,OUT,OUT,'postopt',argv,os.environ.copy(),str(OUT))
        except ValueError:
            if not (out/'report.json').exists():raise
        report=read(out/'report.json')
        if not report['passed_requested_kinematic_checks']:
            write(OUT/'status.json',{'status':'existing_validation_rejected','report_sha256':sha(out/'report.json'),
                    'new_generated_references':0});return
        cohort=S/'new_references_v6/pour/cohort_0005/raw'
        rows=p.producer_records(cohort,'pour',[10100005],report['checkpoint_sha256'])
        row=rows[0];row['source_sample']=str(derived);row['source_sample_sha256']=sha(derived)
        row['task_conditions_source_sha256']=sha(HERE/'paper_conditions.py')
        staged=p.stage_native(plan,'pour',OUT/'postopt',OUT/'native_input',rows)
        assert len(staged)==1
        native=read(p.bound(plan['native_runtime']));script=p.bound(plan['sources']['native_supported'])
        env=p.runtime_env(native['env'],remove_visibility=True)
        env['PYTHONPATH']=plan['native_import_dir']+os.pathsep+str(script.parent)+os.pathsep+env.get('PYTHONPATH','')
        native_out=OUT/'native'
        argv=[native['python'],'-B','-u',str(script),'--headless','--device',plan['native_device'],
              '--output-dir',str(native_out),'--task','pour','--candidate-dir',str(staged[0][1])]
        p.phase(PLAN,sha(PLAN),plan,cfg,OUT,OUT,'native',argv,env,native['cwd'],(native_out/'report.json','native_task_replay_completed'))
        p.native_proofs('pour',staged,native_out)
        passed=bool(rows[0]['admitted'])
        result={'status':'existing_sample_validation_passed' if passed else 'existing_native_rejected',
                'new_generated_references':0,'existing_candidate_id':10100005,'records':rows,
                'kinematic_report_sha256':sha(out/'report.json'),'native_report_sha256':sha(native_out/'report.json')}
        write(OUT/'result.json',result);write(OUT/'status.json',result)
        if passed:
            p.setup_plan(PLAN)
            target=PLAN.parent/'generation.log'
            with target.open('xb') as f:
                env={**os.environ,'PYTHONPATH':str(HERE)+os.pathsep+os.environ.get('PYTHONPATH','')}
                proc=subprocess.Popen([plan['sample_python'],'-B','-u',str(p.bound(plan['sources']['controller'])),
                     '--plan',str(PLAN),'--plan-sha256',sha(PLAN)],cwd=S,env=env,stdin=subprocess.DEVNULL,
                     stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            write(PLAN.parent/'launch.json',{'pid':proc.pid,'plan_sha256':sha(PLAN),
                  'existing_sample_validation':str(OUT/'result.json'),'existing_sample_validation_sha256':sha(OUT/'result.json')})
    except BaseException:
        (OUT/'traceback.txt').write_text(traceback.format_exc())
        write(OUT/'status.json',{'status':'failed_preserved','new_generated_references':0})
        raise

if __name__=='__main__':
    work() if sys.argv[1:]==['--worker'] else prepare()

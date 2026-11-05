"""CPU-only audit for adapter sources; does not instantiate Isaac Lab."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
STAGE=ROOT/'isaac_live_actor_adapter_v2_20261030'
OLD=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/source/isaaclab_tasks/isaaclab_tasks/utils/motion_lib/motion_lib_base.py')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def const_assignment(path,name):
    tree=ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('missing '+name)

def main():
    source=STAGE/'src';adapter=source/'isaac_live_actor.py';config=source/'isaac_live_pick_cfg.py';test=STAGE/'tests/test_isaac_live_actor.py'
    joints=const_assignment(adapter,'G1_27_JOINT_NAMES');legacy=const_assignment(OLD,'JointNamesOrder')
    if list(joints)!=legacy:raise ValueError('joint order mismatch')
    tree=ast.parse(adapter.read_text());functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='live_object_policy_observation']
    if len(functions)!=1:raise ValueError('missing adapter')
    if any(isinstance(n,ast.Attribute) and n.attr=='motion_lib' for n in ast.walk(functions[0])):raise ValueError('reference access in actor')
    cfg=ast.parse(config.read_text());classes={n.name:n for n in cfg.body if isinstance(n,ast.ClassDef)}
    if not {'LiveObjectPolicyCfg','LiveObjectObservationsCfg','G1PickLiveObjectEnvCfg'}<=classes.keys():raise ValueError('missing config class')
    call=subprocess.run([sys.executable,str(test)],capture_output=True,text=True,check=False)
    if call.returncode:raise RuntimeError('unit tests failed: '+call.stderr)
    output={'joint_order_matches_v1_source':True,'joint_count':len(joints),'policy_observation_width':96,
            'legacy_source_sha256':sha(OLD),'files_sha256':{p.name:sha(p) for p in (adapter,config,test,source/'actor_observation.py',Path(__file__))},
            'cpu_test_returncode':call.returncode,'cpu_test_output':call.stderr,
            'limitations':'Mock data and static source checks only; not an Isaac Lab import, manager resolution, runtime, PPO, v2 paper fidelity or old checkpoint compatibility test. Rewards/resets inherited from v1 may use references. Real simulation actor-reference independence and reward/task validity remain to be verified.'}
    with Path(__file__).with_name('audit_report.json').open('x',encoding='utf-8') as f:json.dump(output,f,indent=2,allow_nan=False)
    print(json.dumps(output),flush=True)

if __name__=='__main__':main()

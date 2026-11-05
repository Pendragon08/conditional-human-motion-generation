"""Static independent verification of source hashes and v1 hand command expressions."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');WITNESS=ROOT/'isaac_action_manager_cpu_witness_v2_20261030';OUT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    report=json.loads((WITNESS/'report.json').read_text())
    for path,expected in report['source_hashes'].items():assert sha(Path(path))==expected
    p=next(Path(p) for p in report['source_hashes'] if p.endswith('motion_tracking_pick_env.py'))
    tree=ast.parse(p.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ActionsCfg')
    actions={n.targets[0].id:n.value for n in cls.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
    def params(name):return {kw.arg:ast.dump(kw.value,include_attributes=False) for kw in actions[name].keywords}
    left=params('left_hand_action');right=params('right_hand_action')
    assert left['open_command_expr']==left['close_command_expr'] and right['open_command_expr']!=right['close_command_expr']
    assert report['terms']==['joint_pos','left_hand_action','right_hand_action'] and report['action_dimensions']==[27,1,1] and report['observation_shape']==[2,96]
    out={'witness_report_sha256':sha(WITNESS/'report.json'),'audit_script_sha256':sha(Path(__file__)),'source_count_hash_checked':len(report['source_hashes']),
         'left_open_close_source_ast_equal':True,'right_open_close_source_ast_equal':False,
         'limitations':'Static AST checks and hashes, no simulator, USD joint list or dynamic behavior. Equality is inherited configuration, not evidence of author intent or v2 fidelity.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(out))
if __name__=='__main__':main()

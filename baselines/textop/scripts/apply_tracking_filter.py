"""Apply the paper's two-stage decisions only with actual evaluator evidence.

No file validation, geometric proxy or author-curated membership is accepted
as a substitute for the required relaxed/strict dynamic rollouts.
"""
import argparse,json,pathlib,hashlib
def apply(evidence):
    h=evidence['protocol']
    assert h['privileged_actor_and_critic'] is True and h['domain_randomization'] is False
    assert h['training_iterations']>=40000
    assert h['relaxed_removed_terminations']==['anchor_pos','ee_body_pos']
    assert h['strict_standard_terminations_restored'] is True
    assert h['failure_average_definition'] and h['strict_rollouts_per_motion']>=1
    for key in ['policy_checkpoint','relaxed_rollout_log','strict_rollout_log']:
        x=evidence[key];p=pathlib.Path(x['path']);assert p.is_file(),key
        digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==x['sha256'],key
    decisions=[]
    for x in evidence['motions']:
        assert x['relaxed_episodes']>0 and 0<=x['relaxed_failure_probability']<=1
        if x['relaxed_failure_probability']>.05:decision='reject_relaxed_failure_rate'
        else:
            assert x['strict_episodes']>=h['strict_rollouts_per_motion']
            decision='retain' if x['strict_failures']==0 else 'reject_strict_failure'
        decisions.append({'motion_id':x['motion_id'],'decision':decision})
    return decisions
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve existing filter evidence')
    decisions=apply(json.loads(a.evidence.read_text()));a.output.write_text(json.dumps({'decisions':decisions,'evidence_file':str(a.evidence)},indent=2))

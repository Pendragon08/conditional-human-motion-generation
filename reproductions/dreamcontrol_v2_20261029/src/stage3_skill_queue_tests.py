"""CPU transition tests for automatic continuation; zero task success is not hidden."""
from stage3_skill_queue import pick_transition,next_after_pick,align_task_states

running={'status':'ppo_running','policy_updates':100}
finished={'status':'ppo_completed','policy_updates':5000}
assert pick_transition(running,None,False,False)=='waiting_for_pick_training'
assert pick_transition({'status':'failed'},None,False,False)=='training_failed_needs_repair'
assert pick_transition(finished,None,False,True)=='completion_integrity_needs_repair'
assert pick_transition(finished,None,True,False)=='waiting_for_pick_cleanup'
assert pick_transition(finished,None,True,True)=='ready_to_evaluate_pick'
assert pick_transition(finished,{'status':'entrypoint_smoke_completed','final_evaluation':False,'num_envs':4},True,True)=='evaluation_integrity_needs_repair'
evaluation={'status':'evaluation_completed','final_evaluation':True,'num_envs':100,'success_ratio':0.}
assert pick_transition(finished,evaluation,True,True)=='archive_pick_outcome_and_prepare_next'
print('QUEUE_TESTS passed=7; no GPU or task launch')
short={'status':'ppo_completed','policy_updates':3000}
assert pick_transition(short,None,True,True,3000)=='ready_to_evaluate_pick'
assert pick_transition(short,evaluation,True,True,3000)=='archive_pick_outcome_and_prepare_next'
assert pick_transition(finished,None,True,True,3000)=='completion_integrity_needs_repair'
print('USER_BUDGET_TESTS passed=3;3000 total, not3000 additional')

spec={'tasks':[{'id':'pick'},{'id':'pick_supermimic'},{'id':'punch'}]}
assert next_after_pick(spec)[0]=='pick_supermimic'
old={'tasks':[{'id':'pick','status':'waiting_for_pick_training','policy_updates':300},
              {'id':'punch','status':'pending_implementation'}]}
aligned=align_task_states(old,spec)
assert [x['id'] for x in aligned]==['pick','pick_supermimic','punch']
assert aligned[0]['policy_updates']==300 and aligned[1]['status']=='pending_implementation'
assert next_after_pick({'tasks':[{'id':'pick'},{'id':'punch'}]})[0]=='punch'
print('SUPERMIMIC_HANDOFF_TESTS passed=4; insertion preserves Pick progress and evaluation requirement')

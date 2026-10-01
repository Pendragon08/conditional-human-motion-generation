"""Bounded initialization diagnostic, not official evaluation or training."""
import r3_dreamcontrol_remaining_queue as q

tag = 'r3_sit_no_camera_init_diagnostic_20261001'
log = q.MON / (tag + '.log')
assert not log.exists(), 'No repeated diagnostic'
model = q.TRAIN / 'logs/rsl_rl/g1/2026-10-01_16-54-27-Sit-a/model_1999.pt'
assert q.digest(model) == '4302e29fe6376452a21fe93351e616fb23ec5bb83206eb015df3b5be4bea85c7'
q.write(q.MON / (tag + '.state'), 'RUNNING diagnostic_only official_entrypoint cameras_disabled expected_timeout=120')
rc = q.execute([str(q.ROOT / 'third_party/isaacsim-4.5.0/python.sh'), '-u',
    'scripts/reinforcement_learning/rsl_rl/play_eval.py', '--task=Isaac-Motion-Tracking-Sit-v0',
    '--device=cuda:1', '--num_envs=1000', '--path=' + str(model), '--headless',
    '--id=' + tag], log, q.environment(), 120)
text = log.read_text(errors='replace')
q.write(q.MON / (tag + '.state'),
    f'STOPPED diagnostic_only exit={rc} policy_load_reached={"Loading model checkpoint from:" in text} '
    f'native_malloc_abort={"malloc(): invalid size" in text} no_score_claim=true')
print((q.MON / (tag + '.state')).read_text(), flush=True)

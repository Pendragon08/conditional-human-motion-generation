"""Record the human's explicit current live-view confirmation, not infer it."""
import json
from pathlib import Path
from r3_world_ssh_transport import ROOT, REMOTE, execute

program = '''
from pathlib import Path
import json,time
r=Path(REMOTE)
p=r/'artifacts/monitoring/r3_world_residual_ppo_local_view_gate_20261007.json'
assert not p.exists(),'Preserve existing confirmation receipt'
frame=json.loads((r/'artifacts/monitoring/r3_i1_live_frame.json').read_text())
active=json.loads((r/'artifacts/monitoring/r3_active_dreamcontrol_training.json').read_text())
data={'user_confirmed_local_stream':True,'source':'actual_isaaclab_simulation',
 'local_url':'http://127.0.0.1:6013/live','transport':'SSH TCP MJPEG, not WebRTC',
 'user_answer':'能看到机器人持续运动',
 'prior_answer':'画面不动或打不开','prior_main_page_passed':False,
 'confirmed_stage':'frozen_policy_visibility_preflight; zero PPO',
 'captured_frame_metadata':frame,'active_stage_metadata':active,
 'recorded_unix':time.time(),'confirmation_origin':'direct human asynchronous reply',
 'formal_launch_still_requires_runtime_and_capacity':True}
p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\\n')
print(json.dumps(data,ensure_ascii=False))
'''.replace('r=Path(REMOTE)', 'r=Path('+repr(REMOTE)+')')
receipt=json.loads(execute(program))
out=ROOT/'results/world_residual_ppo_20261007/raw/local_view_gate.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False))

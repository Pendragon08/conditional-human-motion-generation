"""Create a provisional candidate filter from existing audited joint limits.

Only states whose audited window samples never exceed declared limits are kept
for further review. This intentionally conservative zero-excess filter is a
project decision, not an author v2 threshold or full physics verification.
"""
import hashlib,json,math
from pathlib import Path
R=Path(__file__).resolve().parent.parent
REPORTS={'bmlmovi':'bmlmovi_full_robot_windows_audit_20261102.json',
         'kit_100hz_hypothesis':'kit_dart_robot_windows_audit_20261102.json',
         'grab_000_001':'grab_full_robot_windows_audit_20261102.json',
         'grab_002_009':'grab_full_robot_windows_audit_v2_20261102.json'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 cases=[];urdf=None;hashes={}
 for group,name in REPORTS.items():
  path=R/'reports'/name;before=sha(path);report=json.loads(path.read_text(encoding='utf8'))
  if report['training_ready'] is not False or not isinstance(report.get('cases'),list):raise ValueError('invalid audit')
  if urdf is None:urdf=report['urdf_sha256']
  if report['urdf_sha256']!=urdf:raise ValueError('inconsistent URDF')
  ids=set()
  for c in report['cases']:
   sid=c['id'];num=c['joint_values_outside_declared_limits'];excess=c['max_joint_limit_excess_rad']
   if sid in ids or isinstance(num,bool) or not isinstance(num,int) or num<0 or isinstance(excess,bool) or not isinstance(excess,(float,int)) or not math.isfinite(excess) or excess<0 or (num==0)!=(excess==0):raise ValueError('invalid limit evidence')
   ids.add(sid)
   cases.append({'source_group':group,'id':sid,'status':'excluded_joint_limit' if num else 'retained_for_further_review',
     'values_outside_declared_limits':num,'max_excess_rad':excess,'human_sha256':c['human_sha256'],
     'solver_report_sha256':c['solver_report_sha256'],'window_audit_sha256':before,'caption_accepted':False,'training_ready':False})
  if before!=sha(path):raise RuntimeError('audit report changed')
  hashes[name]=before
 if len(cases)!=38:raise ValueError('unexpected audit coverage')
 result={'cases':cases,'retained':sum(c['status']=='retained_for_further_review' for c in cases),
  'excluded':sum(c['status']=='excluded_joint_limit' for c in cases),'urdf_sha256':urdf,'input_sha256':hashes,
  'training_ready':False,'limitations':['Strict zero-excess check can exclude tiny numerical overshoot; source windows are preserved.',
   'Based on prior window-audit receipts, not recomputation; no dynamics, contact, source clock or caption validation.',
   'Retained candidates still require data, time, text, quality and resource checks before training.']}
 out=R/'reports/robot_window_limit_screen_20261103.json'
 with out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
 print(json.dumps({'screened':len(cases),'retained':result['retained'],'excluded':result['excluded'],'training_ready':False}))
if __name__=='__main__':main()

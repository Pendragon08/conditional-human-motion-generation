"""Synthetic protocol regression checks, explicitly not physical experiments."""
import copy,hashlib,json,pathlib,sys,tempfile,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
from strict_tracking_coverage import StrictCoverage,verify_trial
from apply_tracking_filter import apply

class CoverageTests(unittest.TestCase):
 def test_entire_motion(self):
  c=StrictCoverage(4)
  c.observe();c.observe();c.observe(motion_end=True)
  self.assertFalse(verify_trial(c.record(),4,0))
 def test_ten_second_timeout_is_not_success(self):
  c=StrictCoverage(2001)
  for _ in range(499):c.observe()
  c.observe(external_timeout=True)
  with self.assertRaises(AssertionError):verify_trial(c.record(),2001,0)
 def test_failure_on_final_step_wins(self):
  c=StrictCoverage(2);c.observe(terminated=True,motion_end=True)
  self.assertTrue(verify_trial(c.record(),2,0))
 def test_early_end_rejected(self):
  with self.assertRaises(AssertionError):StrictCoverage(100).observe(motion_end=True)
 def test_running_trial_rejected(self):
  with self.assertRaises(AssertionError):StrictCoverage(100).record()
 def test_second_episode_cannot_extend_first(self):
  c=StrictCoverage(2);c.observe(terminated=True)
  with self.assertRaises(AssertionError):c.observe()
 def test_partial_coverage_cannot_be_labeled_complete(self):
  trial={'repeat':0,'start_frame':0,'reference_frames':1001,'control_steps':500,'end_reason':'motion_end','complete':True,'failed':False}
  with self.assertRaises(AssertionError):verify_trial(trial,1001,0)

class EvidenceTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='textop_synthetic_filter_');self.root=pathlib.Path(self.tmp.name)
  self.e={'protocol':{'schema':'textop.dynamic_filter.v2','privileged_actor_and_critic':True,'domain_randomization':False,
    'training_iterations':40000,'num_envs':8192,'relaxed_removed_terminations':['anchor_pos','ee_body_pos'],
    'strict_standard_terminations_restored':True,'failure_average_definition':'pinned_ema_failed_over_failed_plus_success_alpha0.001',
    'strict_rollouts_per_motion':1,'strict_reset_start_frame':0,'control_hz':50,'strict_pelvis_height_threshold_m':.25,'strict_end_effector_height_threshold_m':.5}}
  self.write('policy_checkpoint',b'SYNTHETIC_SCHEMA_TEST_ONLY_NOT_A_TRAINED_POLICY',binary=True)
  sha=self.e['policy_checkpoint']['sha256'];fp='f'*64
  self.train={'status':'TEXTOP_FORMAL_FILTER_TRAINING_FINISHED','mode':'formal','formal_filter_training_started':True,
    'num_envs':8192,'completed_learning_iterations':40000,'optimizer_updates':800000,'reference_sources':1,
    'reference_fingerprint':fp,'latest_checkpoint':{'sha256':sha,'completed_learning_iterations':40000,'actual_optimizer_updates':800000}}
  self.exit={'child_exit_code':0,'timed_out':False,'numerical_status':'TEXTOP_FORMAL_FILTER_TRAINING_FINISHED'}
  self.relaxed={'status':'RELAXED_TRAINED_POLICY_STATISTICS_EXPORTED','policy_checkpoint_sha256':sha,'reference_fingerprint':fp,
    'motions':[{'motion_id':'a'*64,'raw_failed':0,'raw_success':10,'ema_failed':0.,'ema_success':1.,'relaxed_failure_probability':0.,'reference_frames':4}]}
  self.strict={'status':'STRICT_ENTIRE_MOTION_ROLLOUTS_FINISHED','diagnostic_only':False,'policy_checkpoint_sha256':sha,'reference_fingerprint':fp,
    'standard_terminations_restored':True,'pelvis_height_threshold_m':.25,'end_effector_height_threshold_m':.5,
    'motions':[{'motion_id':'a'*64,'trials':[{'repeat':0,'start_frame':0,'reference_frames':4,'control_steps':3,'end_reason':'motion_end','complete':True,'failed':False}]}]}
  self.refresh()
 def tearDown(self):self.tmp.cleanup()
 def write(self,key,obj,binary=False):
  path=self.root/key;path.write_bytes(obj if binary else json.dumps(obj).encode())
  self.e[key]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 def refresh(self):
  for key,obj in [('training_receipt',self.train),('training_exit_receipt',self.exit),('relaxed_rollout_log',self.relaxed),('strict_rollout_log',self.strict)]:self.write(key,obj)
 def test_synthetic_schema_consistency_only(self):self.assertEqual(apply(self.e)[0]['decision'],'retain')
 def test_short_witness_rejected_despite_40k_protocol_header(self):
  self.train.update(status='TEXTOP_FILTER_STATISTICS_RESUME_WITNESS_PASSED',mode='witness',completed_learning_iterations=2,optimizer_updates=40)
  self.refresh()
  with self.assertRaises(AssertionError):apply(self.e)
 def test_unattempted_reference_rejected(self):
  self.relaxed['motions'][0]['raw_success']=0;self.refresh()
  with self.assertRaises(AssertionError):apply(self.e)
 def test_changed_checkpoint_rejected(self):
  pathlib.Path(self.e['policy_checkpoint']['path']).write_bytes(b'CHANGED')
  with self.assertRaises(AssertionError):apply(self.e)
 def test_incomplete_strict_trajectory_rejected(self):
  self.strict['motions'][0]['trials'][0]['control_steps']=1;self.refresh()
  with self.assertRaises(AssertionError):apply(self.e)
 def test_wrong_policy_fingerprint_rejected(self):
  self.strict['reference_fingerprint']='e'*64;self.refresh()
  with self.assertRaises(AssertionError):apply(self.e)
 def test_missing_strict_reference_rejected(self):
  self.strict['motions']=[];self.refresh()
  with self.assertRaises(AssertionError):apply(self.e)
 def test_strict_failure_is_rejection(self):
  self.strict['motions'][0]['trials'][0].update(control_steps=1,end_reason='termination',failed=True,complete=False);self.refresh()
  self.assertEqual(apply(self.e)[0]['decision'],'reject_strict_failure')
 def test_relaxed_high_failure_rejected_without_strict_trial(self):
  self.relaxed['motions'][0].update(raw_failed=1,ema_failed=1.,ema_success=9.,relaxed_failure_probability=.1)
  self.strict['motions']=[];self.refresh()
  self.assertEqual(apply(self.e)[0]['decision'],'reject_relaxed_failure_rate')
 def test_failed_supervisor_rejected(self):
  self.exit['child_exit_code']=1;self.refresh()
  with self.assertRaises(AssertionError):apply(self.e)

if __name__=='__main__':unittest.main(verbosity=2)

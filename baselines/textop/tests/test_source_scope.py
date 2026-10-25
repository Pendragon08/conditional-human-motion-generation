"""Small synthetic source-accounting regressions; no motions or physics."""
import copy,hashlib,json,pathlib,sys,tempfile,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
from source_scope import verify_scope,digest

class ScopeTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='textop_scope_');self.root=pathlib.Path(self.tmp.name)
  self.original=[{'source':'CMU/a_poses.npz','split':'train'},{'source':'CMU/b_poses.npz','split':'val'},{'source':'CMU/missing_poses.npz','split':'train'}]
  self.missing=[self.original[-1]]
  self.rows=[{'canonical_source':'CMU/a','split':'train','dof':29,'features':69,'frames50':100},
             {'canonical_source':'CMU/b','split':'val','dof':29,'features':69,'frames50':200}]
  self.profile={'profile_id':'SYNTHETIC_SCOPE_ONLY','requested_index_sources':3,'unavailable_index_sources':1,
    'available_target_sources':2,'paper_data_scope_exact':False,'author_private_clips_included':0,'author_private_clips_unavailable':403}
  self.ids=[hashlib.sha256(x.encode()).hexdigest() for x in ['CMU/a','CMU/b']]
  self.refresh()
 def tearDown(self):self.tmp.cleanup()
 def entry(self,name,obj,jsonl=False):
  path=self.root/name;path.write_text('\n'.join(json.dumps(x) for x in obj) if jsonl else json.dumps(obj));return {'path':str(path),'sha256':digest(path)}
 def refresh(self):
  original=self.entry('original',self.original);missing=self.entry('missing',self.missing);manifest=self.entry('prepared',self.rows,True)
  self.profile.update(original_index_sha256=original['sha256'],unavailable_index_sha256=missing['sha256'])
  self.coverage={'schema':'textop.public_source_scope.v1','profile_id':self.profile['profile_id'],
    'original_index':original,'unavailable_index':missing,'prepared_manifests':[manifest]}
 def test_declared_gap_is_not_an_unprocessed_available_source(self):
  result=verify_scope(self.profile,self.coverage,self.ids);self.assertEqual(result['missing_available_public_sources'],0);self.assertEqual(result['declared_unavailable_index_sources'],1);self.assertFalse(result['paper_data_scope_exact'])
 def test_missing_available_source_rejected(self):
  self.rows.pop();self.refresh()
  with self.assertRaises(AssertionError):verify_scope(self.profile,self.coverage,self.ids)
 def test_missing_filter_decision_rejected(self):
  with self.assertRaises(AssertionError):verify_scope(self.profile,self.coverage,self.ids[:1])
 def test_duplicate_or_cross_split_source_rejected(self):
  self.rows.append({**self.rows[0],'split':'val'});self.refresh()
  with self.assertRaises(AssertionError):verify_scope(self.profile,self.coverage,self.ids)
 def test_foreign_source_rejected(self):
  self.rows[0]['canonical_source']='CMU/foreign';self.refresh()
  with self.assertRaises(AssertionError):verify_scope(self.profile,self.coverage,self.ids)
 def test_original_index_mutation_rejected(self):
  pathlib.Path(self.coverage['original_index']['path']).write_text('[]')
  with self.assertRaises(AssertionError):verify_scope(self.profile,self.coverage,self.ids)
 def test_silent_gap_count_change_rejected(self):
  self.profile['unavailable_index_sources']=0
  with self.assertRaises(AssertionError):verify_scope(self.profile,self.coverage,self.ids)
 def test_exact_paper_data_claim_rejected(self):
  self.profile['paper_data_scope_exact']=True
  with self.assertRaises(AssertionError):verify_scope(self.profile,self.coverage,self.ids)
 def test_invented_private_clips_rejected(self):
  self.profile['author_private_clips_included']=403
  with self.assertRaises(AssertionError):verify_scope(self.profile,self.coverage,self.ids)
 def test_wrong_declared_index_hash_rejected(self):
  self.profile['original_index_sha256']='0'*64
  with self.assertRaises(AssertionError):verify_scope(self.profile,self.coverage,self.ids)

if __name__=='__main__':unittest.main(verbosity=2)

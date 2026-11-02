"""Verify the declared public subset against real private source manifests.

A disclosed unavailable export is different from an unprocessed available
source. Neither changes the paper-data-equivalence claim to true. All lists of
individual sources stay private; this verifier publishes no per-source data.
"""
import hashlib,json,pathlib
from reconcile_raw_sources import source_key

def digest(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8388608),b''):h.update(block)
 return h.hexdigest()

def verified(entry):
 path=pathlib.Path(entry['path']);assert path.is_file() and digest(path)==entry['sha256']
 return path

def verify_scope(profile,coverage,decision_ids):
 assert coverage['schema']=='textop.public_source_scope.v1' and coverage['profile_id']==profile['profile_id']
 assert profile['paper_data_scope_exact'] is False and profile['author_private_clips_included']==0
 assert profile['author_private_clips_unavailable']>0
 original=coverage['original_index'];missing=coverage['unavailable_index']
 assert original['sha256']==profile['original_index_sha256'] and missing['sha256']==profile['unavailable_index_sha256']
 original_rows=json.loads(verified(original).read_text());missing_rows=json.loads(verified(missing).read_text())
 originals={source_key(x['source']):x['split'] for x in original_rows};unavailable={source_key(x['source']) for x in missing_rows}
 assert len(originals)==len(original_rows)==profile['requested_index_sources']
 assert len(unavailable)==len(missing_rows)==profile['unavailable_index_sources'] and unavailable<=originals.keys()
 expected=originals.keys()-unavailable;assert len(expected)==profile['available_target_sources']
 prepared={}
 for entry in coverage['prepared_manifests']:
  path=verified(entry)
  for line in path.read_text().splitlines():
   row=json.loads(line);key=row['canonical_source']
   assert key not in prepared,'Duplicate prepared source, including across splits'
   assert key in expected and row['split']==originals[key] and row['dof']==29 and row['features']==69
   assert type(row['frames50']) is int and row['frames50']>=2
   prepared[key]=row['split']
 assert prepared.keys()==expected,'Available sources remain unprocessed or missing'
 ids={hashlib.sha256(key.encode()).hexdigest() for key in expected}
 assert len(decision_ids)==len(set(decision_ids)) and set(decision_ids)==ids,'Available sources lack actual filter decisions'
 return {'profile_id':profile['profile_id'],'requested_index_sources':len(originals),'available_prepared_and_evaluated_sources':len(prepared),
         'declared_unavailable_index_sources':len(unavailable),'missing_available_public_sources':0,
         'author_private_clips_included':0,'paper_data_scope_exact':False}

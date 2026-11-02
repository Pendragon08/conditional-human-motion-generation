"""Isolated document metadata fixture; no production inputs/models/jobs."""
import copy,json,pathlib,sys
B=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B/"scripts"))
from ldm100k_verified_input import validate_document
r={"run":"user_ldm100k_prefix_document_v1","source_sha256":"1"*64,"spec_sha256":"2"*64,"export_receipt_sha256":"3"*64,"selected_checkpoint":{"sha256":"4"*64},"inference":{"sha256":"5"*64},"metadata_audit":{"fixture":True}}
b={"run":r["run"],"closure":{"actual_exit_code":0,"fixture":True}}
s={"status":"USER_LDM100K_PREFIX_FRESH_DOCUMENT_VERIFIED","run":r["run"],"actual_exit_code":0,"findings":[],"executor_sha256":r["source_sha256"],"spec_sha256":r["spec_sha256"],"export_receipt_sha256":r["export_receipt_sha256"],"selected_checkpoint_sha256":r["selected_checkpoint"]["sha256"],"inference_sha256":r["inference"]["sha256"],"actual_metadata_audit":r["metadata_audit"],"actual_closure":b["closure"]}
validate_document(s,r,b)
mutations=[("status","WRONG"),("run","other_run"),("actual_exit_code",1),("actual_exit_code",False),("findings",["fixture finding"]),("executor_sha256","0"*64),("spec_sha256","0"*64),("export_receipt_sha256","0"*64),("selected_checkpoint_sha256","0"*64),("inference_sha256","0"*64),("actual_metadata_audit",{}),("actual_closure",{})]
for key,value in mutations:
 wrong=copy.deepcopy(s);wrong[key]=value
 try:validate_document(wrong,r,b)
 except AssertionError:pass
 else:raise AssertionError("Accepted corrupted fixture "+key)
print(json.dumps({"status":"ISOLATED_DOCUMENT_METADATA_FIXTURE_PASSED","negative_cases":len(mutations),"production_inputs_read":0,"model_loads":0,"gpu_jobs":0,"actual_scope_builds":0,"actual_document_verification":False}))

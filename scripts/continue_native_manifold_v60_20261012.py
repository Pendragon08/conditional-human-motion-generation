"""Preserve V58 failed preflight; match the actual immutable failure record.

V58 never launched a random replay. This separate entry verifies its failed
document witness, then uses the same frozen missing-batch continuation logic.
"""
from pathlib import Path
from verify_execution_lineage_v39_20261012 import read,sha

parent=Path(__file__).with_name('continue_native_manifold_v58_20261012.py')
assert sha(parent)=='81f97153cde8d1be81452b00e629fc0dc06e897c33407e2901a56c3f7d9f82a0'
base=Path(__file__).resolve().parents[1]/'results/native_manifold_v56_20261012'
failed=read(base/'CONTINUATION_DOCUMENT_WITNESS_V58.json')
assert failed['exit_code']==1 and failed['status'].startswith('FAIL')
assert sha(base/'failed_main_invocation_state.json')=='656983246104825dbb7e7e845d73fd73479f7858c98f0a2e0df9fbbb1be5a8d9'
source=parent.read_text()
old="assert failure['status']=='FAILED_RETAINED' and 'DOCUMENT_WITNESS.json' in failure['error']"
assert source.count(old)==1
source=source.replace(old,"assert failure['status']=='FAILED_RETAINED' and failure['error']==\"FileNotFoundError(2, 'No such file or directory')\"")
source=source.replace('native_manifold_continuation_v58_20261012.md','native_manifold_continuation_v60_20261012.md')
source=source.replace('continuation_manifest_v58.json','continuation_manifest_v60.json')
source=source.replace('continuation_receipt_v58.json','continuation_receipt_v60.json')
source=source.replace('NATIVE_MANIFOLD_V58_CONTINUATION_COMPLETE','NATIVE_MANIFOLD_V60_CONTINUATION_COMPLETE')
source=source.replace("continuation='V58'","continuation='V60'")
source=source.replace("original_v56_source_changed=False,original_failed_invocation_not_success=True,",
    "original_v56_source_changed=False,original_failed_invocation_not_success=True,\n            parent_source_sha256=sha(parent),preserved_v58_failure_sha256=sha(out/'CONTINUATION_DOCUMENT_WITNESS_V58.json'),")
exec(compile(source,str(Path(__file__)),'exec'),globals())

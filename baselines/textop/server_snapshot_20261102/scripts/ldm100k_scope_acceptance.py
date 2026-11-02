"""Require the real closed private metadata scope before any numerical job."""
from ldm100k_verified_input import ROOT, read, sha, owned, closed_job, accepted_verified_input
from ldm100k_scope_contract import RUN, validate_policy, validate_scope_receipt
from ldm100k_generation_protocol import validate_diagnostic_scope


def accepted_scope(binding):
    assert binding['run'] == RUN
    sp = owned(ROOT / 'configs/ldm100k-heldout-scope-spec.json')
    assert sha(sp) == binding['spec_sha256']
    s = read(sp)
    validate_policy(s)
    for name, digest in s['bound_sources'].items():
        assert sha(ROOT / name) == digest, name
    assert sha(ROOT / 'scripts/build_ldm100k_heldout_scope.py') == s['executor_sha256']
    assert sha(ROOT / 'scripts/supervise_ldm100k_heldout_scope.py') == s['supervisor_sha256']
    rp = owned(ROOT / 'ldm100k_user_handoff_20261028' / RUN / 'receipt.json')
    assert sha(rp) == binding['receipt_sha256']
    rec = read(rp)
    validate_scope_receipt(rec, s, sha(sp))
    closure = closed_job('ldm100k_heldout_scope_' + RUN, rp, sha(rp), s['supervisor_sha256'])
    assert closure['exit_sha256'] == binding['exit_sha256'] and closure == binding['closure']
    artifacts = {}
    for name, item in rec['artifacts'].items():
        p = owned(item['path'])
        assert p.parent == rp.parent and p.name == name and p.stat().st_size == item['bytes']
        assert sha(p) == item['sha256']
        artifacts[name] = p
    scope = read(artifacts['diagnostic_scope.private.json'])
    plan = validate_diagnostic_scope(scope)
    verified = accepted_verified_input(s['verified_model'])
    index = read(artifacts['source_index.private.json'])
    assert len(index) == rec['actual_validation_original_families']
    assert len({x['canonical_source'] for x in index}) == len(index)
    assert all(x['split'] == 'val' and x['mirrored'] is False for x in index)
    return dict(receipt=rec, receipt_sha256=sha(rp), scope=scope, plan=plan,
                verified=verified, source_index=index, spec=s, closure=closure)

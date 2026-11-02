"""Draft for independent process verification of a mixed PPO 5000 export.

This module is not an executable or an authorization boundary.  A future
caller must first authenticate the training and export closures, sources and
request, own the common lock, and supervise resources and process exits.
No policy tracking or latency measurement is performed here.
"""
import os


def verify(checkpoint, exported_actor, training_binding, export_receipt):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not __debug__:
        raise RuntimeError('CPU-only, non-optimized interpreter required')
    if (export_receipt.get('status') != 'MIXED_PPO5000_CPU_EXPORTED_AWAITING_INDEPENDENT_VERIFY'
            or export_receipt.get('output_writers_closed') is not True
            or export_receipt.get('training_binding') != training_binding
            or export_receipt.get('independent_process_verification') is not False):
        raise ValueError('unaccepted export receipt')
    import torch
    from augmentation2526_stable_file_v1 import stable_owned_sha256
    from augmentation2526_mixed_ppo_runtime_v1 import content, equal
    from augmentation2526_mixed5000_cpu_core_v1 import audit_and_payload, independent_reload

    if torch.cuda.is_initialized() or torch.cuda.device_count() != 0:
        raise RuntimeError('CUDA must remain disabled')
    checkpoint_before = stable_owned_sha256(checkpoint, 1013)
    actor_before = stable_owned_sha256(exported_actor, 1013)
    recorded = training_binding['checkpoint']
    if (checkpoint_before['sha256'] != recorded['sha256']
            or checkpoint_before['bytes'] != recorded['bytes']
            or checkpoint_before != export_receipt['checkpoint_after']
            or actor_before != export_receipt['inference']):
        raise ValueError('export or checkpoint bytes do not match the receipt')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    artifact = torch.load(exported_actor, map_location='cpu', weights_only=False)
    if content(state) != recorded['content_sha256']:
        raise ValueError('checkpoint content digest mismatch')
    reconstructed, audit, expected, probes = audit_and_payload(state, training_binding)
    if not equal(reconstructed, artifact):
        raise ValueError('independent actor and normalization differ from export')
    original_audit = export_receipt['audit']
    for key in ('fullstate_content_sha256', 'payload_content_sha256', 'probe_output_hashes'):
        if audit[key] != original_audit[key]:
            raise ValueError('independent audit mismatch: ' + key)
    probe = independent_reload(artifact, expected, probes)
    if (stable_owned_sha256(checkpoint, 1013) != checkpoint_before
            or stable_owned_sha256(exported_actor, 1013) != actor_before
            or content(state) != recorded['content_sha256']):
        raise ValueError('input changed during verification')
    return dict(checkpoint=checkpoint_before, inference=actor_before, audit=audit,
                probe=probe, optimizer_updates_performed=0,
                synthetic_probe_only=True, heldout_tracking_accepted=False,
                online_latency_accepted=False)

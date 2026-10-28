"""Select only the genuine user100k LDM fullstate, never a VAE or300k state."""
PATH = '/home/ouyanghao/textop-user-vae100k-ldm-dual-20261028/checkpoints/ldm_state_100000.pth'


def validate_descriptor(checkpoint, export_binding):
    c, b = checkpoint, export_binding
    assert c['path'] == PATH and c['manager_step'] == 100000
    assert type(c['bytes']) is int and 0 < c['bytes'] <= 512 * 1024**2
    assert type(c['actual_optimizer_updates']) is int and 0 < c['actual_optimizer_updates'] <= 100000
    assert b['run'] == 'user_ldm100k_prefix_export_v1'
    for value in [c['sha256'], b['spec_sha256'], b['receipt_sha256'], b['exit_sha256']]:
        assert isinstance(value, str) and len(value) == 64 and all(x in '0123456789abcdef' for x in value)

"""Small REAL CPU filesystem fixtures only; preserve them and never load models."""
import argparse
import json
import os
import pathlib
from filter_recovery_destination import guard_destination

ROOT = pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p = argparse.ArgumentParser(); p.add_argument('--run', required=True); a = p.parse_args()
assert a.run.isascii() and a.run.isidentifier() and os.getuid() == 1013
os.sched_setaffinity(0, {0, 1, 2, 3})
out = ROOT / 'filter_recovery_checkpoint_tools_20261027' / a.run
assert not out.exists(); out.mkdir()
checks = []
def reject(name, target, **kwargs):
    try: guard_destination(out, target, **kwargs)
    except (AssertionError, ValueError): checks.append(name); return
    raise AssertionError('Unexpected acceptance: ' + name)
target = out / 'normal' / 'new_audit'
proof = guard_destination(out, target); assert not target.exists(); checks.append('new_owned_persistent_path_accepted_without_write')
target.mkdir(parents=True)
guard_destination(out, target, require_absent=False); checks.append('created_owned_path_verified')
reject('existing_output_rejected', target)
(out / 'regular_file').write_text('retained fixture', encoding='utf-8')
reject('regular_file_ancestor_rejected', out / 'regular_file' / 'child')
(out / 'link').symlink_to(out / 'normal', target_is_directory=True)
reject('symlink_ancestor_rejected', out / 'link' / 'child')
(out / 'dangling').symlink_to(out / 'not_created', target_is_directory=True)
reject('dangling_symlink_rejected', out / 'dangling' / 'child')
reject('escape_rejected', out / '..' / 'escape')
reject('outside_root_rejected', out.parent / 'outside')
reject('wrong_owner_rejected', out / 'fresh', owner_uid=0)
reject('root_itself_rejected', out)
summary = dict(status='RECOVERY_DESTINATION_CPU_FIXTURES_PASSED', checks=len(checks), names=checks,
    valid_guard=proof, fixtures_retained=True, real_checkpoint_loads=0, sdk_runs=0, training_signals=0)
(out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary))

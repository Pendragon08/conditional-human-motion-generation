"""Reuse strict V42 diagnostic launcher for a separately frozen V43 amendment."""
from pathlib import Path

parent=Path(__file__).with_name('run_reference_observation_v42_20261012.py')
source=parent.read_text()
assert source.count("OUT = ROOT / 'results/reference_observation_v42_20261012'")==1
source=source.replace("OUT = ROOT / 'results/reference_observation_v42_20261012'",
                      "OUT = ROOT / 'results/reference_velocity_v43_20261012'")
for old,new in (
    ('scripts/reference_observation_collect_v42_20261012.py','scripts/reference_velocity_collect_v43_20261012.py'),
    ('scripts/run_reference_observation_v42_20261012.py','scripts/run_reference_velocity_v43_20261012.py'),
    ('docs/reference_observation_v42_20261012.md','docs/reference_velocity_v43_20261012.md')):
    assert old in source
    source=source.replace(old,new)
old="                 'scripts/verify_execution_lineage_v39_20261012.py',\n"
new="                 'scripts/reference_observation_collect_v42_20261012.py',\n" \
    "                 'scripts/run_reference_observation_v42_20261012.py',\n"+old
assert source.count(old)==1
source=source.replace(old,new)
exec(compile(source,str(Path(__file__)),'exec'),globals())

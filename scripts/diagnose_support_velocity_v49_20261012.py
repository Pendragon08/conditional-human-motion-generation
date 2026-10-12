"""Read-only amendment: native per-case COM, preserve failed V48 source."""
from pathlib import Path
parent=Path(__file__).with_name('diagnose_support_velocity_v48_20261012.py')
source=parent.read_text()
old="OUT=ROOT/'results/support_velocity_diagnostic_v48_20261012'"
assert source.count(old)==1
source=source.replace(old,"OUT=ROOT/'results/support_velocity_diagnostic_v49_20261012'")
old="np.asarray(g['com_local'])"
assert source.count(old)==1
source=source.replace(old,"np.asarray(meta['native_body_properties'][n][k]['com'],dtype=float)")
source=source.replace("metric_source_sha256=sha(ROOT/'scripts/physics_guided_metric_v21_20261012.py'),",
    "metric_source_sha256=sha(ROOT/'scripts/physics_guided_metric_v21_20261012.py'),\n        parent_source_sha256=sha(parent),preserved_failed_parent='V48 missing COM injection, no GPU calls',")
exec(compile(source,str(Path(__file__)),'exec'),globals())

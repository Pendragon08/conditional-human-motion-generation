"""V20 passive native export with independently calibrated COM velocity metric."""
from pathlib import Path

previous=Path(__file__).with_name('contact_point_collect_v20_20261012.py')
adapter=previous.read_text()
ending="exec(compile(source, str(Path(__file__)), 'exec'), globals())"
assert adapter.rstrip().endswith(ending)
exec(compile(adapter.rsplit(ending,1)[0],str(previous),'exec'),globals())
old='from physics_guided_core import foot_geometry, heading, metrics\n'
assert source.count(old)==1
source=source.replace(old,'from physics_guided_metric_v23_20261012 import foot_geometry, heading, metrics\n')
exec(compile(source,str(Path(__file__)),'exec'),globals())

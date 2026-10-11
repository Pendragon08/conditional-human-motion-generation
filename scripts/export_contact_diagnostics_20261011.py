"""Export small evidence files; licensed motion arrays remain server-local."""
import tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
names=['contact_sampling_v6_20261011','contact_sampling_v7_20261011',
       'contact_sampling_v8_20261011','contact_sensor_v9_20261011','contact_sensor_v10_20261011',
       'contact_geometry_calibration_20261011','contact_force_fixture_20261011','contact_force_fixture_v2_20261011','contact_force_fixture_v3_20261012','contact_cpu_export_v11_20261012','contact_cpu_validation_v12_20261012']
archive=root/'results/contact_diagnostics_public_v9_20261011.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for name in names:
        directory=root/'results'/name
        for path in sorted(directory.rglob('*')):
            if not path.is_file() or path.suffix not in ('.json','.yaml','.csv','.md','.log'):
                continue
            if path.name in ('source_manifest.json','measurement_manifest.json'):
                continue
            relative=path.relative_to(directory)
            if relative.parts[0]=='raw': relative=Path('public')/relative
            tar.add(path,arcname=str(Path('results')/name/relative),recursive=False)
        if name.startswith('contact_force_fixture'):
            for path in sorted(directory.glob('repeat_*/fixture_arrays.npz')):
                tar.add(path,arcname=str(path.relative_to(root)),recursive=False)
print(archive,archive.stat().st_size)

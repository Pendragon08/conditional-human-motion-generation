"""Point an isolated CPU preflight at a bounded projection preview."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-config', type=Path, required=True)
    p.add_argument('--preview', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = a.preview.resolve(strict=True)
    config = json.loads(a.base_config.read_text())
    config.update({'dataset_root': str(root), 'manifest': str(root / 'manifest.json'),
                   'stats': str(root / 'stats.npz'), 'state_root': str(root),
                   'state_manifest': str(root / 'state_manifest.json'),
                   'run_dir': str(root), 'device': 'cpu',
                   'purpose': 'CPU preflight only; unaccepted bounded projection preview'})
    with a.output.open('x', encoding='utf8') as f:
        json.dump(config, f, indent=2); f.write('\n')


if __name__ == '__main__': main()

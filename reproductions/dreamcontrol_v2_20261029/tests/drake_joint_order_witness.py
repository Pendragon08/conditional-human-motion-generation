"""Inspect original OmniRetarget URDF position names using Drake itself.

Use without a SceneGraph to avoid loading visualization meshes. Drake version
and model hash must accompany output: a different version may order positions
differently. This is an inspection, not verification against publisher output.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from motion_bridge import SOURCE_NAMES


def inspect(path: Path) -> dict:
    plant = MultibodyPlant(time_step=0.0)
    models = Parser(plant).AddModels(str(path.resolve()))
    if len(models) != 1:
        raise ValueError(f'expected a single model, got {len(models)}')
    plant.Finalize()
    model = models[0]
    names = list(plant.GetPositionNames(model))
    expected_base = ['pelvis_qw', 'pelvis_qx', 'pelvis_qy', 'pelvis_qz',
                     'pelvis_x', 'pelvis_y', 'pelvis_z']
    expected_joints = [name + '_q' for name in SOURCE_NAMES]
    return {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'num_positions': plant.num_positions(model), 'position_names': names,
            'matches_expected_base_order': names[:7] == expected_base,
            'matches_bridge_joint_order': names[7:] == expected_joints}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('urdfs', nargs='+', type=Path)
    args = parser.parse_args()
    print(json.dumps({'drake_version': importlib.metadata.version('drake'),
                      'models': [inspect(path) for path in args.urdfs]}, indent=2))


if __name__ == '__main__':
    main()

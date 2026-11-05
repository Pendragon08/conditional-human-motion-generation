"""Scene and scoring adapter for the pinned public DreamControl Pick task."""
import ast
from pathlib import Path

REVISION = 'ec28b508313d6837de52c68247fbc8cf81fe4935'
BASE = Path(__file__).with_name('motion_tracking_interactive_base.py')
PICK = Path(__file__).with_name('motion_tracking_pick_env.py')
PLAY = Path(__file__).with_name('play_eval.py')


def configure(cfg, original_table_center, sim_utils, mdp, EventTerm, SceneEntityCfg,
              ManagerBasedRLEnv, torch):
    from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg

    namespace = locals().copy()
    base_tree = ast.parse(BASE.read_text())
    pick_tree = ast.parse(PICK.read_text())

    def execute(nodes, path):
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)

    def class_fields(tree, name, fields):
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
        nodes = [node for node in cls.body if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in fields]
        assert len(nodes) == len(fields)
        return nodes

    funcs = {'reset_object_state', 'object_above_threshold'}
    execute([node for node in base_tree.body if isinstance(node, ast.FunctionDef)
             and node.name in funcs], BASE)
    execute(class_fields(pick_tree, 'MySceneCfg', {'kitchen', 'object'}), PICK)
    execute(class_fields(pick_tree, 'EventCfg', {'add_object_mass', 'reset_object'}), PICK)
    execute(class_fields(base_tree, 'EventCfg', {'physics_material'}), BASE)
    original_table_size = cfg.scene.table.spawn.size
    cfg.scene.table = namespace['kitchen']
    cfg.scene.object = namespace['object']
    task = next(node for node in pick_tree.body if isinstance(node, ast.ClassDef)
                and node.name == 'G1PickEnvCfg')
    init = next(node for node in task.body if isinstance(node, ast.FunctionDef)
                and node.name == '__post_init__')
    table = next(node for node in init.body if isinstance(node, ast.Assign)
                 and ast.unparse(node.targets[0]) == 'self.scene.kitchen.init_state.pos')
    cfg.scene.table.init_state.pos = ast.literal_eval(table.value)
    cfg.events.upstream_object_mass = namespace['add_object_mass']
    cfg.events.upstream_robot_material = namespace['physics_material']
    shift = [a - b for a, b in zip(cfg.scene.table.init_state.pos, original_table_center)]
    # Align the robot-facing edges because the original tabletop is wider in x.
    shift[0] -= (cfg.scene.table.spawn.size[0] - original_table_size[0]) / 2
    assert abs(shift[0] - 1.90) < 1e-12 and shift[1:] == [0, 0]
    return namespace['reset_object'], namespace['object_above_threshold'], shift

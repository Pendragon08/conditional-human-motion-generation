"""Make an exact official DreamControl train.py copy with one livestream CLI fix.

AppLauncher appends Isaac Kit flags to sys.argv after argparse has run. Hydra
later tries to parse those Kit flags and exits. Restore only the already-parsed
Hydra arguments after the Kit app has consumed its flags. Do not change the
training algorithm, environment, or reward code.
"""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / "third_party/DreamControl/Training/scripts/reinforcement_learning/rsl_rl/train.py"
target = root / "scripts/r3_dreamcontrol_train_livestream.py"
text = source.read_text(encoding="utf-8")
anchor = "simulation_app = app_launcher.app\n"
assert text.count(anchor) == 1, "Official training entry changed; inspect before adapting"
patch = (
    anchor
    + "\n# R3 livestream compatibility: hide Kit's appended flags from Hydra.\n"
    + "sys.argv = [sys.argv[0]] + hydra_args\n"
)
target.write_text(text.replace(anchor, patch, 1), encoding="utf-8")
print(f"R3_DREAMCONTROL_ENTRY_BUILT {target}")

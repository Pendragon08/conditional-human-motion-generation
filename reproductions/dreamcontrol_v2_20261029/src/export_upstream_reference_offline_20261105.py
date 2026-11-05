"""Serialize original DreamControl Viser playback using Viser's own recorder."""
import difflib, hashlib, json, sys
from pathlib import Path
import viser
from viser import _messages

out=Path(sys.argv[1]);info=json.loads((out/'prepared.json').read_text())
original=(out/'upstream_visualize_trajectories.py').read_text();source=original
recorder=None
def begin(server):
    global recorder
    recorder=server.get_scene_serializer()
    for message in [
        _messages.SetCameraPositionMessage(position=(2.15,-2.3,1.55)),
        _messages.SetCameraLookAtMessage(look_at=(.05,0,.75)),
        _messages.SetCameraUpDirectionMessage(position=(0.,0.,1.)),
        _messages.SetCameraFovMessage(fov=.75),
    ]:recorder._insert_message(message)

def capture(server,tstep):
    recorder.insert_sleep(1/20)

def finish(server):
    target=out/'all_four.viser';target.write_bytes(recorder.serialize())
    info.update(status='original_viewer_offline_recorded',frames=660,
        offline_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),offline_bytes=target.stat().st_size,
        rendering='Original DreamControl ViserUrdf playback; Viser native offline serialization')
    (out/'receipt.json').write_text(json.dumps(info,indent=2));server.stop()

def change(a,b):
    global source
    assert source.count(a)==1,a
    source=source.replace(a,b)
change("yourdfpy.URDF.load('../../Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')",f'yourdfpy.URDF.load({str(out/"g1_29body_14hand_viewer.urdf")!r})')
change('server = viser.ViserServer()',"server = viser.ViserServer(host='127.0.0.1', port=18088)")
change('while True:','begin(server)\n    for export_frame in range(num_timesteps):')
change('timestep_slider.value = (timestep_slider.value + 1) % num_timesteps','timestep_slider.value = export_frame')
change('time.sleep(1./20.)','capture(server, tstep)\n    finish(server)')
change('onp.zeros((1000, 1000)','onp.zeros((100, 100)')
change('jnp.array([0.01, 0.01, 1.0])','jnp.array([0.1, 0.1, 1.0])')
(out/'viewer_adapter.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile='upstream',tofile='offline_export_adapter')))
(out/'viewer_executed.py').write_text(source)
sys.argv=['upstream_visualize_trajectories.py','--folder',str(out),'--id','all_four']
exec(compile(source,str(out/'viewer_executed.py'),'exec'),{'__name__':'__main__','begin':begin,'capture':capture,'finish':finish})

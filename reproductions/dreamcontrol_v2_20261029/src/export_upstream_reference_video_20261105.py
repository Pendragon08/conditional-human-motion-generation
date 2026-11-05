"""Finite video export of upstream DreamControl's Viser reference viewer.

The original viewer source is retained and hashed. Its ViserUrdf playback code
is executed with a file/asset adapter and a frame-capture hook.
"""
import difflib
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

OUT = Path(sys.argv[1])
info = json.loads((OUT/'prepared.json').read_text())
sha = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source = (OUT/'upstream_visualize_trajectories.py').read_text()
assert sha(OUT/'upstream_visualize_trajectories.py') == info['original_viewer_sha256']
current, writer, client = None, None, None
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
def camera(eye,target):
    eye,target = np.array(eye),np.array(target)
    z = target-eye;z /= np.linalg.norm(z)
    x = np.cross(z,[0,0,1]);x /= np.linalg.norm(x)
    y = np.cross(z,x)
    xyzw = Rotation.from_matrix(np.stack([x,y,z],axis=1)).as_quat()
    return {'position':eye,'wxyz':xyzw[[3,0,1,2]],'fov':np.deg2rad(48)}
CAMERAS = [camera([2.4,-2.4,1.7],[.1,0,.78]),camera([1.28,-.85,1.25],[.22,0,1.0])]
def checkpoint(**values):
    (OUT/'export_status.json').write_text(json.dumps(values,indent=2))
def close_video():
    global writer
    if writer:
        writer.stdin.close();assert writer.wait(timeout=30) == 0
        video=OUT/(current['slug']+'.mp4')
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
                 '-show_entries','stream=nb_read_frames,width,height,avg_frame_rate,codec_name','-of','json',str(video)]))
        assert int(probe['streams'][0]['nb_read_frames']) == current['frames']
        current['video_sha256']=sha(video);current['video_bytes']=video.stat().st_size;current['probe']=probe
        writer=None
def capture(server,tstep):
    global current,writer,client
    if client is None:
        checkpoint(status='waiting_browser',port=18085)
        deadline=time.monotonic()+600
        while not server.get_clients():
            assert time.monotonic()<deadline,'No local viewer connected'
            time.sleep(.2)
        client=next(iter(server.get_clients().values()))
        client.camera.position=CAMERAS[0]['position'];client.camera.wxyz=CAMERAS[0]['wxyz'];client.camera.fov=CAMERAS[0]['fov']
        server.flush();time.sleep(2)
    row=next(c for c in info['clips'] if c['start']<=tstep<c['start']+c['frames'])
    if current is not row:
        close_video();current=row
        video=OUT/(row['slug']+'.mp4');assert not video.exists()
        writer=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
            '-s','1280x620','-r','20','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18',
            '-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
    local_frame=tstep-row['start'];server.flush()
    views=[client.get_render(540,640,**cam,transport_format='jpeg') for cam in CAMERAS]
    canvas=Image.new('RGB',(1280,620),(22,26,32));canvas.paste(Image.fromarray(views[0]),(0,42));canvas.paste(Image.fromarray(views[1]),(640,42))
    draw=ImageDraw.Draw(canvas);hand=row['hand_closed'][local_frame]
    title=row['task'].replace('_',' ').upper()
    draw.text((14,10),f'{title} | reference {row["candidate_id"]} | {local_frame/20:.2f} s',font=font,fill='white')
    draw.text((650,10),f'Hand plan  L: {"CLOSED" if hand[0] else "OPEN"}   R: {"CLOSED" if hand[1] else "OPEN"}',font=font,fill='white')
    draw.text((14,592),'DreamControl upstream Viser viewer | 29 body + 14 finger joints | Kinematic reference, no object/contact simulation',font=small,fill=(235,235,235))
    writer.stdin.write(np.asarray(canvas).tobytes())
    if local_frame in {0,39,40,70,100,104,120,140,row['frames']-1}:
        canvas.save(OUT/f'{row["slug"]}_frame{local_frame:03}.jpg',quality=91)
    if local_frame%20 == 0:checkpoint(status='rendering',task=row['task'],frame=local_frame,total=row['frames'])
def change(a,b):
    global source
    assert source.count(a)==1,a
    source=source.replace(a,b)
change("yourdfpy.URDF.load('../../Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')",f'yourdfpy.URDF.load({str(OUT/"g1_29body_14hand_viewer.urdf")!r})')
change('server = viser.ViserServer()',"server = viser.ViserServer(host='127.0.0.1', port=18085, label='DreamControl original reference viewer')")
change('while True:', 'for export_frame in range(num_timesteps):')
change('timestep_slider.value = (timestep_slider.value + 1) % num_timesteps','timestep_slider.value = export_frame')
change('time.sleep(1./20.)','capture(server, tstep)\n    close_video()\n    server.stop()')
# Same terrain size; fewer floor-only vertices reduces browser transfer cost.
change('onp.zeros((1000, 1000)', 'onp.zeros((100, 100)')
change('jnp.array([0.01, 0.01, 1.0])','jnp.array([0.1, 0.1, 1.0])')
original=(OUT/'upstream_visualize_trajectories.py').read_text()
(OUT/'viewer_adapter.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile='upstream',tofile='video_export_adapter')))
(OUT/'viewer_executed.py').write_text(source)
sys.argv=['upstream_visualize_trajectories.py','--folder',str(OUT),'--id','all_four']
checkpoint(status='starting')
try:
    exec(compile(source,str(OUT/'viewer_executed.py'),'exec'),{'__name__':'__main__','capture':capture,'close_video':close_video})
    info['status']='four_upstream_videos_exported';info['executed_source_sha256']=sha(OUT/'viewer_executed.py')
    (OUT/'receipt.json').write_text(json.dumps(info,indent=2));checkpoint(status='completed',clips=4,frames=660)
except BaseException:
    if writer:
        writer.stdin.close();writer.wait(timeout=30)
    import traceback
    checkpoint(status='failed_preserved',traceback=traceback.format_exc());raise

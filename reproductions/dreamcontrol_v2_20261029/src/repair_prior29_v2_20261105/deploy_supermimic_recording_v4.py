from pathlib import Path
import hashlib,json,os,shutil,subprocess
s=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/repair_prior29_v2_20261105')
def replace(text,a,b):
    assert text.count(a)==1,a
    return text.replace(a,b)
old=s/'src/supermimic_legacy_v1'
new=s/'src/supermimic_video_legacy_v4'
new.mkdir()
for f in old.glob('*.py'): shutil.copyfile(f,new/f.name)
p=new/'evaluate_native2000_tasks_v2_20261105.py'
source=p.read_text()
source=replace(source,'            def capture(env):','''            video_frames = []

            def capture_frame(env):
                robot = env.scene['robot']
                obj = env.scene['object']
                origin = env.scene.env_origins[0]
                root = robot.data.root_link_state_w[0]
                object_root = obj.data.root_state_w[0]
                video_frames.append(torch.cat((root[:3] - origin, root[3:7], robot.data.joint_pos[0],
                                                object_root[:3] - origin, object_root[3:7])).cpu().numpy())

            def capture(env):''')
source=replace(source,'                steps = env.episode_length_buf','''                if bool(first[0]):
                    capture_frame(env)
                steps = env.episode_length_buf''')
source=replace(source,"            pos = sim_env.scene['object'].data.root_pos_w - sim_env.scene.env_origins","""            video_joints = list(robot.joint_names)
            capture_frame(sim_env)
            pos = sim_env.scene['object'].data.root_pos_w - sim_env.scene.env_origins""")
source=replace(source,"    write(args.output / 'report.json', report)","""    frames = np.stack(video_frames)
    assert frames.shape == (report['episodes'][0]['steps'] + 1, 57)
    assert len(video_joints) == len(set(video_joints)) == 43 and np.isfinite(frames).all()
    assert np.allclose(frames[0, 50:53], positions[0], atol=2e-5, rtol=0)
    trace = args.output / 'episode_000_physics.npz'
    with trace.open('xb') as file:
        np.savez_compressed(file, root_pose_wxyz=frames[:, :7], joint_position=frames[:, 7:50],
                            object_pose_wxyz=frames[:, 50:], joint_names=video_joints, fps=50)
    report['video_trace'] = {
        'path': str(trace), 'sha256': sha(trace), 'candidate_id': rows[0]['candidate_id'],
        'frames': len(frames), 'fps': 50, 'selection': 'fixed first episode of this exact evaluation',
        'table_size': list(cfg.scene.table.spawn.size), 'table_center': list(cfg.scene.table.init_state.pos),
        'object_size': list(cfg.scene.object.spawn.size), 'robot_asset_sha256': report['asset_sha256'],
        'scope': 'Observed robot and dynamic object poses before auto-reset, including initial state',
    }
    write(args.output / 'report.json', report)""")
compile(source,str(p),'exec')
p.write_text(source)
c=s/'src/supermimic_repaired_controller_v4.py'
body=(s/'src/supermimic_repaired_controller_v3.py').read_text().replace('new_references_v3','new_references_v4').replace('evaluation_controller_v3','evaluation_controller_v4').replace('supermimic_repaired_v3','supermimic_repaired_v4').replace('supermimic_legacy_v1','supermimic_video_legacy_v4')
body=replace(body,"            update('object_evaluation_progress', object_evaluation=results)","""            trace = result['video_trace']
            checked(out / 'episode_000_physics.npz', trace['sha256'])
            require(trace['candidate_id'] == result['episodes'][0]['candidate_id'], 'Video provenance mismatch')
            results[task]['video_trace'] = trace
            update('object_evaluation_progress', object_evaluation=results)""")
compile(body,str(c),'exec')
with c.open('x') as f:f.write(body)
with (s/'supermimic_repaired_v4.log').open('xb') as log:
    p=subprocess.Popen(['/usr/bin/python3','-B','-u',str(c)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=s,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
print(json.dumps({'pid':p.pid,'sha256':hashlib.sha256(c.read_bytes()).hexdigest()}))

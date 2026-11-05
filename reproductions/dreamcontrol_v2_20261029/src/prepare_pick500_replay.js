const fs=require('fs'),path=require('path');
let s=fs.readFileSync(path.join(__dirname,'stage3_replay_pick29.py'),'utf8');
function rep(a,b){if(!s.includes(a))throw Error('base differs: '+a.slice(0,70));s=s.replace(a,b);}
rep("            sources.append(source);reports.append(report)",
`            assert 'task_object_position' in source and 'task_object_quaternion_wxyz' in source
            assert np.allclose(source['task_object_position'],report['task_object_position'],atol=1e-12)
            assert np.allclose(source['task_object_quaternion_wxyz'],report['task_object_quaternion_wxyz'],atol=1e-12)
            sources.append(source);reports.append(report)`);
rep("        frames=len(sources[0]['q'])",`        frames=len(sources[0]['q'])
        obj=env.scene['object']
        object_state=obj.data.default_root_state.clone()
        object_state[:,:3]=torch.tensor(np.stack([x['task_object_position'] for x in sources]),device=env.device)+env.scene.env_origins
        object_state[:,3:7]=torch.tensor(np.stack([x['task_object_quaternion_wxyz'] for x in sources]),device=env.device)
        object_state[:,7:]=0.
        obj.write_root_state_to_sim(object_state)
        env.scene.write_data_to_sim();env.sim.forward();env.scene.update(env.physics_dt)
        object_reset_error=float((obj.data.root_pos_w-object_state[:,:3]).abs().max())
        assert object_reset_error<1e-5,'per-reference object placement mismatch'`);
rep("            result={'eligible_for_calibrated_scene_reference':accepted,",`            result={'task_object_position':sources[k]['task_object_position'].tolist(),
                    'task_object_quaternion_wxyz':sources[k]['task_object_quaternion_wxyz'].tolist(),
                    'per_reference_object_reset_error_m':object_reset_error,
                    'eligible_for_calibrated_scene_reference':accepted,`);
rep("'kinematic_report':str(directory/'report.json'),'grasp_time_s':2.})",`'kinematic_report':str(directory/'report.json'),'grasp_time_s':2.,
                                    'task_object_position':sources[k]['task_object_position'].tolist(),
                                    'task_object_quaternion_wxyz':sources[k]['task_object_quaternion_wxyz'].tolist()})`);
rep("manifest={'asset_sha256':sha(ASSET),'scene_sha256':SCENE_SHA,'references':bank,",`manifest={'asset_sha256':sha(ASSET),'scene_sha256':SCENE_SHA,'references':bank,
                      'object_pose_policy':'per_reference_fixed','task':'pick','minimum_formal_reference_count':500,`);
fs.writeFileSync(path.join(__dirname,'stage3_pick500_replay.py'),s);
console.log('isolated per-reference-object native replay prepared');

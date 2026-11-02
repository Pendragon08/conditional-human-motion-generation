import pathlib,subprocess,os,json
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');d=r/'upstream/TextOpRobotMDAR/dataset/BABEL-AMASS-ROBOT-23dof-FULL-50fps'
cmd=[str(r/'env/bin/python'),str(r/'scripts/train_official.py'),'--config-name=train_mvae','expname=TextOp_Public_Full_MVAE_20261024','seed=0','data.weighted_sample=true','data.datadir='+str(d),'data.action_statistics_path='+str(d/'action_statistics.json'),'skeleton.asset.assetRoot='+str(r/'upstream/TextOpRobotMDAR/description/robots/g1'),'train.manager.stages=[100000,50000,50000]','data.num_primitive=4','train.manager.use_rollout=True','base_dir='+str(r/'runs'),'--cfg','job','--resolve']
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',WANDB_MODE='disabled')
p=subprocess.run(cmd,env=env,cwd=r/'upstream/TextOpRobotMDAR',capture_output=True,text=True)
(r/'receipts/config_probe.json').write_text(json.dumps({'returncode':p.returncode,'command':cmd,'stderr':p.stderr},indent=2))
(r/'receipts/mvae_resolved_config.yaml').write_text(p.stdout)
freeze=subprocess.run([str(r/'env/bin/python'),'-m','pip','freeze'],capture_output=True,text=True,env=env)
(r/'receipts/pip-freeze.txt').write_text(freeze.stdout)
print('CONFIG_EXIT',p.returncode,'STDERR',p.stderr[-2000:]);print(p.stdout[:850])

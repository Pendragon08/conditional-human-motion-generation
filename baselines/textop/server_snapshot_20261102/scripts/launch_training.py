"""Single-run launch gated by pinned assets + independent environment witness."""
import json,os,pathlib,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
if __name__=='__main__':
    lock=ROOT/'receipts/launcher.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    try:
        deadline=time.time()+3*3600
        required=['download_complete.json','kernel_witness.json','doc_witness_pass.json']
        while not all((ROOT/'receipts'/x).exists() for x in required):
            if time.time()>deadline:raise TimeoutError('Data/environment gate timed out')
            time.sleep(15)
        receipt=json.loads((ROOT/'receipts/download_complete.json').read_text())
        assert all(x['verified'] for x in receipt['files'])
        d=ROOT/'upstream/TextOpRobotMDAR/dataset/BABEL-AMASS-ROBOT-23dof-FULL-50fps'
        command=[str(ROOT/'env/bin/python'),str(ROOT/'scripts/train_official.py'),'--config-name=train_mvae','expname=TextOp_Public_Full_MVAE_20261024','seed=0','data.weighted_sample=true','data.datadir='+str(d),'data.action_statistics_path='+str(d/'action_statistics.json'),'skeleton.asset.assetRoot='+str(ROOT/'upstream/TextOpRobotMDAR/description/robots/g1'),'train.manager.stages=[100000,50000,50000]','data.num_primitive=4','train.manager.use_rollout=True','base_dir='+str(ROOT/'runs')]
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',WANDB_MODE='disabled',PYTHONUNBUFFERED='1',HYDRA_FULL_ERROR='1')
        def affinity():os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
        log=open(ROOT/'logs/mvae.log','ab')
        proc=subprocess.Popen(command,cwd=ROOT/'upstream/TextOpRobotMDAR',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=affinity)
        (ROOT/'receipts/training_launch.json').write_text(json.dumps({'pid':proc.pid,'command':command,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'complete_public_data':True,'total_manager_steps':200000,'gpu':1,'gpu_fraction':.65,'cpu_threads':4},indent=2))
        print('TEXTOP_TRAINING_PID',proc.pid,flush=True)
        code=proc.wait()
        (ROOT/'receipts/training_exit.json').write_text(json.dumps({'returncode':code,'time':time.time()}))
    finally:
        # Retain lock as an execution record; no automatic duplicate starts.
        pass

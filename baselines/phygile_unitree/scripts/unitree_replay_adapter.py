# Unitree-specific worker transport; preserves independent replay transactions.
"""Transactional rollback by fresh-process replay of accepted history.

Private PhysX solver state cannot be serialized through Isaac Lab's exposed
state API. Instead each attempt recreates the same seeded scene and replays
every accepted reference/control interval. Replayed actual states must agree
with the committed history before executing the new continuation. Rejections
only discard the tentative ledger; no live simulator is reused after failure.
"""
from pathlib import Path
import copy,hashlib,json,subprocess
import numpy as np
from physics_prefix import Motion,Execution


def save_motion(path,motion):
    np.savez_compressed(path,states=motion.states,body_positions=motion.body_positions,
        fps=np.array([motion.fps]),schema=np.asarray(motion.state_schema),body_names=np.asarray(motion.body_names))


def load_motion(path):
    with np.load(path,allow_pickle=False) as d:
        return Motion(d['states'],d['body_positions'],float(d['fps'][0]),str(d['schema']),tuple(d['body_names'].tolist()))


class UnitreeReplayPrefixAdapter:
    def __init__(self,root,checkpoint,motion_file,device,seed,output):
        self.root=Path(root).resolve();self.output=Path(output).resolve()
        self.output.mkdir(parents=True,exist_ok=False)
        self.config={'checkpoint':str(Path(checkpoint).resolve()),'motion_file':str(Path(motion_file).resolve()),
            'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            'device':device,'seed':seed}
        self.ledger=[];self.calls=[];self.invocations=[]
        directory=self._worker(None,0,'initialize')
        self._current=load_motion(directory/'current.npz')

    def _worker(self,candidate,prefix_frames,mode='rollout'):
        directory=self.output/f'invocation_{len(self.invocations):03d}'
        directory.mkdir()
        if candidate is not None:save_motion(directory/'candidate.npz',candidate)
        request={**self.config,'mode':mode,'history':copy.deepcopy(self.ledger),
            'candidate':str(directory/'candidate.npz') if candidate is not None else None,
            'prefix_frames':prefix_frames,'output':str(directory)}
        request_path=directory/'request.json';request_path.write_text(json.dumps(request,indent=2)+'\n')
        command=['bash',str(self.root/'scripts/run_isaac.sh'),str(self.root/'scripts/unitree_replay_worker.py'),
            '--request',str(request_path),'--headless','--device',self.config['device'],'--kit_args=--/renderer/multiGpu/enabled=false --/renderer/activeGpu=1 --/plugins/carb.tasking.plugin/threadCount=4 --/plugins/omni.tbb.globalcontrol/maxThreadCount=4']
        with (directory/'worker.log').open('w') as log:
            result=subprocess.run(command,cwd=self.root,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        record={'directory':str(directory),'command':command,'exit_code':result.returncode}
        self.invocations.append(record)
        if result.returncode or not (directory/'worker_result.json').exists():
            raise RuntimeError('Missing completed replay worker: '+str(directory))
        proof=json.loads((directory/'worker_result.json').read_text())
        if proof['status']!='COMPLETED' or proof['checkpoint_sha256']!=self.config['checkpoint_sha256']:
            raise ValueError('Worker identity/completion mismatch')
        if not proof['gmt_parameters_unchanged']:raise ValueError('Worker changed the frozen GMT')
        record['result']=proof
        return directory

    def current(self):
        return self._current

    def snapshot(self):
        return {'ledger':copy.deepcopy(self.ledger),'current':self._current}

    def restore(self,snapshot):
        self.ledger=copy.deepcopy(snapshot['ledger']);self._current=snapshot['current']

    def rollout(self,candidate,prefix_frames):
        np.testing.assert_allclose(self._current.states[-1],candidate.states[prefix_frames-1],atol=2e-6,rtol=0)
        directory=self._worker(candidate,prefix_frames)
        actual=load_motion(directory/'continuation.npz')
        with np.load(directory/'raw.npz',allow_pickle=False) as d:raw={k:d[k] for k in d.files}
        np.testing.assert_allclose(raw['actual_start_state'],candidate.states[prefix_frames-1],atol=2e-6,rtol=0)
        self.ledger.append({'candidate':str(directory/'candidate.npz'),'prefix_frames':prefix_frames,
            'actual':str(directory/'continuation.npz'),
            'candidate_sha256':candidate.digest(),'actual_sha256':actual.digest()})
        self._current=actual.slice(-1)
        self.calls.append({'candidate_sha256':candidate.digest(),**raw,'states':actual.states,
            'body_positions':actual.body_positions,'worker_directory':str(directory)})
        return Execution(actual,raw['terminated'],candidate.states[prefix_frames-1].copy(),candidate.digest())

    def close(self):
        pass  # Every worker has already exited; there is no live simulator.

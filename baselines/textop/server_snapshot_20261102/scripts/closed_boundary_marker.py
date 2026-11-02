"""Own-run temporary-path stop request: no signals, no live-code modifications.

The frozen trainer writes receipt.partial at its closed batch boundary. Reserving
that normally absent temporary name as our own EMPTY directory makes that write
raise IsADirectoryError, entering the existing emergency checkpoint handler.
This does not interrupt any checkpoint write. The original final receipt cannot
be refreshed through this marker; retain it and audit the real saved file/exits.
"""
import json, os, pathlib, time
from quick_pidfd import BoundProcess
from closed_boundary_interrupt import identity, require_identity


def request_boundary_save(expected, run_dir, evidence_dir, wait_seconds=600):
    run_dir=pathlib.Path(run_dir);out=pathlib.Path(evidence_dir)
    assert os.getuid()==expected['uid']==1013
    assert run_dir.resolve()==run_dir and run_dir.is_dir() and run_dir.stat().st_uid==1013
    assert out.resolve()==out and out.is_dir() and out.stat().st_uid==1013
    bound=BoundProcess(expected['pid'],expected['start_ticks'])
    marker=run_dir/'receipt.partial';created=False;marker_stat=None
    proof={'status':'PREPARING_OWN_TEMP_MARKER','training_signals':0,'source_edits':0,'worker':expected}
    def save():
        p=out/'marker_control.partial';p.write_text(json.dumps(proof,indent=2),encoding='utf-8');p.replace(out/'marker_control.json')
    try:
        require_identity(identity(expected['pid']),expected)
        assert bound.alive() and identity(expected['pid'])['state'] not in ('T','t')
        old=(run_dir/'receipt.json').read_bytes();(out/'original_receipt_before_pause.json').write_bytes(old)
        deadline=time.monotonic()+10
        while True:
            try:
                marker.mkdir(mode=0o700);created=True;marker_stat=marker.stat();break
            except FileExistsError:
                assert not marker.is_dir(),'Existing marker: inspect, never duplicate pause'
                assert time.monotonic()<deadline,'Original temporary writer remains active'
                time.sleep(.01)
        proof.update(status='WAITING_EXISTING_EMERGENCY_HANDLER',marker_inode=marker_stat.st_ino,
                     marker_device=marker_stat.st_dev,marker_path=str(marker),created_at=time.time())
        save()
        deadline=time.monotonic()+wait_seconds
        while bound.alive() and time.monotonic()<deadline:time.sleep(.2)
        proof['actual_worker_exited']=not bound.alive()
        assert not bound.alive(),'No forced stop: marker retained for pending boundary, inspect original worker'
        current=marker.stat()
        assert current.st_ino==marker_stat.st_ino and current.st_dev==marker_stat.st_dev and current.st_uid==1013
        assert marker.is_dir() and not marker.is_symlink() and not any(marker.iterdir())
        marker.rmdir();created=False
        proof.update(status='WORKER_EXITED_TEMP_MARKER_REMOVED_CHECKPOINT_AUDIT_REQUIRED',marker_removed=True)
        save();return proof
    except BaseException as e:
        proof.update(status='MARKER_HANDOFF_NEEDS_INSPECTION',error=repr(e),marker_retained=created)
        save();raise
    finally:bound.close()

"""Bound shutdown only after diagnostics and durable receipts are complete.

No cameras/writers are used by this diagnostic. A native SDK close hang is
retained explicitly; the fallback is not reported as native teardown success.
"""
import faulthandler,os,pathlib,sys,threading

def close(app,receipt,save,output,simulation=None,grace_seconds=20,
          successful_statuses=('TEXTOP_TRACKER_PHYSICS_KERNEL_PASSED',)):
    finished=threading.Event();lock=threading.Lock()
    stack=(pathlib.Path(output)/'native_teardown_stack.txt').open('w')
    faulthandler.dump_traceback_later(10,repeat=False,file=stack)
    complete=receipt.get('status') in successful_statuses and receipt.get('output_writers_closed') is True
    code=0 if complete else 1
    receipt.update(native_close_returned=False,teardown_fallback_used=False,shutdown_grace_seconds=grace_seconds)
    save()
    def watchdog():
        if finished.wait(grace_seconds):return
        with lock:
            receipt.update(teardown_fallback_used=True,native_close_returned=False,
                shutdown_reason='SDK close did not return after the completed headless job and closed output writers; process exit releases its private resources',
                shutdown_completion_verified=complete)
            save();stack.flush();os.fsync(stack.fileno());sys.stdout.flush();sys.stderr.flush()
        os._exit(code)
    threading.Thread(target=watchdog,daemon=True,name='bounded_headless_shutdown').start()
    if simulation is not None:
        simulation.stop();simulation.clear_all_callbacks()
    app.close(wait_for_replicator=False)
    finished.set();faulthandler.cancel_dump_traceback_later()
    with lock:
        receipt.update(native_close_returned=True,teardown_fallback_used=False);save();stack.close()

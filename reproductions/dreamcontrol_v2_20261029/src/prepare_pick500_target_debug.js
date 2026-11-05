const fs=require('fs'),path=require('path');
const L=__dirname;
let s=fs.readFileSync(path.join(L,'stage3_pick500_target_video.py'),'utf8');
const old='try:main()\nfinally:app.close(wait_for_replicator=False)';
if(!s.includes(old))throw Error('source boundary');
s=s.replace(old,`try:main()
except BaseException as e:
    import traceback
    failure={'status':'failed_preserved','error':repr(e),'traceback':traceback.format_exc()}
    (a.output_dir/'failure.json').write_text(json.dumps(failure,indent=2))
    print(failure['traceback'],flush=True)
    raise
finally:app.close(wait_for_replicator=False)`);
fs.writeFileSync(path.join(L,'stage3_pick500_target_video_v4.py'),s);

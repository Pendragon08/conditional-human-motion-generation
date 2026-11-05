"""Summarize six already audited human reconstructions without rerunning them."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
IDS=('000162','004028','006749','007129','008894','009466')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert (HERE/'queue_status.txt').read_text().splitlines()[-1]=='complete'
    rows=[]
    for sid in IDS:
        path=HERE/sid; report=json.loads((path/'human_report.json').read_text()); audit=json.loads((path/'human_audit.json').read_text())
        assert report['id']==sid and audit['human_report_sha256']==sha(path/'human_report.json')
        assert audit['artifact_sha256']==report['artifact_sha256']==sha(path/'human_rawfps.npz')
        assert audit['frames_verified']==report['frames']
        rows.append({'id':sid,'frames':report['frames'],'reference_fps':report['reference_fps'],
            'source_directory':report['source_parent'],'human_report_sha256':sha(path/'human_report.json'),
            'human_audit_sha256':sha(path/'human_audit.json'),'artifact_sha256':report['artifact_sha256'],
            'same_model_max_abs_error_m':audit['same_model_max_abs_error_m'],'training_ready':False})
    result={'cases':rows,'total_frames':sum(x['frames'] for x in rows),'training_ready':False,
            'limitations':['Reference FPS is metadata, not independently measured time.',
                'All source paths share one nominal recording directory; no independent-subject claim.',
                'Same-model human reconstruction check; no robot, text-semantic or physics acceptance.']}
    with (HERE/'batch_summary.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'cases':len(rows),'frames':result['total_frames'],'training_ready':False}))
if __name__=='__main__':main()

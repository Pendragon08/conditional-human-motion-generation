const f=require('fs'),p=__dirname+'/';let s=f.readFileSync(p+'stage3_supported_task_postopt_v3.py','utf8');const needle='        began=time.monotonic()';if(!s.includes(needle))throw Error('warm initialization insertion');
s=s.replace(needle,`        previous_dir=R/'stage3_supported_references_20261104/pilot_postopt_v2'/('candidate_'+str(m['candidate_id']).zfill(6))
        previous=json.loads((previous_dir/'report.json').read_text());warm=previous_dir/'solver_raw.npz'
        assert previous['sample_sha256']==sha(sample) and previous['solver_raw_sha256']==sha(warm)
        with np.load(warm,allow_pickle=False) as z:
            initial_q=np.clip(z['q'].astype(float),self.low+1e-6,self.high-1e-6).astype(np.float32);initial_root=z['root'].copy()
        assert initial_q.shape==(180,29) and initial_root.shape==(180,7) and np.isfinite(initial_q).all() and np.isfinite(initial_root).all()
        assert np.max(np.abs(np.linalg.norm(initial_root[:,:4],axis=1)-1))<1e-4
        initial_sha=sha(warm);initial_values_sha=hashlib.sha256(initial_q.tobytes()+initial_root.tobytes()).hexdigest()
`+needle);
s=s.split("'initial_state_sha256':initial_sha").join("'initial_state_sha256':initial_sha,'initial_values_sha256':initial_values_sha,'initialization':'previous raw fit clipped only for optimizer initialization'");f.writeFileSync(p+'stage3_supported_task_postopt_v4.py',s,{flag:'wx'});
let g=f.readFileSync(p+'stage3_supported_postopt_guard_v3.py','utf8');g=g.split('pilot_postopt_v3').join('pilot_postopt_v4').split('pilot_postopt_input_v3').join('pilot_postopt_input_v4').split('stage3_supported_task_postopt_v3.py').join('stage3_supported_task_postopt_v4.py');f.writeFileSync(p+'stage3_supported_postopt_guard_v4.py',g,{flag:'wx'});
let l=f.readFileSync(p+'launch_supported_postopt_v3_20261104.py','utf8');l=l.split('stage3_supported_postopt_guard_v3.py').join('stage3_supported_postopt_guard_v4.py').split('pilot_postopt_v3').join('pilot_postopt_v4').split('stage3_supported_task_postopt_v3.py').join('stage3_supported_task_postopt_v4.py');l=l.replace("ast.parse(p.read_text())","ast.parse(p.read_text())");f.writeFileSync(p+'launch_supported_postopt_v4_20261104.py',l,{flag:'wx'});

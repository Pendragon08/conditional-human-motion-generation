// Create an isolated resumable-by-batch sampler; retain the original2-reference baseline.
const fs=require('fs'), path=require('path');
const dir=__dirname;
let s=fs.readFileSync(path.join(dir,'stage3_sample_pick29.py'),'utf8');
function replace(a,b){if(!s.includes(a))throw Error('source differs: '+a.slice(0,60));s=s.replace(a,b);}
replace('import json\n','import json\nimport hashlib\nimport shutil\n');
replace("    output=parser.parse_args().output_dir\n    output.mkdir(exist_ok=False)",
`    parser.add_argument('--count',type=int,required=True)
    parser.add_argument('--start-index',type=int,default=0)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--seed',type=int,default=20261104)
    args=parser.parse_args()
    assert 1<=args.count<=1000 and 1<=args.batch_size<=4 and args.start_index>=0
    root_output=args.output_dir
    root_output.mkdir(exist_ok=False)
    records=root_output/'records.jsonl';records.open('x').close()
    def root_status(**value):
        temp=root_output/'status.tmp';temp.write_text(json.dumps(value,indent=2));temp.replace(root_output/'status.json')
    root_status(status='initializing',requested_candidates=args.count,accepted_references=0,PPO_started=False)
    output=root_output`);
replace('    frames=120;batch=2',`    frames=120;batch=min(args.batch_size,args.count)`);
replace('    world=np.broadcast_to(initial,(frames,22,3)).copy()\n    grasp=np.array([.35,-.20,.94]);lift=np.array([.35,-.20,1.10])',
`    world=np.broadcast_to(initial,(batch,frames,22,3)).copy()
    candidate_ids=list(range(start,start+batch))
    object_positions=np.array([np.array(SCENE['object_center'])+
        np.r_[np.random.default_rng(args.seed+i).uniform(-.06,.06,2),0.] for i in candidate_ids])
    grasp=object_positions+np.array([0.,0.,.04]);lift=object_positions+np.array([0.,0.,.20])`);
replace('        world[k,21]=','        world[:,k,21]=');
replace('target=torch.tensor(canonical,device=device,dtype=torch.float32)[None].repeat(batch,1,1,1)',
        'target=torch.tensor(canonical,device=device,dtype=torch.float32)');
replace('    began=time.monotonic();torch.manual_seed(20261103)',
`    began=time.monotonic();torch.manual_seed(args.seed+1048576+start)
    noise=torch.cat([torch.randn((1,263,1,frames),generator=torch.Generator(device=device).manual_seed(args.seed+i),
                                device=device) for i in candidate_ids])`);
replace('            clip_denoised=False,model_kwargs=', '            noise=noise,clip_denoised=False,model_kwargs=');
replace("metadata={'caption':prompt,'fps':20,'frames':frames,'checkpoint_sha256':expected,",
`metadata={'caption':prompt,'fps':20,'frames':frames,'checkpoint_sha256':expected,
                  'candidate_id':candidate_ids[k],'initial_noise_seed':args.seed+candidate_ids[k],
                  'denoising_batch_seed':args.seed+1048576+start,
                  'initial_noise_sha256':hashlib.sha256(noise[k].cpu().numpy().tobytes()).hexdigest(),
                  'feature_content_sha256':hashlib.sha256(features[k].cpu().numpy().tobytes()).hexdigest(),
                  'task_object_position':object_positions[k].tolist(),
                  'object_xy_distribution':'uniform+-0.06m within current table; project baseline, not paper full distribution',`);
replace("results.append({'path':str(p),'sha256':sha(p),'wrist_error_at_grasp_m':float(error[40]),",
`results.append({'path':str(p),'sha256':sha(p),'candidate_id':candidate_ids[k],
                        'initial_noise_sha256':metadata['initial_noise_sha256'],
                        'feature_content_sha256':metadata['feature_content_sha256'],
                        'task_object_position':object_positions[k].tolist(),
                        'wrist_error_at_grasp_m':float(error[40]),`);
const begin=s.indexOf('    frames=120;batch=min');
const end=s.indexOf('\n\n\nif __name__');
if(begin<0||end<0)throw Error('cannot locate candidate block');
let block=s.slice(begin,end).replace('frames=120;batch=min(args.batch_size,args.count)',
                                    'frames=120;batch=min(args.batch_size,args.start_index+args.count-start)');
s=s.slice(0,begin)+`    all_results=[]
    root_status(status='sampling',requested_candidates=args.count,generated_candidates=0,accepted_references=0,PPO_started=False)
    for start in range(args.start_index,args.start_index+args.count,args.batch_size):
        assert shutil.disk_usage('/data').free>8*1024**3+356*1024**2+100*1024**2,'sampling disk protection'
        output=root_output/('batch_'+str(start).zfill(6));output.mkdir(exist_ok=False)
`+block.split('\n').map(x=>'    '+x).join('\n')+`
        for item in results:
            with records.open('a') as f:f.write(json.dumps(item)+'\\n')
        all_results.extend(results)
        root_status(status='sampling',requested_candidates=args.count,generated_candidates=len(all_results),
                    accepted_references=0,latest_batch=start,PPO_started=False)
    assert len(all_results)==args.count
    assert len({r['candidate_id'] for r in all_results})==args.count
    assert len({r['initial_noise_sha256'] for r in all_results})==args.count
    assert len({r['feature_content_sha256'] for r in all_results})==args.count
    final={'status':'sampling_completed_not_postoptimized','requested_candidates':args.count,
           'generated_candidates':len(all_results),'accepted_references':0,
           'records_sha256':sha(records),'checkpoint_sha256':expected,'scene_sha256':SCENE_SHA,
           'PPO_started':False,'note':'candidate counts are not admitted reference counts'}
    with (root_output/'report.json').open('x') as f:json.dump(final,f,indent=2)
    root_status(**final)
`+s.slice(end);
fs.writeFileSync(path.join(dir,'stage3_sample_pick29_batch.py'),s);
console.log('wrote isolated batched sampler');

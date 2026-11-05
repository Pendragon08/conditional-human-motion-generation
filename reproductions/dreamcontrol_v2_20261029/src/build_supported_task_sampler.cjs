const fs=require('fs'),dir=__dirname+'/';
let s=fs.readFileSync(dir+'stage3_sample_pick29_batch.py','utf8');
function replace(a,b){if(!s.includes(a))throw Error('missing '+a);s=s.replace(a,b)}
replace('from g1_fk import URDFFK','from g1_fk import URDFFK\nfrom stage3_supported_task_conditions import build as task_conditions, TASKS');
replace("parser.add_argument('--count',type=int,required=True)","parser.add_argument('--count',type=int,required=True)\n    parser.add_argument('--task',choices=TASKS,required=True)");
const first=s.indexOf('        frames=120;batch=');const end=s.indexOf('        # Canonical X',first);if(first<0||end<0)throw Error('bounds');
s=s.slice(0,first)+`        frames=180;batch=min(args.batch_size,args.start_index+args.count-start)
        candidate_ids=list(range(start,start+batch))
        conditions=task_conditions(args.task,candidate_ids,args.seed,initial)
        world=conditions['world'];object_positions=conditions['object_initial']
        scene=conditions['scene'];scene_sha=conditions['scene_sha256']
`+s.slice(end);
const m=s.indexOf('        mask=torch.zeros(batch,frames,22,');const me=s.indexOf('        spatial_mean=',m);if(m<0||me<0)throw Error('mask');
s=s.slice(0,m)+"        mask=torch.tensor(conditions['mask'],device=device,dtype=torch.bool)\n"+s.slice(me);
replace("prompt='Pick up the object from the table.'","prompt=conditions['prompt']");
s=s.split("'scene_sha256':SCENE_SHA").join("'scene_sha256':scene_sha").split("'scene':SCENE").join("'scene':scene");
replace("'object_xy_distribution':'uniform+-0.06m within current table; project baseline, not paper full distribution'","'object_xy_distribution':'uniform+-0.025m project tabletop pilot, not a published full distribution'");
replace("'requested_constraint_points':[0,4,5,21]","'requested_constraint_points':([0,10,11,21] if args.task=='pour' else [0,10,11,20,21]),\n                      'task':args.task,'task_plan':conditions['plans'][k],\n                      'task_conditions_source_sha256':sha(Path(__file__).with_name('stage3_supported_task_conditions.py'))");
replace("metadata_utf8=np.frombuffer(json.dumps(metadata).encode(),dtype=np.uint8))","desired_object_positions=conditions['desired_object_positions'][k],\n                    desired_wrist_quat_wxyz=conditions['desired_wrist_quat_wxyz'][k],planned_hand_closed=conditions['planned_hand_closed'][k],\n                    metadata_utf8=np.frombuffer(json.dumps(metadata).encode(),dtype=np.uint8))");
replace('Generate task-specific Pick candidates','Generate three-task position-conditioned candidates');
fs.writeFileSync(dir+'stage3_sample_supported29_batch.py',s,{flag:'wx'});

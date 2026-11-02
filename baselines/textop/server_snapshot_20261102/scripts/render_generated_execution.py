"""Truthful generated-execution provenance around the unchanged Mesa renderer."""
import argparse,json,pathlib,re,sys
from generated_pipeline_protocol import gate,sha
import render_quick_execution_video as visual
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();sp=ROOT/'configs/generated-execution-video-spec.json';spec=json.loads(sp.read_text())
assert spec['generated_video_enabled'] and sha(__file__)==spec['wrapper_sha256'] and re.fullmatch('[a-z0-9_]+',a.run)
def generated_trace(pipeline_run,trial_number,render_spec):
 assert pipeline_run==spec['pipeline_run'] and trial_number==0 and render_spec['actual_rendering_enabled']
 sdkp=ROOT/'configs/quick-generated-sdk-spec.json';sdk=json.loads(sdkp.read_text());assert sha(sdkp)==spec['generated_sdk_spec_sha256'];gate(ROOT,sdk)
 root=ROOT/'generated_execution_private'/pipeline_run;fp=root/'pipeline.json';combined=json.loads(fp.read_text())
 assert sha(fp)==spec['pipeline_receipt_sha256'] and combined['status']=='GENERATED_REFERENCE_PIPELINE_TRIAL_CLOSED' and combined['output_writers_closed']
 for stage in ['export','replay']:
  rp=root/stage/'receipt.json';ep=ROOT/'receipts'/('generated_pipeline_'+pipeline_run+'_'+stage+'_exit.json');r=json.loads(rp.read_text());ex=json.loads(ep.read_text())
  assert ex['evaluation_receipt_sha256']==sha(rp) and ex['supervision_passed'] and ex['child_exit_code']==0 and ex['child_reaped'] and not ex['timed_out']
  assert sha(ROOT/'receipts'/('generated_pipeline_'+pipeline_run+'_'+stage+'_process_exit.json'))==ex['process_exit_sha256'] and sha(ROOT/'logs'/('generated_pipeline_'+pipeline_run+'_'+stage+'_resources.jsonl'))==ex['resources_sha256']
  assert r['source_sha256']==sdk['executor_sha256'] and r['spec_sha256']==sha(sdkp) and r['input_generated_npz_sha256']==sdk['input']['sha256'] and r['output_writers_closed']
  bound=next(x for x in combined['phases'] if x['stage']==stage);assert bound['receipt_sha256']==sha(rp) and bound['exit_sha256']==sha(ep)
 replay=root/'replay';r=json.loads((replay/'receipt.json').read_text());assert r['status']=='GENERATED_REFERENCE_PHYSICS_TRIAL_FINISHED' and r['optimizer_updates']==0 and r['teacher_completed_iterations']==5000
 from quick_pilot_protocol import load_teacher,authorize_teacher
 quick=json.loads((ROOT/'configs/quick-pilot-spec.json').read_text());cp=authorize_teacher(quick,load_teacher(ROOT,quick));assert r['teacher_checkpoint_sha256']==cp['sha256']
 index=replay/'executed_traces.private.json';assert sha(index)==r['private_trace_index_sha256'];entries=json.loads(index.read_text());assert len(entries)==1
 entry=entries[0];folder=pathlib.Path(entry['folder']);assert folder.resolve()==folder and folder.parent==replay/'execution_private'
 meta=json.loads((folder/'trace.private.json').read_text());path=folder/'executed_poses.private.npz'
 assert meta['runtime_kind']=='isaaclab_physics' and meta['status']=='FINISHED_PHYSICAL_TRIAL' and meta['phase']=='INITIAL_THEN_PRE_AUTORESET_TERMINATION_COMPUTE'
 assert sha(folder/'poses.private.bin')==meta['binary_sha256'] and sha(path)==meta['npz_sha256']==entry['npz_sha256']
 assert meta['samples']==entry['samples']==meta['trial']['control_steps']+1 and meta['trial']==r['trial']
 assert len(render_spec['software_render_witnesses'])==2
 for witness in render_spec['software_render_witnesses']:
  rp=ROOT/'video_validation'/witness['run']/'receipt.json';ep=ROOT/'receipts'/('execution_video_'+witness['run']+'_exit.json');wr=json.loads(rp.read_text());ex=json.loads(ep.read_text())
  assert sha(rp)==witness['receipt_sha256'] and sha(ep)==witness['exit_sha256'] and ex['evaluation_receipt_sha256']==sha(rp) and ex['child_exit_code']==0 and ex['supervision_passed']
  assert wr['source_sha256']==render_spec['renderer_sha256'] and wr['render_signature']==visual.render_signature(render_spec) and wr['nvidia_libraries_loaded']==[] and 'llvmpipe' in wr['gl_renderer'].lower()
 return path,meta,{'source_kind':'trained_generated_reference_actual_isaaclab_execution','pipeline_run':pipeline_run,'pipeline_sha256':sha(fp),'replay_receipt_sha256':sha(replay/'receipt.json'),
  'generated_input_sha256':sdk['input']['sha256'],'execution_trace_sha256':sha(path),'wrapper_sha256':sha(__file__),'render_spec_sha256':sha(sp),'reference_animation_used':False,'real_time_interactive_closed_loop_verified':False}
visual.actual_trace=generated_trace
# Only the provenance callback changes. The exact seeded/independently validated
# visual kernel, all geometry/light/plane/camera/encoding/timing math stay intact.
sys.argv=[str(ROOT/'scripts/render_quick_execution_video.py'),'--spec',str(sp),'--mode','actual','--run',a.run,'--pilot-run',spec['pipeline_run'],'--trial','0']
visual.main()

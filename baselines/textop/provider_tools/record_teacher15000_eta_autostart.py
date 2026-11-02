"""Record measured ETA and the user's unconditional post-filter authorization."""
import pathlib,json,math,datetime,hashlib,statistics,shutil
B=pathlib.Path(__file__).resolve().parents[1];D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
O=B/'results/teacher15000_direct_strict_20261027';F=O/'teacher15000_direct_full_v1'
items=[]
for name in ['full_observe_v2.json','full_observe_v3.json','user_eta_v1.json']:
 p=F/name;j=json.loads(p.read_text(encoding='utf-8'));r=j['files']['receipt']
 assert r['status']=='RUNNING_TEACHER15000_STRICT' and r['optimizer_updates']==0
 items.append((j['observed_unix'],r['actual_vectorized_control_calls'],r,p))
rates=[(b[1]-a[1])/(b[0]-a[0]) for a,b in zip(items,items[1:])]
rate=statistics.median(rates);now,calls,r,path=items[-1]
remaining=max(0,r['planned_maximum_control_calls']-calls)
seconds=remaining/rate;lo=math.floor(seconds/1800)*1800;hi=math.ceil(seconds/1800)*1800
tz=datetime.timezone(datetime.timedelta(hours=8));stamp=lambda t:datetime.datetime.fromtimestamp(t,tz).isoformat()
eta=dict(schema='textop.measured_strict_eta.v1',sample_beijing=stamp(now),run='teacher15000_direct_full_v1',
 source_progress_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
 finished=r['finished_trials'],passed=r['complete_trials'],failed=r['failed_trials'],pending=8278-r['finished_trials'],
 recent_control_calls_per_second=rates,median_control_calls_per_second=rate,
 planned_maximum_control_calls=r['planned_maximum_control_calls'],completed_control_calls=calls,
 remaining_planned_calls_conservative=remaining,estimated_remaining_seconds=seconds,
 estimated_finish_beijing=stamp(now+seconds),rounded_window_beijing=[stamp(now+lo),stamp(now+hi)],
 assumptions=['Recent throughput continues on shared GPU','Longer motions follow shorter motions; action count is not a linear work measure',
 'Remaining maximum planned vector calls overestimates work if all environments terminate early in a batch'],
 scope='Full physical replay only; closed-result audit, accepted corpus and generator launch follow automatically')
(O/'eta_user_request_20261027.json').write_text(json.dumps(eta,indent=2),encoding='utf-8')
plan=dict(schema='textop.direct_strict_to_generator_autostart.v1',
 latest_user_request='还要多久能完成全部物理筛选？筛完后直接用通过的动作启动生成器训练，不用再问我。',
 autostart_authorized=True,additional_user_confirmation_required=False,
 strict_run='teacher15000_direct_full_v1',strict_spec_sha256='91184fff75f8a0579a759abb418d4a0186d3e42a9a48c176e3d33facc768a1cd',
 checkpoint_completed=15000,checkpoint_sha256=r['policy_checkpoint_sha256'],source_total=8278,
 teacher_must_remain_stopped=True,prefilter_8100_resume_authorized=False,
 input_selection='Actual standard strict complete passes only; original retained REFERENCE plus aligned BABEL; mirror augmentation follows source family',
 physical_thresholds_changed=False,historical_ema_bypass_disclosed=True,
 requires_before_corpus=['All8278 finished or explicit failure accounted; no unattempted classified failed',
 'Original SDK/process/supervisor/outer actual exits and closed writers; common lock released',
 'Private trial journal/finite poses/hashes/counts and original source provenance accepted'],
 requires_before_generator=['Nonempty actual-pass motion-text corpus','Source and mirror family split; exact pose leakage check',
 'TRAIN-only69D statistics; complete real CLIP embeddings','Actual loading, gradient, optimizer/RNG restore acceptance',
 'Persistent checkpoint/log budget and free disk/RAM/GPU/commonlock acceptance'],
 training={'fresh_filtered_vae_primitive_steps':200000,'then_fresh_filtered_ldm_primitive_steps':300000,'history':2,'future':8,'primitives':4},
 generator_launch_enabled=False,actual_full_filter_complete=False,filtered_corpus_accepted=False,generator_started=False,
 current_stage='FULL_STRICT_RUNNING_HANDOFF_PREPARATION',
 preparation_findings=['Existing generator8278 corpus and VAE entry are explicitly PREFILTER-only and must not be reused unchanged or relabeled',
 'Reuse unchanged numerical VAE runtime only through a new actual-pass corpus/provenance entry and new acceptance witnesses',
 'Full LDM entry and persistent future capacity are still pending actual implementation/acceptance; these are automatic work, not user approval requests'],
 latest_eta_evidence='results/teacher15000_direct_strict_20261027/eta_user_request_20261027.json')
p=B/'configs/teacher15000-full-to-generator-autostart-20261027.json';p.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
text='''\n\n用户再次明确授权（2026-10-27）：全部物理筛查结束后直接用实际通过动作启动新生成器，不再询问。授权记录见configs/teacher15000-full-to-generator-autostart-20261027.json。实际完整回放退出、训练集和容量验收由系统自动完成；当前未筛选8100步状态不接续。生成器尚未启动，语料验收尚未完成。\n'''
for name in ['reports/teacher15000_direct_strict_20261027.md']:
 q=B/name;old=q.read_text(encoding='utf-8')
 if text not in old:q.write_text(old+text,encoding='utf-8')
for src in [O/'eta_user_request_20261027.json',p,B/'reports/teacher15000_direct_strict_20261027.md']:
 dst=D/src.relative_to(B);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert src.read_bytes()==dst.read_bytes()
print(json.dumps(eta,ensure_ascii=False))

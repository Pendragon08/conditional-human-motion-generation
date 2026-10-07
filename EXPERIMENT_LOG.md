# New candidate experiment log

## 2026-10-06 — Execution feedback F / world-model W

- Human user authorized CLoSD-style feedback transfer to DreamControl and literature-grounded world-model feasibility ablations.
- Primary-source literature audit and independent ablation review completed; proposed novelty remains unproven.
- Existing frozen Kick/Punch policies and original references are preserved. No extra post-retarget IK; no PPO update in the initial module pilot.
- New environment invocation uses unchanged MuJoCo spec `91455383`; fresh doc-follow witness and actual outputs must be checked before launch.
- Protocol: `plans/feedback_world_model_ablation_20261006.md`; predictive config: `configs/feedback_world_model_pilot_20261006.json`.
- Planned next steps are not outcomes: F representation bridge, W data/forecast ablations, control upper-bound/intervention, then F×W only if individual gates pass.

### Actual module execution

- Primary-source evidence reviewed for CLoSD/DreamControl and9 officially published world-model-related papers. Full-CloSD-prefix versus sparse-G1 feedback, dynamics versus hidden-current-state estimation kept separate.
- Fresh environment witness PASS for exact frozen Kick/Punch tensors, observations, real moving physics/EGL, and CUDA backprop. Probe20episodes9083transitions29.60s was not used as training data.
- Collection started10:15:51UTC, completed10:39:49UTC:1096episodes;260076training,70363validation,140881test transitions plus14733heldout push transitions. Full-reference splits/finite states/fall records retained.
- First absolute sparse-reference calibration FAIL (~9.6cmmean); origin/delta follow-up test exploration improved reference-only errors, preserved both. Residual-v3 zero-identity/translation fixtures and input provenance checked.
- F planned/actual/shuffled exactly once,3Kicksamples perarm,seed42,DDPM1000. Shape/finite/inputhintagreement checks PASS. Actual context fit8.20→3.43cm; shuffled15.31cm. Originalgoal errors17.36/16.89/15.69cm; cannotclaimfeedback improves generalgoal/control. Batchlatencies94.70/107.27/93.74s, noonline orpostgenerationrobotexecution.
- W4neuralarms×3seed×2500updates completed plusridge;13checkpointSHA verified. H10 per-coordinate body FK macroRMSE multi-step1.939cm vsridge2.301cm, state-only2.445cm. Relative improvement15.73% [12.69,18.85]% and20.72% [17.47,23.75]%; coveragecomplete, predictivePASS. Ensemble notprimary. No learned controlbenefit measured.
- Procedural deviations: fitauto-start10:39:50UTC preceded fresh fit-kernel witness10:42:25UTC. Wrapper modified while active; Python saved complete results, butbash cursor then encountered new token and launcherexit2. OriginalFAILED state/logpreserved. No repeatedfit; execution_audit.json separates valid computedfiles from abnormal launcher.
- Oracle physics-only interface freshwitness PASS onvalidationref36 bothtasks; exactcomplete-stateclone/actualstateuntouched/boundedcosts. At10:54:39UTC launched unique tmux r3_feedback_world_oracle_20261006, frozenPPO/simple/oracle,40heldoutreferences and3candidate seeds, no PPO or extraIK. Its results are pending and cannot be substituted with predictivePASS.
- Rawarrays/newweights retained in privateignored artifacts; smallresults, source/config/doc/reports intended for newprivateGitHub only. Baseline results andteacherreports unchanged.

### Oracle outcome and learned-control launch

- Physicsoracle completed200trials andnormalexit0 at11:03:31UTC. RecoveryAUC .3191559(PPO)/.2735308(simple)/.2244945(oracle); virtualsuccess .425/.45/.491667; fall .4/.275/.316667. FormalgatePASS:29.66% vsPPO [10.89,41.79]% referenceCI. Independentposthoc diagnosticvsSimple17.93% [1.32,30.94]% butKickAUC worsethanSimple, noall-task/metricdominanceclaim.
- Codefixedshortfallpenalty4*remainingsteps disclosed; fixedcandidate oracleisnotglobal upperbound onworldmodeldesigns. Timing~140ms/decision, offline synchronousonly.
- Newfresh learned-control docwitness PASS exact invocation onvalidationref36, multi5/single/stateonly forKick/Punch: fittedmodelSHA, actualobs/batchedPPO parity, finitecosts, actualstateuntouched, stateonlychooseszeroresidual. Stateonlycontroller analyticallyPPO-identical, thus no redundantphysicaltest.
- At11:13:23UTC launchedunique r3_feedback_world_learned_control_20261006,240trials (multi5/single×3seeds×40refs) withunchangedcandidates/H10/cost/init/refs/push. PPO/simple/oracle200previousresultsreused. No newmodeltraining/PPO/IK/feedbackgeneration. Results pending; mustassess actualperformance, notinferfrompredictivegate.

### Learned-control final outcome

- At11:22:52UTC uniquequeueCOMPLETE normalexit0; Pythonwall566.10s. All240trials unique/complete, configuration/script/model/oracle receipts verified. Physicsoracle200trials reused, not rerun; totalcomparison440trials.
- Preregisteredprimarymulti5 controlgateFAIL: AUC .270613 vsPPO.319156 (15.21%,95%referenceCI[-4.72,28.33]%),simple.273531 (1.07%,CI[-16.12,13.87]%). Goal47.5%vs42.5% passed; falls42.5%vs40% failed. No swap toone-step auxiliary orretuning.
- Kickmulti5AUC.255598 worsePPO.243775/simple.193712, falls45%; Punch.285628 betterpointmeanvsPPO.394537/simple.353349, falls40%. Multi5vsSingle3.52%CI[-8.76,15.60]% isnotcontrolproof formulti-steptraining.
- Candidatescoringonly:multi5mean72.32ms,max250.00ms,25/2132decisionsexceed100ms(1.17%). Noonline/50Hz/hardwareclaim. Threecoupledmodel-candidateseedresults andtasksplit preserved.
- Freshread-onlyreviewer confirmed finalgateFAIL andappropriate stop. Cross-run prefixdeterministic reconstruction isnot per-reference bytehashverification; sameexposed WMtestreferences reused inallstages. Proposedfailurecausesremainhypotheses.
- Stopcurrentinterfaceexpansion; do notstartF×W/formalPPO. Fnominal sparsefeedback only, nofullprefix orphysicalcontrol. Newreports/results/code will becommitted tothenewprivate repository; teacherbaselineHTML remainsunchanged.


## 2026-10-06 — Isaac feedback continuation and world-model PPO design

- User explicitly requests continued F verification, W+PPO improvement, Isaac first then MuJoCo. Existing baseline training unchanged.
- Original upstream retarget/refinement completed once perplanned/actual/shuffled:3/3,2/3,2/3. NoextraIK; retargetgateFAIL.
- Fresh documented capture once500×18 normalexit0,6originalrefs49–54,80N/0.2s world-y push; allfinite,noreset/nofall. Duplicate maxq0.37763rad allrun and0.15148rad branch; notexactPhysX clone.
- Fresh revised smoke once500×15 normalexit0; generated outputs frompriornominalMuJoCo sparse contexts, no perturbation. Fixed/planned/actual/shuffled goalhits2/3,2/3,1/3,0/3. Missingref47actual/shuffled keptfailed; wrongref48 fallsstep105. Originaltargets/clockfixed,noatbranchrobotreset. Prefixqmax0.185325rad; formalcausalFnot evaluated.
- Independentwitness arrays/hashes andpreservedcapture receipts verified. Capturegenericregenerated=true namingbug retained/explicitlycorrected interpretation; revisedsource only afterexit. Independentcandidatepickle analysis tool missingjaxlie preserved; no install.
- W+PPO reviewerled plan uses version-frozenstop-gradient predictive representations andtrueIsaacreward/done/GAE, fivecontrolledarms/3seeds, notimaginedPPO.98,304,000transitions/run proposed, notexecuted; extra modelcomputeexplicit. NoformalPPO/newviewGate started.
- Report23/resultscode preservefailure/stoprules, oldMPCFAILuntouched. Innovationnotvalidated; baseline-onlyteacherHTMLunchanged.

## 2026-10-07 — World-model PPO prediction stop and residual development

- v1 real Isaac shared warmup completed11,796,480 unique transitions/25resets;9GRU fits plusridge. Three seeds H10root-frameFK failed>=10%improvement vsridge; formal5armPPO never launched. Correctv3 smoke16env48steps really performed2diagnosticPPO/1auxupdates; originalnonprivilegedcritic normalization corrected, oldv1/v2 failures preserved. Archive e55daae pushed.
- User continued execution. Independent reviewer required direct-GRU, action-zeroinbothbranches,4-reference-fold OOF residualshuffle, oldtestretirement, update0checkpointchoice, and actualgradient/optimizerreceipts. Newboundedstageuses44train/11validation only; noindependenttestorformalPPO claims.
- Fresh disturbance probe exactly once480x16=7680, exit0,10forcebufferchecks8x80N,error0;4realresets,1recoveryreset.11arrayhashes/size/finite anddone/episodeauditPASS. Wrappermodeprobe/internalsummarycollect distinction explicit; oldprogresslast457doesnotreplacefinal480.
- Newunique tmux r3_world_residual_development_20261007 started2026-10-07 05:57UTC. Seeded residualCUDA kernelPASS; queueRUNNING collectseed42 observed. Planned3sharedcaptures8192x480,4GRUfits/seed each1000updatesbatch1024. NoWMimaginedPPO, extraIK, orbaseline rerun. Current independentactualscene monitorGPU1; computeGPU0.

- Residual development outputs completed for3seeds:11,796,480transitions/5121resets and12,000CUDA updates; reference-paired development improvement vsRidge22.01/22.37/22.70%. Validation selects checkpoints, not independent proof. Supplemental doc audit:6wrapperreturns0+sentinels, actualchildexitunknown due delayed rc capture; no rerun.
- Fresh32sample Kick confirmation generation once seed20261007, frozenofficialOmniControl plus byte-identical originalretarget/refine:22staticaccepted,10officialrejects,11first-IDselected+11eligibleunselected, noextraIK. Torchbackward/JAXJIT and independent static/dedup hashes PASS. Predictionconfirmation/PPO gates stillpending.

- Confirmation firstdocument transportattempt blockedbeforequeue: PowerShellpipeCRLF→tmuxredirection;0byte log,no session/state/model_lock/data. Fixed localbase64/subprocess transport preserves originalattempt; second witnessread-only pausedbeforelaunch to correct metricwording. Actualdev+confirmationcode metricalwayssqrt(mean(windowMSE))perreference→referenceequalmean; protocolwording correctedandoldtextarchivedbeforeanyconfirmationdata/result, no code/thresholdchange. Freshtransportv3doc-follow requested.

## 2026-10-07 新动作残差预测确认与PPO准备

新生成32条Kick，原重定向静态接受22，按预注册编号选择11；无追加IK或物理结果选择。三seed42/43/44各983,040真实转移，总2,949,120，reset1,948；六child自然rc0。独立审计复核36输出哈希、67,584窗口逐项重建、bootstrap差异0，clean/recovery全PASS。恢复H10根相对FK预测比动作条件Ridge改善20.26%/19.05%/22.08%。详见报告26与结构化summary。

这是预测确认，0PPO/aux/norm更新，不是控制收益或创新验证成功。旧预测FAIL、旧开发退出码回执缺陷、运输/初始化阻塞均保全，不改阈值重测。主五臂PPO＋两个必要解释臂审查见plans/world_residual_ppo_ablation_review_20261007.md。当前新runner仅进入独立256环境×2更新诊断；正式8192×500×3seed未启动。监控label按实际stage/budget显示，不将预检或预测采集写成训练。正式前须smoke、8192容量及本轮实际场景本地确认。

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

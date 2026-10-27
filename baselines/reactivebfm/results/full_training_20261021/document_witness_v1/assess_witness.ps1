$ErrorActionPreference = 'Stop'
$evidenceRoot = $PSScriptRoot
$repoRoot = 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/reactivebfm_reproduction_20261021/repository'
$utf8 = [Text.UTF8Encoding]::new($false)
function Read-Json($relative) { return (Get-Content -LiteralPath (Join-Path $evidenceRoot $relative) -Raw | ConvertFrom-Json -Depth 50) }
function Hash-Text($value) { return [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($utf8.GetBytes($value))).ToLowerInvariant() }
function Parse-RemoteManifest($relative) {
    $map = [ordered]@{}
    foreach ($line in [IO.File]::ReadAllLines((Join-Path $evidenceRoot $relative))) {
        if ($line -notmatch '^([0-9a-f]{64})  (.+)$') { throw 'Malformed remote manifest.' }
        $map[$Matches[2]] = $Matches[1]
    }
    return $map
}
function Compare-Maps($before, $after) {
    $differences = [Collections.Generic.List[object]]::new()
    foreach ($path in @(@($before.Keys) + @($after.Keys) | Sort-Object -Unique)) {
        if (-not $before.Contains($path)) { $differences.Add(@{path=$path;kind='added';after=$after[$path]}) }
        elseif (-not $after.Contains($path)) { $differences.Add(@{path=$path;kind='removed';before=$before[$path]}) }
        elseif ($before[$path] -ne $after[$path]) { $differences.Add(@{path=$path;kind='changed';before=$before[$path];after=$after[$path]}) }
    }
    return ,$differences.ToArray()
}
function Set-Digest($map, $paths) { return Hash-Text ((@($paths | Sort-Object | ForEach-Object { $_ + "`t" + $map[$_] }) -join "`n") + "`n") }
$invocation = Read-Json 'documented_run.invocation.json'
$receipt = Read-Json 'remote_receipts/receipt.json'
$rank0 = Read-Json 'remote_receipts/ddp/status_rank0.json'
$rank1 = Read-Json 'remote_receipts/ddp/status_rank1.json'
$parity = Read-Json 'remote_receipts/parity.json'
$hold = Read-Json 'remote_receipts/hold/receipt.json'
$loop = Read-Json 'remote_receipts/loop/receipt.json'
$metadata = Read-Json 'remote_receipts/loop/session/online_motion_20261021_160723/metadata.json'
$config = Get-Content -LiteralPath (Join-Path $repoRoot 'baselines/reactivebfm/configs/full_training_v1.json') -Raw | ConvertFrom-Json
$checks = [Collections.Generic.List[object]]::new()
function Check($name, $expected, $actual, $passed) { $checks.Add([ordered]@{name=$name;expected=$expected;actual=$actual;passed=[bool]$passed}) }
Check 'exit_code' 0 $invocation.exit_code ($invocation.exit_code -eq 0)
Check 'receipt_status' 'FULL_RUNTIME_WITNESS_PASSED' $receipt.status ($receipt.status -eq 'FULL_RUNTIME_WITNESS_PASSED')
foreach ($rank in @($rank0,$rank1)) {
    Check ('rank'+$rank.rank+'_status') 'PROBE_COMPLETED with exit0' @($rank.status,$rank.exit_code) ($rank.status -eq 'PROBE_COMPLETED' -and $rank.exit_code -eq 0)
    Check ('rank'+$rank.rank+'_optimizer_updates') 4 $rank.optimizer_updates ($rank.optimizer_updates -eq 4 -and $rank.final_official_step - $rank.official_step_at_start -eq 4)
    Check ('rank'+$rank.rank+'_global_batch') 1024 $rank.batch_size_global ($rank.batch_size_global -eq 1024 -and $rank.world_size -eq 2 -and $rank.configuration.batch_size_local -eq 512)
    Check ('rank'+$rank.rank+'_frozen_configuration') 'equal to full_training_v1.json' $rank.configuration (($rank.configuration | ConvertTo-Json -Depth 30 -Compress) -eq ($config | ConvertTo-Json -Depth 30 -Compress))
    $components = @($rank.dataset_component_lengths.PSObject.Properties | Where-Object {$_.Value -gt 0} | ForEach-Object {$_.Name} | Sort-Object)
    Check ('rank'+$rank.rank+'_three_dataset_components') @('100style_g1_36dim','amass_g1_36dim','kungfu_g1_36dim') $rank.dataset_component_lengths (($components -join ',') -eq '100style_g1_36dim,amass_g1_36dim,kungfu_g1_36dim')
}
Check 'gradient_parity' 'FP32 parity pass, atol1e-6/rtol1e-5, >=200 nonzero tensors' $parity ($parity.status -eq 'RECOMPUTATION_GRADIENT_PARITY_PASSED' -and $parity.atol -eq 1e-6 -and $parity.rtol -eq 1e-5 -and $parity.nonzero_gradient_tensors -ge 200 -and $parity.gradient_tensors_compared -ge 200 -and $parity.maximum_absolute_gradient_error -le 1e-6)
Check 'static_hold' '8 simulated seconds, finite/upright' $hold ($hold.status -eq 'STATIC_REFERENCE_PHYSICS_EXECUTED' -and $hold.finite -and $hold.upright_entire_rollout_height_proxy -and [Math]::Abs($hold.simulation_seconds - 8) -lt 1e-6 -and $hold.control_steps -eq 400 -and -not $hold.reference_forcing)
Check 'closed_loop_control' '100 actual policy calls' $loop.policy_steps ($loop.policy_steps -eq 100 -and $loop.physics_steps -eq 100 -and $receipt.closed_loop_policy_steps -eq 100)
Check 'closed_loop_integrations' '400 actual MuJoCo integrations / 2 simulated seconds' @($loop.mujoco_integrator_steps,$loop.simulation_seconds) ($loop.mujoco_integrator_steps -eq 400 -and $receipt.closed_loop_integrator_steps -eq 400 -and [Math]::Abs($loop.simulation_seconds - 2) -lt 1e-6)
Check 'closed_loop_planner_calls' '>=2' @($loop.planner_calls).Count (@($loop.planner_calls).Count -ge 2 -and @($loop.planner_calls).Count -eq $receipt.closed_loop_planner_calls)
Check 'closed_loop_feedback' 'real state prefixes, no reference forcing, stop at max policy steps' @($loop.replan_prefix_mode,$loop.reference_forcing,$metadata.stop_reason) ($loop.replan_prefix_mode -eq 'real' -and -not $loop.reference_forcing -and $metadata.stop_reason -eq 'max_policy_steps_reached' -and @($loop.planner_calls.prefix_sha256 | Sort-Object -Unique).Count -eq 12)
$localBefore = Read-Json 'local_before.json'
$localAfter = Read-Json 'local_after.json'
$localBeforeMap = [ordered]@{}
$localAfterMap = [ordered]@{}
foreach ($e in $localBefore) { $localBeforeMap[$e.path] = $e.sha256 }
foreach ($e in $localAfter) { $localAfterMap[$e.path] = $e.sha256 }
$remoteBefore = Parse-RemoteManifest 'remote_sources_before.stdout.txt'
$remoteAfter = Parse-RemoteManifest 'remote_sources_after.stdout.txt'
$localDifferences = Compare-Maps $localBeforeMap $localAfterMap
$remoteDifferences = Compare-Maps $remoteBefore $remoteAfter
Check 'local_source_config_doc_ledger_unchanged' 'no added/removed/changed files' $localDifferences ($localDifferences.Count -eq 0)
Check 'remote_supplementary_and_vendor_source_unchanged' 'no added/removed/changed files' $remoteDifferences ($remoteDifferences.Count -eq 0)
$sourceSets = [Collections.Generic.List[object]]::new()
foreach ($group in @($localBefore | Group-Object set)) {
    $paths = @($group.Group.path)
    $sourceSets.Add([ordered]@{location='local';set=$group.Name;file_count=$paths.Count;before_sha256=(Set-Digest $localBeforeMap $paths);after_sha256=(Set-Digest $localAfterMap $paths)})
}
foreach ($prefix in @('scripts/','upstream/','configs/')) {
    $paths = @($remoteBefore.Keys | Where-Object {$_.StartsWith($prefix)})
    $sourceSets.Add([ordered]@{location='remote';set=$prefix.TrimEnd('/');file_count=$paths.Count;before_sha256=(Set-Digest $remoteBefore $paths);after_sha256=(Set-Digest $remoteAfter $paths)})
}
$crossMismatches = [Collections.Generic.List[object]]::new()
$localOnly = [Collections.Generic.List[string]]::new()
foreach ($e in $localBefore | Where-Object {$_.set -in @('supplementary_source','official_vendor_source','configuration')}) {
    $relative = $e.path.Replace('baselines/reactivebfm/','')
    if (-not $remoteBefore.Contains($relative)) {$localOnly.Add($relative)}
    elseif ($remoteBefore[$relative] -ne $e.sha256) {$crossMismatches.Add(@{path=$relative;local=$e.sha256;remote=$remoteBefore[$relative]})}
}
Check 'deployed_source_bytes_match_local' 'all 239 deployed files match local' $crossMismatches ($crossMismatches.Count -eq 0 -and $remoteBefore.Count -eq 239)
Check 'rank_source_receipts_match_manifest' 'train_full.py receipt source hash matches frozen remote source' @($rank0.source_sha256,$rank1.source_sha256) ($rank0.source_sha256 -eq $remoteBefore['scripts/train_full.py'] -and $rank1.source_sha256 -eq $remoteBefore['scripts/train_full.py'])
Check 'loop_source_receipt_matches_manifest' 'run_closed_loop.py receipt source hash matches frozen remote source' $loop.source_sha256 ($loop.source_sha256 -eq $remoteBefore['scripts/run_closed_loop.py'])
$copied = @(Read-Json 'copied_evidence.json')
$copiedBinary = @(Get-ChildItem -LiteralPath (Join-Path $evidenceRoot 'remote_receipts') -Recurse -File | Where-Object {$_.Extension -in @('.npz','.npy','.pt','.pth','.pkl','.pickle')})
Check 'text_evidence_only' '21 JSON/text files, no checkpoint/pose binaries copied' @($copied.Count,$copiedBinary.Count) ($copied.Count -eq 21 -and $copiedBinary.Count -eq 0)
$allPassed = @($checks | Where-Object {-not $_.passed}).Count -eq 0
$result = [ordered]@{
    status=$(if($allPassed){'DOCUMENT_WITNESS_PASSED'}else{'DOCUMENT_WITNESS_FAILED'})
    scope='One exact documented runtime invocation; execution check only, not full training or paper-performance reproduction.'
    setup_sources=@('baselines/reactivebfm/docs/full_runtime.md','baselines/reactivebfm/configs/full_training_v1.json','baselines/reactivebfm/configs/env_spec.json','.aris/compute/reactivebfm-20261021.md')
    invocation_count=1
    invocation=$invocation
    preflight='output_absent.invocation.json / output_absent.stdout.txt; absent immediately before launch'
    source_sets=$sourceSets.ToArray()
    source_changes=@{local=$localDifferences;remote=$remoteDifferences;cross_host_hash_mismatches=$crossMismatches.ToArray();local_helpers_not_deployed=$localOnly.ToArray()}
    checks=$checks.ToArray()
    receipt_sha256=(Get-FileHash -LiteralPath (Join-Path $evidenceRoot 'remote_receipts/receipt.json') -Algorithm SHA256).Hash.ToLowerInvariant()
    evidence_copy=@{files=21;inventory='generated_inventory.stdout.txt';hash_manifest='copied_evidence.json';excluded_binary_files=@('hold/simulated_states.npz','loop/session/online_motion_20261021_160723/motion_buffers.npz')}
    limits=@('Four optimizer updates per rank do not complete the configured 1,000,000-update schedule or 400,000-step warmup.','FP32 parity uses one real-data batch and 37 deterministic gate/output parameter perturbations; it does not establish BF16 or whole-training equivalence.','The 101-update pilot model falls: closed-loop minimum root height 0.1476634592m; upright proxy is false.','Closed-loop planner calls take roughly 0.36-0.56s with frame-overrun warnings; no real-time or motion-quality claim.','Static uprightness is a root-height proxy; the hold receipt reports 400 controls/8 simulated seconds but no separate integration counter.','Only generated JSON/text evidence was copied; checkpoint and pose binaries remain remote.','No processes were signalled, stopped or restarted; continuity of unrelated jobs was not independently audited with PID snapshots.','The overlay package versions were not re-audited during this run; the frozen environment specification was used without setup changes.')
    evidence_notes=@('The controller UTC interval and remote receipt UTC values use different clocks. Preflight remote 16:06:33Z occurred inside controller interval 16:06:30.5126009Z..16:06:31.4817076Z; preserve both without treating their cross-clock order as runtime order.','capture_summary.json contains two serialized void-return placeholders in its run array; documented_run.invocation.json is the authoritative exact invocation/timestamp/exit record. No runtime retry was made.')
    completed_utc=[DateTime]::UtcNow.ToString('o')
}
[IO.File]::WriteAllText((Join-Path $evidenceRoot 'witness.json'), ($result | ConvertTo-Json -Depth 60), $utf8)
$markdown = @"
# Independent document runtime witness

**$($result.status)**. The frozen command from `docs/full_runtime.md` was executed verbatim exactly once after an immediate output-absence check. It exited **0** with `FULL_RUNTIME_WITNESS_PASSED`. No remote setup, repairs, retries, package/source/config changes, or unrelated process signals were performed.

Controller UTC: **$($invocation.start_utc)** to **$($invocation.end_utc)**. Exact invocation, byte-preserved stdout/stderr and exit status: `documented_run.invocation.json`, `documented_run.stdout.txt`, `documented_run.stderr.txt`. Remote clocks are about two seconds ahead of the controller in the recorded preflight; receipt timestamps are retained separately.

| Document requirement | Independent comparison with generated receipts | Result |
| --- | --- | --- |
| Both DDP ranks complete four actual updates; global batch1024 | Both `PROBE_COMPLETED`, exit0; optimizer updates4; official step0→4; local batch512 × world2. Both configurations equal the frozen JSON. | Pass |
| All three released dataset components | Per rank: AMASS1403, 100style6268, Kungfu687 entries. | Pass |
| FP32 recomputation gradient parity | 365 compared/nonzero gradient tensors; max absolute error0; atol1e-6, rtol1e-5; 37 deterministic gate/output parameter perturbations. | Pass |
| Finite/upright static tracker for8s | 400 controls, 7.999999999999852 simulated seconds, finite=true, upright height proxy=true, minimum root height0.788038m; reference forcing=false. | Pass |
|100 closed-loop controls/400 MuJoCo integrations; >=2 planner calls | Policy calls100; actual integration counter400;2 simulated seconds;12 planner calls.11 replanning events at policy steps9..99 agree with12 calls including the initial plan.12 unique state-prefix hashes; real feedback; no reference forcing. | Pass |
| Sources/configuration/documentation unchanged | Zero before/after changes: local13 supplementary files,226 vendor files,3 configs,2 docs and ledger; remote10 supplementary files,226 vendor files and3 configs. All239 deployed file hashes match local bytes. | Pass |

Three local helpers are intentionally absent from the remote deployment: `scripts/audit_official_data.py`, `scripts/build_overlay.py`, `scripts/verify_github_backup.py`. There are no cross-host hash mismatches. Bytecode/cache files were excluded from source manifests. Per-set digests and all comparisons are in `witness.json`; raw before/after manifests are retained.

Copied **21 generated JSON/text files**, including both rank receipts, metrics, per-phase command/stdout/stderr, parity/hold/loop receipts and closed-loop event metadata. SHA256 manifest: `copied_evidence.json`. No checkpoint or pose binary was copied; the two generated NPZ files are only listed in the remote inventory. Main receipt SHA256: **$($result.receipt_sha256)**.

This establishes execution only. The configured1,000,000-update schedule and400,000-step warmup are not completed. FP32 one-batch parity does not establish BF16/full-training equivalence. The old101-update pilot falls (minimum root height0.147663m); frame-overrun warnings and0.36–0.56s planner calls preclude a real-time claim. Static uprightness is a height proxy, and its receipt has no separate integration counter. No paper metric or trained-planner quality is reproduced. Existing processes were not signalled; unrelated-job PID continuity and package versions were not separately re-audited.

Evidence note: the auxiliary `capture_summary.json` run array includes two serialized void-return placeholders; the separate `documented_run.invocation.json` is authoritative. This affects local evidence formatting only; there was no runtime retry.
"@
[IO.File]::WriteAllText((Join-Path $evidenceRoot 'witness.md'), $markdown, $utf8)
Write-Host ($result.status + '; checks=' + $checks.Count + '; failed=' + @($checks | Where-Object {-not $_.passed}).Count)
if (-not $allPassed) { exit 1 }

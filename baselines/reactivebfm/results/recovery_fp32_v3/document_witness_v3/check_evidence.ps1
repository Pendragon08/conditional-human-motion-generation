$ErrorActionPreference = 'Stop'
$root = 'D:\具身智能\ReactiveBFM复现_20261021\repair_20261022\document_witness_v3'
$raw = Join-Path $root 'raw_outputs'
$checks = [System.Collections.Generic.List[object]]::new()
function Add-Check($name, $actual, $expected, [bool]$passed) {
  $checks.Add([pscustomobject]@{ name=$name; passed=$passed; actual=$actual; expected=$expected })
}
function Add-Equal($name, $actual, $expected) {
  Add-Check $name $actual $expected (($null -ne $actual) -and ($actual -eq $expected))
}
function Add-Finite($name, $actual) {
  $ok = ($null -ne $actual) -and -not [double]::IsNaN([double]$actual) -and -not [double]::IsInfinity([double]$actual)
  Add-Check $name $actual 'finite numeric value' $ok
}
$invocation = Get-Content -LiteralPath (Join-Path $root 'invocation_result.json') -Raw | ConvertFrom-Json
$doc = Get-Content -LiteralPath (Join-Path $root 'raw_inputs\recovery_fp32_v3.md') -Raw
$documented = [regex]::Match($doc, '(?s)```powershell\r?\n(.*?)\r?\n```').Groups[1].Value
$savedCommand = [System.IO.File]::ReadAllText((Join-Path $root 'command.ps1')).TrimEnd("`r", "`n")
Add-Equal 'saved command matches documented invocation verbatim' $savedCommand $documented
Add-Equal 'documented invocation executed exactly once' $invocation.invocation_count 1
Add-Equal 'SSH invocation exit' $invocation.final_exit_code 0
[System.IO.File]::WriteAllText((Join-Path $root 'local_ssh_combined_output.txt'), ($invocation.r.output + (($invocation.polls | ForEach-Object { $_.output }) -join '')), [System.Text.UTF8Encoding]::new($false))
$pins = Get-Content -LiteralPath (Join-Path $raw 'source_pins_verified.json') -Raw | ConvertFrom-Json
$supervisor = Get-Content -LiteralPath (Join-Path $raw 'supervisor_exit.json') -Raw | ConvertFrom-Json
$stdout = Get-Content -LiteralPath (Join-Path $raw 'supervisor_stdout.txt') -Raw
Add-Equal 'supervisor exit' $supervisor.exit_code 0
Add-Check 'stdout model loading' (@(Select-String -InputObject $stdout -Pattern 'loading model from checkpoint:' -AllMatches).Matches.Value) 'loading model from checkpoint:' ($stdout -match 'loading model from checkpoint:')
Add-Check 'stdout model and model_avg loading' (@(Select-String -InputObject $stdout -Pattern 'loading both model and model_avg' -AllMatches).Matches.Value) 'loading both model and model_avg' ($stdout -match 'loading both model and model_avg')
Add-Check 'stdout optimizer loading' (@(Select-String -InputObject $stdout -Pattern 'loading optimizer state from checkpoint:' -AllMatches).Matches.Value) 'loading optimizer state from checkpoint:' ($stdout -match 'loading optimizer state from checkpoint:')
Add-Equal 'checkpoint CPU inspection' $pins.checkpoint_inspection.status 'CHECKPOINT_CPU_INSPECTION_PASSED'
Add-Equal 'inspected initial optimizer updates' $pins.checkpoint_inspection.optimizer_updates 10021
Add-Equal 'inspected model tensor count' $pins.checkpoint_inspection.model.model.tensor_count 388
Add-Equal 'inspected EMA tensor count' $pins.checkpoint_inspection.model.model_avg.tensor_count 388
Add-Equal 'inspected model finite' $pins.checkpoint_inspection.model.model.all_finite $true
Add-Equal 'inspected EMA finite' $pins.checkpoint_inspection.model.model_avg.all_finite $true
Add-Equal 'inspected optimizer state count' $pins.checkpoint_inspection.optimizer_state_count 365
Add-Equal 'inspected optimizer finite' $pins.checkpoint_inspection.all_optimizer_tensors_finite $true
Add-Equal 'inspected optimizer step value count' @($pins.checkpoint_inspection.optimizer_step_values).Count 1
Add-Equal 'inspected optimizer step value' $pins.checkpoint_inspection.optimizer_step_values[0] 10021
Add-Equal 'source entrypoint pin count' @($pins.sources.PSObject.Properties).Count 6
Add-Check 'upstream Python tree is nonempty' $pins.upstream_python_file_count 'positive file count' ($pins.upstream_python_file_count -gt 0)
$rankSummary = @()
foreach ($rank in 0,1) {
  $status = Get-Content -LiteralPath (Join-Path $raw "status_rank$rank.json") -Raw | ConvertFrom-Json
  $health = Get-Content -LiteralPath (Join-Path $raw "health_rank$rank.json") -Raw | ConvertFrom-Json
  $trace = @(Get-Content -LiteralPath (Join-Path $raw "health_rank$rank.jsonl") | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
  foreach ($pair in @(@('rank',$rank), @('status','PROBE_COMPLETED'), @('optimizer_updates',10029), @('official_step_at_start',10021), @('probe_updates',8), @('exit_code',0), @('full_training_completed',$false), @('physics_evaluated',$false))) {
    Add-Equal "rank$rank status $($pair[0])" $status.($pair[0]) $pair[1]
  }
  foreach ($pair in @(@('rank',$rank), @('status','PROBE_COMPLETED'), @('optimizer_updates',10029), @('initial_optimizer_updates',10021), @('resumed_updates',8), @('attempts_since_resume',8), @('skipped_attempts',0), @('optimizer_step_calls',10029), @('validated_optimizer_updates',10029), @('ema_validated',$true), @('step_validated',$true), @('process_status','PROBE_COMPLETED'), @('supervisor_exit_code',0), @('process_running',$false), @('full_training_completed',$false))) {
    Add-Equal "rank$rank health $($pair[0])" $health.($pair[0]) $pair[1]
  }
  foreach ($field in 'grad_norm_pre_clip_fp64','grad_norm_post_clip_fp64','param_norm_fp64') { Add-Finite "rank$rank final $field" $health.$field }
  Add-Equal "rank$rank per-attempt trace row count" $trace.Count 8
  foreach ($i in 0..7) {
    $row=$trace[$i]
    foreach ($pair in @(@('rank',$rank), @('initial_optimizer_updates',10021), @('optimizer_updates',(10022+$i)), @('optimizer_step_calls',(10022+$i)), @('validated_optimizer_updates',(10022+$i)), @('attempts_since_resume',(1+$i)), @('resumed_updates',(1+$i)), @('skipped_attempts',0), @('step_applied',$true), @('step_validated',$true), @('ema_validated',$true))) {
      Add-Equal "rank$rank attempt$($i+1) $($pair[0])" $row.($pair[0]) $pair[1]
    }
    foreach ($field in 'grad_norm_pre_clip_fp64','grad_norm_post_clip_fp64','param_norm_fp64') { Add-Finite "rank$rank attempt$($i+1) $field" $row.$field }
  }
  foreach ($pin in $status.configuration.recovery.source_sha256.PSObject.Properties) {
    Add-Equal "rank$rank source pin $($pin.Name)" $pins.sources.($pin.Name) $pin.Value
  }
  Add-Equal "rank$rank upstream tree pin" $pins.upstream_python_tree_sha256 $status.configuration.recovery.upstream_python_tree_sha256
  Add-Equal "rank$rank train entrypoint pin" $status.source_sha256 $pins.sources.'scripts/train_full.py'
  Add-Equal "rank$rank health guard pin" $health.guard_sha256 $pins.sources.'scripts/numerical_guard.py'
  Add-Equal "rank$rank shared GPU fraction configuration" $status.configuration.memory_fraction_per_gpu 0.65
  Add-Equal "rank$rank CPU threads configuration" $status.configuration.cpu_threads_per_rank 4
  Add-Equal "rank$rank FP32 configuration" $status.configuration.use_bf16 $false
  $rankSummary += [pscustomobject]@{ rank=$rank; status=$status.status; initial=$health.initial_optimizer_updates; updates=$health.optimizer_updates; resumed=$health.resumed_updates; attempts=$health.attempts_since_resume; skipped=$health.skipped_attempts; optimizer_step_calls=$health.optimizer_step_calls; validated_optimizer_updates=$health.validated_optimizer_updates; ema_validated=$health.ema_validated; trace_rows=$trace.Count; process_running=$health.process_running; pre_clip_norm=$health.grad_norm_pre_clip_fp64; post_clip_norm=$health.grad_norm_post_clip_fp64; parameter_norm=$health.param_norm_fp64; elapsed_seconds=$status.elapsed_seconds }
}
$failures = @($checks | Where-Object { -not $_.passed })
$summary = [pscustomobject]@{ result=$(if($failures.Count -eq 0){'PASS'}else{'FAIL'}); checked_utc=[DateTime]::UtcNow.ToString('o'); invocation_count=$invocation.invocation_count; ssh_exit_code=$invocation.final_exit_code; supervisor_exit_code=$supervisor.exit_code; total_checks=$checks.Count; passed_checks=$checks.Count-$failures.Count; failed_checks=$failures.Count; ranks=$rankSummary; discrepancies=$failures; checks=$checks }
$summary | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath (Join-Path $root 'checks.json') -Encoding UTF8
$inventory = @(Get-ChildItem -LiteralPath $raw -File | ForEach-Object { [pscustomobject]@{ name=$_.Name; bytes=$_.Length; sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() } })
$inventory | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $root 'raw_output_inventory.json') -Encoding UTF8
$summary | Select-Object result,total_checks,passed_checks,failed_checks,ranks,discrepancies | ConvertTo-Json -Depth 8
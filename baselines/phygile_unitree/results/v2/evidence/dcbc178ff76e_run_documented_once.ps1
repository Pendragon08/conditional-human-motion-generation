$ErrorActionPreference = 'Continue'
$taskWitness = 'D:\具身智能\PhyGile复现\Unitree框架适配_20261022\closure_mjlab_doc_witness_v1'
$taskSource = 'D:\具身智能\PhyGile复现\Unitree框架适配_20261022\source_mj_closure_v1'
$taskUtf8 = New-Object System.Text.UTF8Encoding($true)
function Save-TaskJson($taskName, $taskValue) {
    [IO.File]::WriteAllText((Join-Path $taskWitness $taskName), ($taskValue | ConvertTo-Json -Depth 12), $taskUtf8)
}
function Get-TaskSourceHashes {
    $taskManifest = Get-Content -LiteralPath (Join-Path $taskSource 'source_manifest.json') -Raw | ConvertFrom-Json
    $taskRows = @()
    foreach ($taskEntry in $taskManifest.PSObject.Properties) {
        $taskFile = Join-Path $taskSource $taskEntry.Name
        $taskActual = (Get-FileHash -LiteralPath $taskFile -Algorithm SHA256).Hash.ToLowerInvariant()
        $taskRows += [pscustomobject]@{path=$taskEntry.Name; expected_sha256=$taskEntry.Value; actual_sha256=$taskActual; matches=($taskActual -eq $taskEntry.Value)}
    }
    return $taskRows
}
$taskBefore = @(Get-TaskSourceHashes)
Save-TaskJson 'source_hashes_before.json' $taskBefore
Copy-Item -LiteralPath (Join-Path $taskSource 'source_manifest.json') -Destination (Join-Path $taskWitness 'source_manifest.json')
if (@($taskBefore | Where-Object { -not $_.matches }).Count -gt 0) { throw 'Immutable source manifest mismatch before invocation' }
$taskStart = [DateTime]::UtcNow
Save-TaskJson 'launch.json' ([pscustomobject]@{started_utc=$taskStart.ToString('o'); wrapper_pid=$PID; powershell_version=$PSVersionTable.PSVersion.ToString(); invocation_count=1; limit_seconds=1200; no_repair_no_retry=$true})
$global:LASTEXITCODE = $null
& {
& 'D:/phygile_unitree_env_20261022/Scripts/python.exe' 'D:/具身智能/PhyGile复现/Unitree框架适配_20261022/source_mj_closure_v1/scripts/run_mjlab_closed_loop.py' --parent 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/phygile' --unitree-checkout 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_unitree_20261022/vendor_unitree_rl_mjlab' --assets 'D:/具身智能/PhyGile复现/Unitree框架适配_20261022/private_assets_contact_v3' --output 'D:/具身智能/PhyGile复现/Unitree框架适配_20261022/closure_mjlab_v1'
} 1> (Join-Path $taskWitness 'stdout.txt') 2> (Join-Path $taskWitness 'stderr.txt')
$taskActualExit = $LASTEXITCODE
$taskEnd = [DateTime]::UtcNow
[IO.File]::WriteAllText((Join-Path $taskWitness 'actual_exit_code.txt'), [string]$taskActualExit, $taskUtf8)
Save-TaskJson 'process_result.json' ([pscustomobject]@{started_utc=$taskStart.ToString('o'); ended_utc=$taskEnd.ToString('o'); duration_seconds=($taskEnd-$taskStart).TotalSeconds; actual_exit_code=$taskActualExit; invocation_count=1})
$taskAfter = @(Get-TaskSourceHashes)
Save-TaskJson 'source_hashes_after.json' $taskAfter
Save-TaskJson 'source_integrity.json' ([pscustomobject]@{file_count=$taskBefore.Count; before_all_match=(@($taskBefore | Where-Object { -not $_.matches }).Count -eq 0); after_all_match=(@($taskAfter | Where-Object { -not $_.matches }).Count -eq 0); manifest_before_sha256=(Get-FileHash -LiteralPath (Join-Path $taskWitness 'source_manifest.json') -Algorithm SHA256).Hash.ToLowerInvariant(); manifest_after_sha256=(Get-FileHash -LiteralPath (Join-Path $taskSource 'source_manifest.json') -Algorithm SHA256).Hash.ToLowerInvariant()})
$taskVerification = 'D:\具身智能\PhyGile复现\Unitree框架适配_20261022\closure_mjlab_v1\verification.json'
if (Test-Path -LiteralPath $taskVerification) { Copy-Item -LiteralPath $taskVerification -Destination (Join-Path $taskWitness 'verification.json') }
exit $taskActualExit
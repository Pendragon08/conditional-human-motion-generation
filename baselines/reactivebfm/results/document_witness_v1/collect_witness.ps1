$ErrorActionPreference = 'Stop'
$witnessRoot = $PSScriptRoot
if (Test-Path -LiteralPath (Join-Path $witnessRoot 'documented_run.process.json')) {
    throw 'This directory already contains a completed documented invocation. Do not rerun or overwrite the witness.'
}
$repository = 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/reactivebfm_reproduction_20261021/repository'
$remoteRoot = '/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021'
$receiptRemote = "$remoteRoot/witness_document_v1.json"
$target = 'ouyanghao@10.147.17.32'
$key = 'C:/Users/Pende/.ssh/id_ed25519'
$utf8 = [System.Text.UTF8Encoding]::new($false)
$docs = @(
    'baselines/reactivebfm/docs/environment.md',
    '.aris/compute/reactivebfm-20261021.md',
    'baselines/reactivebfm/configs/env_spec.json'
)

function Save-Json($path, $value) {
    [System.IO.File]::WriteAllText($path, ($value | ConvertTo-Json -Depth 30), $utf8)
}

function Invoke-LoggedProcess($name, $exe, $arguments) {
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $exe
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    foreach ($argument in $arguments) { $psi.ArgumentList.Add($argument) }
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $psi
    $startUtc = [DateTime]::UtcNow.ToString('o')
    $null = $process.Start()
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()
    $stdout = $stdoutTask.GetAwaiter().GetResult()
    $stderr = $stderrTask.GetAwaiter().GetResult()
    $endUtc = [DateTime]::UtcNow.ToString('o')
    [System.IO.File]::WriteAllText((Join-Path $witnessRoot "$name.stdout.txt"), $stdout, $utf8)
    [System.IO.File]::WriteAllText((Join-Path $witnessRoot "$name.stderr.txt"), $stderr, $utf8)
    $record = [ordered]@{
        executable = $exe
        arguments = @($arguments)
        utc_start = $startUtc
        utc_end = $endUtc
        exit_code = $process.ExitCode
        stdout_file = "$name.stdout.txt"
        stderr_file = "$name.stderr.txt"
        output_encoding = 'UTF-8 without BOM; decoded process stdout/stderr; newline content preserved'
    }
    Save-Json (Join-Path $witnessRoot "$name.process.json") $record
    return [PSCustomObject]@{ Record = $record; Stdout = $stdout; Stderr = $stderr; ExitCode = $process.ExitCode }
}

function Invoke-LoggedSsh($name, $command) {
    return Invoke-LoggedProcess $name 'C:/Windows/System32/OpenSSH/ssh.exe' @('-i', $key, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', $target, $command)
}

function Save-DocumentSnapshot($suffix) {
    $records = @()
    foreach ($relative in $docs) {
        $source = Join-Path $repository $relative
        $label = $relative.Replace('/', '__')
        $copy = "$label.$suffix.snapshot"
        [System.IO.File]::Copy($source, (Join-Path $witnessRoot $copy), $true)
        $records += [ordered]@{ path = $relative; sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash.ToLowerInvariant(); snapshot = $copy }
    }
    Save-Json (Join-Path $witnessRoot "documents.$suffix.json") $records
    return $records
}

$beforeDocuments = Save-DocumentSnapshot 'before'
$environmentText = [System.IO.File]::ReadAllText((Join-Path $repository $docs[0]))
$fence = [regex]::Match($environmentText, '(?s)```sh\r?\n([^\r\n]+)\r?\n```')
if (-not $fence.Success) { throw 'The documented single-line sh command was not found.' }
$documentedCommand = $fence.Groups[1].Value
[System.IO.File]::WriteAllText((Join-Path $witnessRoot 'exact_command.txt'), $documentedCommand, $utf8)

$preflightCommand = "if test -e '$receiptRemote'; then printf '%s\n' 'OUTPUT_ALREADY_EXISTS'; exit 73; fi; printf '%s\n' 'OUTPUT_ABSENT'; pgrep -af '$remoteRoot.*([w]itness_runtime.py|[t]rain)' || test `$? -eq 1"
$preflight = Invoke-LoggedSsh 'preflight' $preflightCommand
if ($preflight.ExitCode -ne 0 -or $preflight.Stdout.Trim() -ne 'OUTPUT_ABSENT') { throw 'Preflight did not prove absent output and no project witness/training process.' }

$sourceCommand = "find '$remoteRoot/code' -type f ! -path '*/.git/*' ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum"
$sourceBefore = Invoke-LoggedSsh 'sources.before' $sourceCommand
if ($sourceBefore.ExitCode -ne 0 -or -not $sourceBefore.Stdout.Contains('/code/scripts/witness_runtime.py')) { throw 'Source before-hash inventory failed.' }

$run = Invoke-LoggedSsh 'documented_run' $documentedCommand

$sourceAfter = Invoke-LoggedSsh 'sources.after' $sourceCommand
$receiptProofCommand = "if test -f '$receiptRemote'; then sha256sum '$receiptRemote'; else printf '%s\n' 'RECEIPT_MISSING'; exit 74; fi"
$receiptProof = Invoke-LoggedSsh 'receipt.remote_sha256' $receiptProofCommand
$receiptCopy = $null
if ($receiptProof.ExitCode -eq 0) {
    $receiptCopy = Invoke-LoggedProcess 'receipt.copy' 'C:/Windows/System32/OpenSSH/scp.exe' @('-i', $key, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', "${target}:$receiptRemote", (Join-Path $witnessRoot 'witness_document_v1.json'))
}
$afterDocuments = Save-DocumentSnapshot 'after'
$localReceiptHash = $null
$receipt = $null
if (Test-Path -LiteralPath (Join-Path $witnessRoot 'witness_document_v1.json')) {
    $localReceiptHash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $witnessRoot 'witness_document_v1.json')).Hash.ToLowerInvariant()
    $receipt = Get-Content -LiteralPath (Join-Path $witnessRoot 'witness_document_v1.json') -Raw | ConvertFrom-Json
}
$remoteReceiptHash = if ($receiptProof.ExitCode -eq 0) { ($receiptProof.Stdout.Trim() -split '\s+')[0].ToLowerInvariant() } else { $null }
$sourceBeforeHash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $witnessRoot 'sources.before.stdout.txt')).Hash.ToLowerInvariant()
$sourceAfterHash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $witnessRoot 'sources.after.stdout.txt')).Hash.ToLowerInvariant()
$documentsUnchanged = (@(Compare-Object ($beforeDocuments | ForEach-Object { "$($_.path) $($_.sha256)" }) ($afterDocuments | ForEach-Object { "$($_.path) $($_.sha256)" })).Count -eq 0)
$summary = [ordered]@{
    witness_kind = 'independent agent-follows-doc single documented invocation'
    setup_inputs = $docs
    exact_command = $documentedCommand
    invocation_count = 1
    utc_start = $run.Record.utc_start
    utc_end = $run.Record.utc_end
    exit_code = $run.ExitCode
    output_absence_confirmed = ($preflight.ExitCode -eq 0 -and $preflight.Stdout.Trim() -eq 'OUTPUT_ABSENT')
    preflight_trace = 'preflight.process.json'
    run_trace = 'documented_run.process.json'
    receipt_file = if ($receipt) { 'witness_document_v1.json' } else { $null }
    remote_receipt_sha256 = $remoteReceiptHash
    local_receipt_sha256 = $localReceiptHash
    receipt_hashes_match = ($null -ne $localReceiptHash -and $localReceiptHash -eq $remoteReceiptHash)
    source_inventory_scope = 'All regular files under remote code/, excluding .git, __pycache__, and *.pyc; sha256sum sorted by path'
    source_inventory_before_sha256 = $sourceBeforeHash
    source_inventory_after_sha256 = $sourceAfterHash
    source_inventory_file_count_before = @($sourceBefore.Stdout.TrimEnd() -split '\r?\n').Count
    source_inventory_file_count_after = @($sourceAfter.Stdout.TrimEnd() -split '\r?\n').Count
    source_inventory_before_exit_code = $sourceBefore.ExitCode
    source_inventory_after_exit_code = $sourceAfter.ExitCode
    source_files_unchanged = ($sourceBefore.ExitCode -eq 0 -and $sourceAfter.ExitCode -eq 0 -and $sourceBeforeHash -eq $sourceAfterHash)
    document_before = $beforeDocuments
    document_after = $afterDocuments
    documents_unchanged = $documentsUnchanged
    final_receipt_status = if ($receipt) { $receipt.status } else { $null }
    claims_scope = 'Selected runtime/kernel/official model forward and backward only. No optimizer training, physics, paper performance, target reaching, or hardware validation.'
    deviations = @()
}
Save-Json (Join-Path $witnessRoot 'witness.json') $summary
$summary | ConvertTo-Json -Depth 30

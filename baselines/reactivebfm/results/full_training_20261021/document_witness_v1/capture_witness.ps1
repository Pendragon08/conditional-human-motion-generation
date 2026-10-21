$ErrorActionPreference = 'Stop'
$repoRoot = 'C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/reactivebfm_reproduction_20261021/repository'
$baselineRoot = Join-Path $repoRoot 'baselines/reactivebfm'
$evidenceRoot = Join-Path $baselineRoot 'results/full_training_20261021/document_witness_v1'
$remoteTarget = 'ouyanghao@10.147.17.32'
$remoteCode = '/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/code'
$remoteOutput = '/home/ouyanghao/reactivebfm-full-20261021/full_witness_document_v1'
$documentedCommand = '/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/code/scripts/full_runtime_witness.py --runtime-root /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021 --output /home/ouyanghao/reactivebfm-full-20261021/full_witness_document_v1'
$utf8 = [System.Text.UTF8Encoding]::new($false)

function Write-JsonEvidence($name, $value) {
    [IO.File]::WriteAllText((Join-Path $evidenceRoot $name), ($value | ConvertTo-Json -Depth 50), $utf8)
}

function Invoke-SshCaptured($label, $remoteCommand) {
    $outPath = Join-Path $evidenceRoot ($label + '.stdout.txt')
    $errPath = Join-Path $evidenceRoot ($label + '.stderr.txt')
    $startUtc = [DateTime]::UtcNow.ToString('o')
    $psi = [Diagnostics.ProcessStartInfo]::new('C:/Windows/System32/OpenSSH/ssh.exe')
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    foreach ($arg in @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', $remoteTarget, $remoteCommand)) { $psi.ArgumentList.Add($arg) }
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $psi
    $outStream = [IO.File]::Create($outPath)
    $errStream = [IO.File]::Create($errPath)
    try {
        if (-not $process.Start()) { throw 'SSH failed to start.' }
        $outCopy = $process.StandardOutput.BaseStream.CopyToAsync($outStream)
        $errCopy = $process.StandardError.BaseStream.CopyToAsync($errStream)
        $process.WaitForExit()
        $outCopy.GetAwaiter().GetResult()
        $errCopy.GetAwaiter().GetResult()
        $exitCode = $process.ExitCode
    } finally {
        $outStream.Dispose()
        $errStream.Dispose()
    }
    $endUtc = [DateTime]::UtcNow.ToString('o')
    $record = [ordered]@{label=$label;remote_command=$remoteCommand;ssh_options=@('-o','BatchMode=yes','-o','ConnectTimeout=15');start_utc=$startUtc;end_utc=$endUtc;exit_code=$exitCode;stdout_file=($label+'.stdout.txt');stderr_file=($label+'.stderr.txt')}
    Write-JsonEvidence ($label + '.invocation.json') $record
    if ($exitCode -ne 0 -and $label -ne 'documented_run') { throw "$label failed with exit $exitCode. No runtime retry is permitted." }
    return $record
}

function Get-LocalManifest {
    $sets = @(
        @{ name='supplementary_source'; path=(Join-Path $baselineRoot 'scripts') },
        @{ name='official_vendor_source'; path=(Join-Path $baselineRoot 'upstream') },
        @{ name='configuration'; path=(Join-Path $baselineRoot 'configs') },
        @{ name='documentation'; path=(Join-Path $baselineRoot 'docs') }
    )
    $entries = [Collections.Generic.List[object]]::new()
    foreach ($set in $sets) {
        foreach ($file in (Get-ChildItem -LiteralPath $set.path -File -Recurse | Sort-Object FullName)) {
            if ($file.FullName -match '[\\/](__pycache__|\.git)[\\/]' -or $file.Extension -in @('.pyc','.pyo')) { continue }
            $entries.Add([ordered]@{set=$set.name;path=([IO.Path]::GetRelativePath($repoRoot,$file.FullName).Replace('\','/'));bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()})
        }
    }
    $ledger = Join-Path $repoRoot '.aris/compute/reactivebfm-20261021.md'
    $entries.Add([ordered]@{set='compute_ledger';path='.aris/compute/reactivebfm-20261021.md';bytes=(Get-Item -LiteralPath $ledger).Length;sha256=(Get-FileHash -LiteralPath $ledger -Algorithm SHA256).Hash.ToLowerInvariant()})
    return @($entries.ToArray())
}

Write-JsonEvidence 'local_before.json' (Get-LocalManifest)
$sourceManifestCommand = "cd $remoteCode && find configs scripts upstream -type f ! -path '*/__pycache__/*' ! -path '*/.git/*' ! -name '*.pyc' ! -name '*.pyo' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum"
$before = Invoke-SshCaptured 'remote_sources_before' $sourceManifestCommand
$preflight = Invoke-SshCaptured 'output_absent' "test ! -e $remoteOutput && date -u +%Y-%m-%dT%H:%M:%SZ && printf 'OUTPUT_ABSENT\n'"
Write-Host 'Preflight passed; launching the exact documented command once.'
$run = Invoke-SshCaptured 'documented_run' $documentedCommand
Write-Host ("Documented command exited " + $run.exit_code + '; capturing unchanged-source checks and evidence.')
$after = Invoke-SshCaptured 'remote_sources_after' $sourceManifestCommand
Write-JsonEvidence 'local_after.json' (Get-LocalManifest)
$inventory = Invoke-SshCaptured 'generated_inventory' "find $remoteOutput -type f -printf '%P\t%s\n' | LC_ALL=C sort"
$textFiles = Invoke-SshCaptured 'generated_text_files' "find $remoteOutput -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.txt' -o -name '*.log' -o -name '*.md' -o -name '*.csv' \) -printf '%P\n' | LC_ALL=C sort"
$receiptsRoot = Join-Path $evidenceRoot 'remote_receipts'
[IO.Directory]::CreateDirectory($receiptsRoot) | Out-Null
$copied = [Collections.Generic.List[object]]::new()
foreach ($relative in [IO.File]::ReadAllLines((Join-Path $evidenceRoot 'generated_text_files.stdout.txt'))) {
    if ([string]::IsNullOrWhiteSpace($relative)) { continue }
    if ($relative -match '(^/|(^|/)\.\.(/|$)|[\r\n])' -or $relative -notmatch '^[A-Za-z0-9_./-]+$') { throw 'Unsafe evidence path.' }
    $localDestination = Join-Path $receiptsRoot $relative
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($localDestination)) | Out-Null
    $label = 'copy_' + ([Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($utf8.GetBytes($relative))).Substring(0,16).ToLowerInvariant())
    $copy = Invoke-SshCaptured $label ("cat '" + $remoteOutput + '/' + $relative + "'")
    [IO.File]::Move((Join-Path $evidenceRoot ($label+'.stdout.txt')), $localDestination)
    $copied.Add([ordered]@{path=$relative;bytes=(Get-Item -LiteralPath $localDestination).Length;sha256=(Get-FileHash -LiteralPath $localDestination -Algorithm SHA256).Hash.ToLowerInvariant();capture_invocation=($label+'.invocation.json')})
}
Write-JsonEvidence 'copied_evidence.json' $copied.ToArray()
Write-JsonEvidence 'capture_summary.json' ([ordered]@{invocation_count=1;run=$run;remote_sources_before_sha256=(Get-FileHash -LiteralPath (Join-Path $evidenceRoot 'remote_sources_before.stdout.txt') -Algorithm SHA256).Hash.ToLowerInvariant();remote_sources_after_sha256=(Get-FileHash -LiteralPath (Join-Path $evidenceRoot 'remote_sources_after.stdout.txt') -Algorithm SHA256).Hash.ToLowerInvariant();local_before_sha256=(Get-FileHash -LiteralPath (Join-Path $evidenceRoot 'local_before.json') -Algorithm SHA256).Hash.ToLowerInvariant();local_after_sha256=(Get-FileHash -LiteralPath (Join-Path $evidenceRoot 'local_after.json') -Algorithm SHA256).Hash.ToLowerInvariant();copied_files=$copied.Count;complete_utc=[DateTime]::UtcNow.ToString('o')})
Write-Host ('Capture complete; copied ' + $copied.Count + ' JSON/text files. No checkpoints or pose bytes copied.')

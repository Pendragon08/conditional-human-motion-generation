$ErrorActionPreference = 'Stop'
$original = Get-Process -Id 43528 -ErrorAction SilentlyContinue
if ($original -and $original.ProcessName -eq 'python') {
    Wait-Process -Id 43528 -ErrorAction SilentlyContinue
}
$python = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$script = Join-Path $PSScriptRoot 'upload_grab_checked.py'
$report = Join-Path (Split-Path $PSScriptRoot) 'reports\grab_s10_and_mesh_crc_20261030.json'
& $python $script $report
exit $LASTEXITCODE

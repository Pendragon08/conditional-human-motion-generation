$taskRepo = Split-Path -Parent $PSScriptRoot
$taskMon = Join-Path $taskRepo 'artifacts\monitoring'
[IO.Directory]::CreateDirectory($taskMon) | Out-Null
$taskMutex = [Threading.Mutex]::new($false, 'Local\R3CloSDSceneTunnel')
if (-not $taskMutex.WaitOne(0)) { exit }
try {
    while ($true) {
        & 'C:\Windows\System32\OpenSSH\ssh.exe' -NT `
            -b '10.147.17.250' -L '127.0.0.1:7001:127.0.0.1:7000' `
            -i 'C:/Users/Pende/.ssh/id_ed25519' `
            -o BatchMode=yes -o ConnectTimeout=20 `
            -o ServerAliveInterval=10 -o ServerAliveCountMax=3 `
            -o ExitOnForwardFailure=yes 'ouyanghao@10.147.17.32' `
            2>> (Join-Path $taskMon 'r3_closd_scene_tunnel.log')
        Start-Sleep -Seconds 5
    }
} finally {
    $taskMutex.ReleaseMutex()
    $taskMutex.Dispose()
}

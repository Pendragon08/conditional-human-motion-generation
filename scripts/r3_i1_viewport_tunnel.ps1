$taskRepo = Split-Path -Parent $PSScriptRoot
$taskMon = Join-Path $taskRepo 'artifacts\monitoring'
[IO.Directory]::CreateDirectory($taskMon) | Out-Null
$taskMutex = [Threading.Mutex]::new($false, 'Local\R3I1ViewportTunnel')
if (-not $taskMutex.WaitOne(0)) { exit }
try {
    while ($true) {
        & 'C:\Windows\System32\OpenSSH\ssh.exe' -NT `
            -L '127.0.0.1:6013:127.0.0.1:6013' `
            -i 'C:/Users/Pende/.ssh/id_ed25519' `
            -o BatchMode=yes -o ConnectTimeout=20 `
            -o ServerAliveInterval=10 -o ServerAliveCountMax=3 `
            -o ExitOnForwardFailure=yes 'ouyanghao@10.147.17.32' `
            2>> (Join-Path $taskMon 'r3_i1_viewport_tunnel.log')
        Start-Sleep -Seconds 5
    }
} finally {
    $taskMutex.ReleaseMutex()
    $taskMutex.Dispose()
}

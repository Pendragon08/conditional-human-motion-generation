# Keep a private server-local proxy forwarding alive for the licensed CLoSD
# Hugging Face snapshot. The local proxy listens only on this PC, and the
# remote SSH port is bound to the server's loopback interface.
$ErrorActionPreference = 'Stop'
$sshExe = Join-Path $env:WINDIR 'System32\OpenSSH\ssh.exe'
$keyPath = 'C:/Users/Pende/.ssh/id_ed25519'
$logPath = 'D:/具身智能/humanoid-combat-motion/records/monitoring/r3_closd_proxy_keepalive.log'
$logDir = Split-Path -Parent $logPath
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
while ($true) {
    & $sshExe -N -R '127.0.0.1:18181:127.0.0.1:7897' -i $keyPath -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 ouyanghao@10.147.17.32
    Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) ssh_exit=$LASTEXITCODE reconnecting"
    Start-Sleep -Seconds 5
}

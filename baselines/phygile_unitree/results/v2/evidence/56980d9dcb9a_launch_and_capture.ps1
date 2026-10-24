$ErrorActionPreference = 'Stop'
$evidenceRoot = 'D:\具身智能\PhyGile复现\Unitree框架适配_20261022\contact_doc_witness_v3'
$receiptPath = Join-Path $evidenceRoot 'command_receipt.json'
$receipt = Get-Content -Raw -Encoding UTF8 -LiteralPath $receiptPath | ConvertFrom-Json
$process = Start-Process -FilePath $receipt.shell -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $evidenceRoot 'execute_once.ps1')) -WorkingDirectory $receipt.cwd -WindowStyle Hidden -RedirectStandardOutput $receipt.stdout -RedirectStandardError $receipt.stderr -PassThru
$receipt.status = 'RUNNING'
$receipt | Add-Member -NotePropertyName pid -NotePropertyValue $process.Id
$receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $receiptPath -Encoding UTF8
Write-Output ('LAUNCHED_PID=' + $process.Id)
$process.WaitForExit()
$process.Refresh()
$receipt.status = 'EXITED'
$receipt | Add-Member -NotePropertyName exit_code -NotePropertyValue $process.ExitCode
$receipt | Add-Member -NotePropertyName finished_utc -NotePropertyValue ([datetime]::UtcNow.ToString('o'))
$receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $receiptPath -Encoding UTF8
Write-Output ('ACTUAL_EXIT_CODE=' + $process.ExitCode)

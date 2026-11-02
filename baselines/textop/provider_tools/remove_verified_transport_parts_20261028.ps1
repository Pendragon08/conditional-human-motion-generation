param([Parameter(Mandatory=$true)][string]$Proof,[Parameter(Mandatory=$true)][string]$Evidence,[Parameter(Mandatory=$true)][string]$BackedUpCommit)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false)
$utf8=New-Object System.Text.UTF8Encoding($false)
$j=Get-Content -LiteralPath $Proof -Encoding UTF8 -Raw | ConvertFrom-Json
if ($j.status -ne 'D_DUPLICATE_TRANSPORT_PARTS_RECOVERABILITY_VERIFIED' -or $BackedUpCommit -notmatch '^[0-9a-f]{40}$') { throw 'Invalid recovery proof' }
$root=(Resolve-Path -LiteralPath $j.allowed_root).Path
$full=(Resolve-Path -LiteralPath $j.retained_archive.path).Path
if ([IO.Path]::GetPathRoot($root) -ne 'D:\' -or [IO.Path]::GetDirectoryName($full) -ne $root) { throw 'Unexpected absolute target root' }
$ownerAnchor=[IO.Path]::GetDirectoryName([IO.Path]::GetDirectoryName($root))
$sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
function Check-OwnPlain([string]$path) {
 $here=$path
 while ($true) {
  $item=Get-Item -LiteralPath $here -Force
  if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Linked target or ancestor' }
  $owner=(Get-Acl -LiteralPath $here).GetOwner([Security.Principal.SecurityIdentifier]).Value
  if ($owner -ne $sid) { throw 'Foreign owner' }
  if ($here -eq $ownerAnchor) { break }
  $here=[IO.Path]::GetDirectoryName($here)
  if ([string]::IsNullOrEmpty($here)) { throw 'Target escaped intended directory' }
 }
}
function New-Record([string]$name,$value) {
 $p=Join-Path $Evidence $name
 $raw=$utf8.GetBytes(($value | ConvertTo-Json -Depth 12))
 $f=[IO.File]::Open($p,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
 try { $f.Write($raw,0,$raw.Length);$f.Flush($true) } finally { $f.Dispose() }
}
Check-OwnPlain $full
if (@($j.targets).Count -ne 4) { throw 'Only the four fixed complete transport parts may be removed' }
$expected=@('contact-v1.tar.gz.part000','contact-v1.tar.gz.part001','contact-v1.tar.gz.part002','contact-v1.tar.gz.part003')
$retained=[IO.File]::Open($full,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read)
$held=New-Object System.Collections.ArrayList
$deleted=New-Object System.Collections.ArrayList
try {
 $h=[Security.Cryptography.SHA256]::Create()
 try { $actual=([BitConverter]::ToString($h.ComputeHash($retained))).Replace('-','').ToLowerInvariant() } finally { $h.Dispose() }
 if ($actual -ne $j.retained_archive.sha256 -or $retained.Length -ne [int64]$j.retained_archive.bytes) { throw 'Retained full archive differs' }
 for ($i=0;$i -lt 4;$i++) {
  $row=$j.targets[$i];$target=Join-Path $root $expected[$i]
  if ($row.name -ne $expected[$i] -or $row.path -ne $target -or (Resolve-Path -LiteralPath $target).Path -ne $target) { throw 'Unexpected exact target' }
  Check-OwnPlain $target
  $f=[IO.File]::Open($target,[IO.FileMode]::Open,[IO.FileAccess]::Read,([IO.FileShare]::Read -bor [IO.FileShare]::Delete))
  [void]$held.Add($f)
  if ($f.Length -ne [int64]$row.bytes) { throw 'Target size changed' }
  $h=[Security.Cryptography.SHA256]::Create()
  try { $actual=([BitConverter]::ToString($h.ComputeHash($f))).Replace('-','').ToLowerInvariant() } finally { $h.Dispose() }
  if ($actual -ne $row.sha256 -or $row.github_digest -ne ('sha256:'+$actual)) { throw 'Target bytes changed' }
 }
 $before=([IO.DriveInfo]::new('D:\')).AvailableFreeSpace
 New-Record 'D_apply_intent.json' @{status='EXACT_FOUR_REDUNDANT_TRANSPORT_PARTS_APPLY_INTENT';backed_up_commit=$BackedUpCommit;proof_sha256=(Get-FileHash -LiteralPath $Proof -Algorithm SHA256).Hash.ToLowerInvariant();full_archive_retained=$full;target_count=4}
 for ($i=0;$i -lt 4;$i++) {
  $target=Join-Path $root $expected[$i]
  Remove-Item -LiteralPath $target -ErrorAction Stop
  $held[$i].Dispose()
  [void]$deleted.Add($expected[$i])
  New-Record ('D_removed_'+$i+'.json') @{name=$expected[$i];bytes=$j.targets[$i].bytes;retained_full_archive=$full}
 }
 $after=([IO.DriveInfo]::new('D:\')).AvailableFreeSpace
 if (-not (Test-Path -LiteralPath $full -PathType Leaf) -or $after -lt 2147483648) { throw 'Preserved archive or reserve check failed' }
 $result=@{status='VERIFIED_REDUNDANT_TRANSPORT_PARTS_REMOVED';backed_up_commit=$BackedUpCommit;removed_names=@($deleted);duplicate_logical_bytes=[int64]$j.total_duplicate_transport_bytes;D_free_before_bytes=$before;D_free_after_bytes=$after;D_observed_free_delta_bytes=($after-$before);full_archive_retained=$true;original_archive_members_retained=$j.all_original_archive_members_sha256_verified;datasets_removed=0;checkpoints_removed=0;unique_records_removed=0;recursive_delete_used=$false}
 New-Record 'D_cleanup.json' $result
 $result | ConvertTo-Json -Depth 8 -Compress
} catch {
 New-Record 'D_apply_failure.json' @{status='KEEP_RETAINED_ARCHIVE_AND_INSPECT_EXISTING_JOURNAL';error=$_.Exception.Message;deleted_names=@($deleted)}
 throw
} finally {
 foreach ($f in $held) { $f.Dispose() }
 $retained.Dispose()
}

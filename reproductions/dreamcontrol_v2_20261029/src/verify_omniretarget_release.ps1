param(
    [string]$OutputPath = 'reports/omniretarget_release_provenance_20261101.json'
)
$ErrorActionPreference='Stop'
$revision='135259130158f1cb1799e4ab9b6566c938285fc9'
$base="https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset/raw/$revision"
$root='data/OmniRetarget_Dataset'
if (Test-Path -LiteralPath $OutputPath) { throw 'Output already exists; preserve prior report' }
$head = git ls-remote https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset HEAD
if ($LASTEXITCODE -ne 0) { throw 'Unable to inspect official repository HEAD' }
$headHash=($head -split '\s+')[0]
$readme=(Invoke-WebRequest -Uri "$base/README.md" -UseBasicParsing -TimeoutSec 20).Content
if ($readme -notmatch '(?m)^license:\s*mit\s*$' -or $readme -notmatch 'OMOMO' -or $readme -notmatch 'In-house MoCap') {
    throw 'Pinned dataset card does not match recorded metadata'
}
$remote=ssh -o BatchMode=yes -o ConnectTimeout=8 10.147.17.32 'cd /data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/data/OmniRetarget_Dataset && sha256sum robot-object.zip robot-terrain.zip robot-object-terrain.zip'
if ($LASTEXITCODE -ne 0) { throw 'Unable to hash remote archive copies' }
$remoteHash=@{}
foreach($line in $remote) {
    if ($line -notmatch '^([a-f0-9]{64})\s+(.+)$') { throw 'Unexpected remote hash output' }
    $remoteHash[$Matches[2]]=$Matches[1]
}
$records=@()
foreach($name in @('robot-object.zip','robot-terrain.zip','robot-object-terrain.zip')) {
    $pointer=(Invoke-WebRequest -Uri "$base/$name" -UseBasicParsing -TimeoutSec 20).Content
    if ($pointer -notmatch '(?m)^oid sha256:([a-f0-9]{64})\s*$') { throw "Missing pointer hash: $name" }
    $upstream=$Matches[1]
    if ($pointer -notmatch '(?m)^size ([0-9]+)\s*$') { throw "Missing pointer size: $name" }
    $size=[long]$Matches[1]
    $file=Get-Item -LiteralPath (Join-Path $root $name)
    $local=(Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash.ToLowerInvariant()
    if ($file.Length -ne $size -or $local -ne $upstream -or $remoteHash[$name] -ne $upstream) {
        throw "Official, local and server copies differ: $name"
    }
    $records+=@{archive=$name;bytes=$size;official_pointer_sha256=$upstream;local_sha256=$local;
                server_sha256=$remoteHash[$name];pointer_url="$base/$name"}
}
$report=[ordered]@{status='three_official_release_archives_match'; revision=$revision; remote_head_at_check=$headHash;
    dataset_card_url="$base/README.md";dataset_card_metadata_license='mit';
    card_describes_source_by_subset=@{'robot-object'='OMOMO';'robot-terrain'='in-house MoCap';'robot-object-terrain'='in-house MoCap'};
    archives=$records;training_ready=$false;
    limitations=@('Dataset card is metadata, not an independent legal determination of every upstream right or use case.',
      'Archive byte identity does not verify per-motion source identity, text, time, joint order, 29-to-27 fit or physics.',
      'LAFAN1 is explicitly not included in the released dataset; no full-mix or training acceptance is claimed.')}
$json=$report|ConvertTo-Json -Depth 5
$resolved=Join-Path (Get-Location) $OutputPath
$f=[System.IO.File]::Open($resolved,[System.IO.FileMode]::CreateNew)
try { $bytes=[System.Text.UTF8Encoding]::new($false).GetBytes($json);$f.Write($bytes,0,$bytes.Length) } finally { $f.Dispose() }
Write-Output $json

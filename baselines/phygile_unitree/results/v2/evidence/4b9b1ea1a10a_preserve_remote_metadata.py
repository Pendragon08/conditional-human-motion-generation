from pathlib import Path
import subprocess, tarfile, io, hashlib, json
root = Path(__file__).parent
remote_root = "/data/code/ouyanghao/oyh/phygile-unitree-20261022"
remote_command = "cd " + remote_root + " && find runs/closure_unitree_v3 -type f \\( -name '*.json' -o -name '*.log' -o -name '*.txt' \\) -print0 | tar --null -T - -czf - logs/closure_unitree_v3_stdout.txt logs/closure_unitree_v3_stderr.txt logs/closure_unitree_v3_exit.json"
result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "ouyanghao@10.147.17.32", remote_command], capture_output=True, check=False)
(root / "metadata_transfer_stderr.txt").write_bytes(result.stderr)
if result.returncode:
    raise RuntimeError("Read-only metadata transfer failed: " + str(result.returncode))
archive = root / "remote_metadata.tar.gz"
archive.write_bytes(result.stdout)
destination = root / "remote"
destination.mkdir()
with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:gz") as bundle:
    for member in bundle.getmembers():
        target = (destination / member.name).resolve()
        if not target.is_relative_to(destination.resolve()) or not member.isfile():
            raise RuntimeError("Unexpected archive member: " + member.name)
    bundle.extractall(destination, filter="data")
inventory = [{"path": str(p.relative_to(destination)).replace(chr(92), "/"), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(destination.rglob("*")) if p.is_file()]
(root / "copied_metadata_manifest.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
print(json.dumps({"metadata_files": len(inventory), "metadata_bytes": sum(x["bytes"] for x in inventory), "archive_sha256": hashlib.sha256(result.stdout).hexdigest(), "copied_arrays": list(destination.rglob("*.npz"))}))


#!/usr/bin/env python3
"""Verify downloaded release assets and restore selected manifest members.

Only Python's standard library is required. No network access and no training.
Split assets are read as a virtual stream; no concatenated tar is written.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tarfile
import tempfile

BLOCK = 1024 * 1024
MANIFESTS = {"MANIFEST.json", "PUBLICATION_MANIFEST.json"}


def safe_name(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Invalid archive member path")
    p = PurePosixPath(value)
    if p.is_absolute() or any(x in ("", ".", "..") for x in value.split("/")):
        raise ValueError("Unsafe archive path: " + value)
    if any(":" in x for x in p.parts):
        raise ValueError("Drive/stream names are forbidden: " + value)
    return value


def identity(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)


def open_regular(path):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ValueError("Expected regular, non-symlink file: " + str(path))
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
    f = os.fdopen(fd, "rb")
    if identity(os.fstat(f.fileno())) != identity(before):
        f.close()
        raise ValueError("File changed while opening: " + str(path))
    return f


class ConcatReader(io.RawIOBase):
    """Sequential concatenation, bounded memory, with post-read identity checks."""
    def __init__(self, paths, identities):
        self.paths = paths
        self.identities = identities
        self.index = 0
        self.current = None

    def readable(self):
        return True

    def readinto(self, target):
        view = memoryview(target)
        count = 0
        while count < len(view) and self.index < len(self.paths):
            path = self.paths[self.index]
            if self.current is None:
                self.current = open_regular(path)
                if identity(os.fstat(self.current.fileno())) != self.identities[self.index]:
                    raise ValueError("Downloaded source changed after verification: " + str(path))
            n = self.current.readinto(view[count:])
            if n:
                count += n
                continue
            if identity(os.fstat(self.current.fileno())) != self.identities[self.index]:
                raise ValueError("Downloaded source changed while reading: " + str(path))
            self.current.close()
            self.current = None
            self.index += 1
        return count

    def close(self):
        if self.current:
            self.current.close()
        super().close()


def digest_file(path):
    with open_regular(path) as f:
        before = identity(os.fstat(f.fileno()))
        h = hashlib.sha256()
        for data in iter(lambda: f.read(BLOCK), b""):
            h.update(data)
        if identity(os.fstat(f.fileno())) != before or identity(path.lstat()) != before:
            raise ValueError("File changed while hashing: " + str(path))
        return before, h.hexdigest()


def asset_spec(receipt, asset_name):
    if "reconstructed_sha256" in receipt:
        if receipt.get("status") != "completed_all_parts_and_source_hashes_verified":
            raise ValueError("Split-archive receipt is not complete")
        parts = sorted(receipt["parts"], key=lambda x: x["index"])
        if [p["index"] for p in parts] != list(range(1, int(receipt["part_count"]) + 1)):
            raise ValueError("Missing or repeated split-part indices")
        if asset_name and asset_name != receipt["reconstructed_name"]:
            raise ValueError("--asset must match the reconstructed tar name")
        return parts, receipt["reconstructed_bytes"], receipt["reconstructed_sha256"]
    assets = receipt.get("assets", receipt.get("parts", []))
    if asset_name:
        assets = [a for a in assets if a["name"] == asset_name]
    if len(assets) != 1:
        raise ValueError("Choose exactly one independent tar with --asset")
    a = assets[0]
    return [a], a.get("bytes", a.get("size")), a["sha256"]


def verify_downloads(directory, records, expected_bytes, expected_sha):
    directory = directory.resolve(strict=True)
    paths, identities = [], []
    total = 0
    whole = hashlib.sha256()
    for a in records:
        name = safe_name(a["name"])
        if "/" in name:
            raise ValueError("Asset names must be basenames")
        p = directory / name
        part_hash = hashlib.sha256()
        count = 0
        with open_regular(p) as f:
            initial = identity(os.fstat(f.fileno()))
            for data in iter(lambda: f.read(BLOCK), b""):
                count += len(data)
                part_hash.update(data)
                whole.update(data)
            if identity(os.fstat(f.fileno())) != initial or identity(p.lstat()) != initial:
                raise ValueError("Asset changed while verifying: " + name)
        if count != a.get("bytes", a.get("size")) or part_hash.hexdigest() != a["sha256"]:
            raise ValueError("Asset SHA256/byte-count mismatch: " + name)
        total += count
        paths.append(p)
        identities.append(initial)
    if total != expected_bytes or whole.hexdigest() != expected_sha:
        raise ValueError("Concatenated archive SHA256/byte-count mismatch")
    return paths, identities


def manifest_rows(manifest):
    rows = manifest.get("rows", manifest.get("files", []))
    if not rows:
        raise ValueError("Manifest has no files")
    result = {}
    for row in rows:
        name = safe_name(row.get("member", row.get("path")))
        sha = row["sha256"]
        size = row["bytes"]
        if not isinstance(size, int) or size < 0 or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError("Invalid manifest hash/size: " + name)
        record = {"member": name, "bytes": size, "sha256": sha,
                  "original_path": row.get("original_path", row.get("remote_path", row.get("path", name)))}
        if name in result and (result[name]["sha256"], result[name]["bytes"]) != (sha, size):
            raise ValueError("Conflicting manifest entries: " + name)
        result[name] = record
    return result


def inspect_archive(paths, identities, external_manifest):
    manifest = json.loads(external_manifest.read_text(encoding="utf-8-sig")) if external_manifest else None
    members = {}
    with io.BufferedReader(ConcatReader(paths, identities), BLOCK) as source:
        with tarfile.open(fileobj=source, mode="r|*") as archive:
            for item in archive:
                name = safe_name(item.name)
                if not item.isfile():
                    raise ValueError("Non-regular tar member refused: " + name)
                if name in MANIFESTS:
                    if item.size > 32 * BLOCK:
                        raise ValueError("Oversized embedded manifest")
                    embedded = json.load(archive.extractfile(item))
                    if manifest is None:
                        manifest = embedded
                    elif manifest != embedded:
                        raise ValueError("External and embedded manifests differ")
                    continue
                if name in members and members[name] != item.size:
                    raise ValueError("Conflicting duplicate tar member: " + name)
                members[name] = item.size
    if manifest is None:
        raise ValueError("No embedded manifest; supply the published --manifest JSON")
    rows = manifest_rows(manifest)
    if set(members) != set(rows):
        raise ValueError("Tar/manifest member sets differ")
    for name, size in members.items():
        if size != rows[name]["bytes"]:
            raise ValueError("Tar/manifest member size mismatch: " + name)
    return rows


def safe_destination(root, member, create=False):
    safe_name(member)
    current = root
    for part in PurePosixPath(member).parts[:-1]:
        current = current / part
        if current.exists() or current.is_symlink():
            if current.is_symlink() or not current.is_dir():
                raise ValueError("Unsafe output parent: " + str(current))
        elif create:
            current.mkdir()
    target = root.joinpath(*PurePosixPath(member).parts)
    if target.is_symlink():
        raise ValueError("Refuse output symlink: " + str(target))
    if not target.resolve(strict=False).is_relative_to(root):
        raise ValueError("Output escapes destination")
    return target


def restore(paths, identities, rows, selected, output, reserve_bytes):
    if output.is_symlink():
        raise ValueError("Output directory may not be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    root = output.resolve(strict=True)
    needed, skipped = 0, []
    for name in selected:
        destination = safe_destination(root, name)
        row = rows[name]
        if destination.exists():
            ident, sha = digest_file(destination)
            if ident[2] != row["bytes"] or sha != row["sha256"]:
                raise ValueError("Existing content differs; nothing overwritten: " + str(destination))
            skipped.append(name)
        else:
            needed += row["bytes"]
    if shutil.disk_usage(root).free < needed + reserve_bytes:
        raise ValueError("Insufficient output space: need selected bytes plus reserve (no concatenated tar is needed)")
    pending = set(selected) - set(skipped)
    restored = []
    with io.BufferedReader(ConcatReader(paths, identities), BLOCK) as source:
        with tarfile.open(fileobj=source, mode="r|*") as archive:
            for item in archive:
                name = safe_name(item.name)
                if name not in pending:
                    continue
                if not item.isfile() or item.size != rows[name]["bytes"]:
                    raise ValueError("Tar changed after inspection: " + name)
                destination = safe_destination(root, name, create=True)
                fd, temporary = tempfile.mkstemp(prefix=".restore-", suffix=".partial", dir=destination.parent)
                try:
                    h, count = hashlib.sha256(), 0
                    with os.fdopen(fd, "wb") as dest, archive.extractfile(item) as src:
                        for data in iter(lambda: src.read(BLOCK), b""):
                            dest.write(data)
                            h.update(data)
                            count += len(data)
                        dest.flush()
                        os.fsync(dest.fileno())
                    if count != rows[name]["bytes"] or h.hexdigest() != rows[name]["sha256"]:
                        raise ValueError("Member content hash mismatch: " + name)
                    # A hard link publishes verified bytes atomically and refuses an existing name.
                    os.link(temporary, destination)
                    restored.append(name)
                    pending.remove(name)
                finally:
                    # Delete only the temporary file created by this invocation.
                    if os.path.exists(temporary):
                        os.unlink(temporary)
    if pending:
        raise ValueError("Selected members were not found")
    return {"status": "verified_and_restored", "destination": str(root),
            "restored_members": restored, "already_identical_members": skipped,
            "downloaded_assets_preserved": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--downloads", required=True, type=Path)
    parser.add_argument("--asset", help="Select one independent tar asset from a receipt")
    parser.add_argument("--manifest", type=Path, help="Published external manifest when tar has none")
    parser.add_argument("--list", action="store_true", help="Verify the archive and list manifest members")
    parser.add_argument("--member", action="append", default=[], help="Exact archive member; repeat to select more")
    parser.add_argument("--all", action="store_true", help="Restore all manifest members")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reserve-bytes", type=int, default=1024**3 + 100 * 1024**2,
                        help="Keep this many output-volume bytes free (default 1 GiB + 100 MiB)")
    args = parser.parse_args(argv)
    if args.reserve_bytes < 0:
        parser.error("--reserve-bytes must be nonnegative")
    receipt = json.loads(args.receipt.read_text(encoding="utf-8-sig"))
    records, size, sha = asset_spec(receipt, args.asset)
    paths, identities = verify_downloads(args.downloads, records, size, sha)
    rows = inspect_archive(paths, identities, args.manifest)
    if args.list:
        print(json.dumps({"status": "archive_and_manifest_verified", "archive_bytes": size,
                          "archive_sha256": sha, "members": list(rows.values())}, ensure_ascii=False, indent=2))
        return 0
    if not args.output or (bool(args.member) == bool(args.all)):
        parser.error("Restore requires --output and exactly one of --member or --all")
    selected = set(rows) if args.all else set(args.member)
    if selected - set(rows):
        raise ValueError("Unknown selected member(s): " + repr(sorted(selected - set(rows))))
    print(json.dumps(restore(paths, identities, rows, selected, args.output, args.reserve_bytes),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        print("Restore refused: " + str(error), file=sys.stderr)
        raise SystemExit(1)

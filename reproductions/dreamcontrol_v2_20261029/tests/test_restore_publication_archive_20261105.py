"""Small archive-integrity tests; never reads project artifacts or credentials."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "src/restore_publication_archive_20261105.py"
spec = importlib.util.spec_from_file_location("publication_restore", SCRIPT)
restore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore)


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.downloads = self.base / "downloads"
        self.downloads.mkdir()
        self.payload = b"project checkpoint test\x00\x01"
        self.name = "weights/example.pt"
        self.sha = hashlib.sha256(self.payload).hexdigest()
        self.manifest = {"rows": [{"member": self.name, "bytes": len(self.payload), "sha256": self.sha,
                                    "original_path": "/original/model.pt"}]}
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            for name, content in [("MANIFEST.json", json.dumps(self.manifest).encode()), (self.name, self.payload)]:
                member = tarfile.TarInfo(name)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
        self.data = stream.getvalue()
        self.parts = []
        for i, offset in enumerate(range(0, len(self.data), 1111), 1):
            content = self.data[offset:offset + 1111]
            name = "test.part%04d" % i
            (self.downloads / name).write_bytes(content)
            self.parts.append({"index": i, "name": name, "bytes": len(content),
                               "sha256": hashlib.sha256(content).hexdigest()})
        self.receipt = {"status": "completed_all_parts_and_source_hashes_verified", "parts": self.parts,
                        "part_count": len(self.parts), "reconstructed_name": "test.tar",
                        "reconstructed_bytes": len(self.data), "reconstructed_sha256": hashlib.sha256(self.data).hexdigest()}

    def verified(self):
        args = restore.asset_spec(self.receipt, None)
        paths, ids = restore.verify_downloads(self.downloads, *args)
        return paths, ids, restore.inspect_archive(paths, ids, None)

    def test_split_restore_and_identical_skip(self):
        paths, ids, rows = self.verified()
        out = self.base / "restored"
        result = restore.restore(paths, ids, rows, {self.name}, out, 0)
        self.assertEqual(result["restored_members"], [self.name])
        self.assertEqual((out / self.name).read_bytes(), self.payload)
        self.assertEqual(len(list(self.downloads.iterdir())), len(self.parts))
        again = restore.restore(paths, ids, rows, {self.name}, out, 0)
        self.assertEqual(again["already_identical_members"], [self.name])
        self.assertFalse((self.downloads / "test.tar").exists())

    def test_different_existing_content_refused(self):
        paths, ids, rows = self.verified()
        out = self.base / "restored"
        (out / self.name).parent.mkdir(parents=True)
        (out / self.name).write_bytes(b"do not replace")
        with self.assertRaisesRegex(ValueError, "Existing content differs"):
            restore.restore(paths, ids, rows, {self.name}, out, 0)
        self.assertEqual((out / self.name).read_bytes(), b"do not replace")

    def test_corrupt_part_refused(self):
        path = self.downloads / self.parts[0]["name"]
        content = bytearray(path.read_bytes())
        content[0] ^= 1
        path.write_bytes(content)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.verified()

    def test_missing_part_refused(self):
        self.receipt["parts"] = self.receipt["parts"][:-1]
        with self.assertRaisesRegex(ValueError, "Missing or repeated"):
            restore.asset_spec(self.receipt, None)

    def test_unsafe_paths(self):
        for name in ("../escape", "/absolute", "C:/file", "safe/../escape", "safe\\file", "safe/file:ads"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                restore.safe_name(name)

    def test_low_space_refused_before_content_write(self):
        paths, ids, rows = self.verified()
        out = self.base / "restored"
        usage = type("Usage", (), {"free": 0})()
        with patch.object(restore.shutil, "disk_usage", return_value=usage):
            with self.assertRaisesRegex(ValueError, "Insufficient output space"):
                restore.restore(paths, ids, rows, {self.name}, out, 1)
        self.assertFalse((out / self.name).exists())

    def test_download_changed_after_verification(self):
        paths, ids, rows = self.verified()
        paths[0].write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed after verification"):
            restore.restore(paths, ids, rows, {self.name}, self.base / "restored", 0)

    def test_incomplete_receipt_refused(self):
        self.receipt["status"] = "failed_preserved"
        with self.assertRaisesRegex(ValueError, "not complete"):
            restore.asset_spec(self.receipt, None)


if __name__ == "__main__":
    unittest.main()

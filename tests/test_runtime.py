import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.setup_local_llm import fetch, extract_verified_zip


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / 'tmp/runtime-tests'
        root.mkdir(parents=True, exist_ok=True)
        # Preserve fixtures: user explicitly prohibits automatic deletion.
        self.directory = Path(tempfile.mkdtemp(dir=root))

    def test_download_verified_and_no_overwrite(self):
        content = b'fixture-runtime-bytes'
        expected = hashlib.sha256(content).hexdigest()
        target = self.directory / 'runtime.bin'
        with patch('urllib.request.urlopen', return_value=io.BytesIO(content)) as request:
            fetch('https://example.invalid/runtime', target, expected)
            self.assertEqual(target.read_bytes(), content)
            fetch('https://example.invalid/runtime', target, expected)
            self.assertEqual(request.call_count, 1)
        with self.assertRaises(ValueError):
            fetch('https://example.invalid/runtime', target, '0' * 64)
        self.assertEqual(target.read_bytes(), content)

    def test_extract_rejects_traversal_and_existing_changes(self):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('../escape.txt', 'bad')
        archive.seek(0)
        with self.assertRaises(ValueError):
            extract_verified_zip(archive, self.directory / 'runtime')
        self.assertFalse((self.directory / 'escape.txt').exists())
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('server.txt', 'verified')
        archive.seek(0)
        extract_verified_zip(archive, self.directory / 'runtime')
        target = self.directory / 'runtime/server.txt'
        target.write_text('local-edit', encoding='utf-8')
        archive.seek(0)
        with self.assertRaises(ValueError):
            extract_verified_zip(archive, self.directory / 'runtime')
        self.assertEqual(target.read_text(), 'local-edit')


if __name__ == '__main__':
    unittest.main()

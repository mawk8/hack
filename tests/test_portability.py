"""A venv must not be shadowed by a different machine's .tools/python."""
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch


class PortabilityTests(unittest.TestCase):
    def test_venv_keeps_its_own_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        local = str(root / '.tools' / 'python')
        for source in ('avito/__init__.py', 'scripts/audit_data.py'):
            with self.subTest(source=source), \
                 patch.object(sys, 'prefix', str(root / 'synthetic-venv')), \
                 patch.object(Path, 'is_dir', return_value=True), \
                 patch.object(sys, 'path', [p for p in sys.path if Path(p) != Path(local)]):
                runpy.run_path(str(root / source))
                self.assertNotIn(local, sys.path)


if __name__ == '__main__':
    unittest.main()

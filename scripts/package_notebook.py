"""Create a source-only upload bundle, never raw advertisements or model weights."""
from datetime import datetime, timezone
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def package():
    target = ROOT / 'data/export' / ('avito-source-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.zip')
    target.parent.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / 'requirements-prototype.txt']
    for directory, suffix in [('avito', '.py'), ('avito', '.html'), ('scripts', '.py'), ('scripts', '.ps1')]:
        paths.extend((ROOT / directory).glob('*' + suffix))
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as out:
        for path in sorted(set(paths)):
            out.write(path, str(path.relative_to(ROOT)).replace('\\', '/'))
    print(target)
    return target


if __name__ == '__main__':
    package()

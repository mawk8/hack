"""Install official portable Windows llama.cpp and pinned Qwen GGUF locally.

Run explicitly with --install to download (~5 GB model plus runtime). Existing
files are verified, never overwritten or deleted. No system installation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = '7c41481f57cb95916b40956ab2f0b139b296d974'
MODEL_NAME = 'Qwen3-8B-Q4_K_M.gguf'
MODEL_SHA256 = 'd98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785'
TAG_URL = 'https://github.com/ggml-org/llama.cpp/releases/download/v1.0.0/nightly-tag.txt'
TAG_SHA256 = 'e8352faf0647cae078e6be57cd129ba534ec0664b162231a68955379e9c05e4b'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fetch(url, destination, expected_sha):
    """Create-only download; preserve interrupted/mismatched files for inspection."""
    destination = Path(destination)
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha):
        raise ValueError('A published SHA256 is required')
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={'User-Agent': 'avito-hackathon-prototype'})
        with urllib.request.urlopen(request, timeout=60) as response, destination.open('xb') as out:
            while chunk := response.read(4 * 1024 * 1024):
                out.write(chunk)
    if sha256(destination) != expected_sha:
        raise ValueError(f'Checksum mismatch; file preserved, not used: {destination}')
    return destination


def extract_verified_zip(archive, directory):
    directory = Path(directory).resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (directory / member.filename).resolve()
            if not target.is_relative_to(directory) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError('Unsafe path in runtime archive')
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            content = bundle.read(member)
            if target.exists():
                if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(content).digest():
                    raise ValueError(f'Existing runtime file differs; preserved: {target}')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as out:
                    out.write(content)


def install(root=ROOT):
    root = Path(root).resolve()
    downloads = root / '.tools/llama-downloads'
    tag_path = fetch(TAG_URL, downloads / 'v1.0.0-nightly-tag.txt', TAG_SHA256)
    tag = tag_path.read_text(encoding='utf-8').strip()
    if not re.fullmatch(r'b\d+', tag):
        raise ValueError('Unexpected official runtime tag')
    request = urllib.request.Request(
        f'https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}',
        headers={'User-Agent': 'avito-hackathon-prototype'})
    with urllib.request.urlopen(request, timeout=60) as response:
        release = json.load(response)
    assets = release['assets']
    # Driver on the available laptop is566.07; avoid CUDA13 builds.
    builds = [a for a in assets if a['name'].startswith(f'llama-{tag}-bin-win-cuda-12')
              and a['name'].endswith('-x64.zip')]
    if not builds:
        raise RuntimeError('Pinned release has no Windows CUDA12 build; select a verified compatible release explicitly')
    selected = sorted(builds, key=lambda a: a['name'])[0]
    cuda = re.search(r'cuda-(12[.\d]*)-x64', selected['name']).group(1)
    libraries = [a for a in assets if a['name'].startswith('cudart-')
                 and f'cuda-{cuda}-x64' in a['name'] and a['name'].endswith('.zip')]
    runtime = root / '.tools' / f'llama-{tag}-cuda-{cuda}'
    records = []
    for asset in [selected, *libraries]:
        checksum = (asset.get('digest') or '').removeprefix('sha256:')
        archive = fetch(asset['browser_download_url'], downloads / asset['name'], checksum)
        extract_verified_zip(archive, runtime)
        records.append({'name': asset['name'], 'sha256': checksum, 'url': asset['browser_download_url']})
    servers = list(runtime.rglob('llama-server.exe'))
    if len(servers) != 1:
        raise RuntimeError('Expected one llama-server.exe in verified runtime')
    model_url = f'https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/{MODEL_REVISION}/{MODEL_NAME}'
    model = fetch(model_url, root / 'data/models' / MODEL_NAME, MODEL_SHA256)
    record = {'runtime_tag': tag, 'assets': records, 'server': str(servers[0]),
              'model': str(model), 'model_sha256': MODEL_SHA256, 'model_revision': MODEL_REVISION,
              'host': '127.0.0.1', 'port': 8080, 'ctx_size': 8192, 'parallel': 1}
    target = root / '.tools/llama-runtime.json'
    encoded = json.dumps(record, indent=2, ensure_ascii=False) + '\n'
    if target.exists():
        if target.read_text(encoding='utf-8') != encoded:
            raise ValueError('Existing runtime manifest differs; preserved')
    else:
        with target.open('x', encoding='utf-8') as out:
            out.write(encoded)
    print(encoded)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true', help='Download official runtime and5GB GGUF')
    args = parser.parse_args()
    if args.install:
        install()
    else:
        print(json.dumps({'installed': (ROOT / '.tools/llama-runtime.json').exists(),
                          'model': MODEL_NAME, 'download_requires': '--install'}, indent=2))

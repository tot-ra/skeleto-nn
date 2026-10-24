"""Fetch explicit source pins and the upstream SHA-256-checked model bundle."""
from __future__ import annotations
from pathlib import Path
import subprocess
import sys

PORT_REVISION = '60a5f4c0b5eef3d893d4a8c253cc3fc68924c559'
GGML_REVISION = 'da8e2a54b919557b816b1c2beb432ccb766f7ce0'


def checkout(url: str, path: Path, revision: str) -> None:
    if (path / '.git').exists():
        actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != revision:
            raise RuntimeError(f'{path}: source revision differs; refusing to overwrite it')
        return
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f'{path}: refusing to overwrite a nonempty source directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'clone', '--depth', '1', '--no-recurse-submodules', url, str(path)], check=True)
    actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != revision:
        subprocess.run(['git', '-C', str(path), 'fetch', '--depth', '1', 'origin', revision], check=True)
    subprocess.run(['git', '-C', str(path), 'checkout', '--detach', revision], check=True)


def prepare(path: Path) -> None:
    checkout('https://github.com/localai-org/motion-bricks.cpp.git', path, PORT_REVISION)
    # The upstream submodule commit is unavailable. This alternative is an explicit
    # tested pin, never an automatic fallback to the mutable upstream main branch.
    checkout('https://github.com/ggml-org/ggml.git', path / 'ggml', GGML_REVISION)
    subprocess.run([sys.executable, str(path / 'scripts/download_gguf_weights.py')], check=True)

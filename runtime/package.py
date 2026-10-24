"""Copy a verified standalone bundle into a macOS game app or executable folder."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
from .build import digest


def package(source: Path, destination: Path) -> Path:
    source = source.resolve()
    destination = destination.resolve()
    if destination.suffix == '.app':
        if not (destination / 'Contents/MacOS').is_dir():
            raise ValueError('destination is not an exported macOS app')
        destination = destination / 'Contents/Resources/neural_runtime'
    else:
        destination = destination / 'neural_runtime'
    manifest = json.loads((source / 'runtime.json').read_text())
    if manifest.get('protocol') != 'skeleto-nn/motion/1':
        raise ValueError('unsupported runtime protocol')
    verified = []
    for name, entry in manifest['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe bundle path')
        original = source / relative
        if original.stat().st_size != entry['bytes'] or digest(original) != entry['sha256']:
            raise ValueError(f'corrupt bundle file: {name}')
        verified.append((original, relative))
    for original, relative in verified:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
    shutil.copy2(source / 'runtime.json', destination / 'runtime.json')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(package(args.source, args.destination))

"""Build and deploy the standalone native motion worker and licensed model bundle.

python -m runtime.build --source /path/to/motion-bricks.cpp --output /path/to/neural_runtime
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import uuid
from .dependencies import GGML_REVISION, PORT_REVISION, prepare



def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def build(source: Path, output: Path, build_dir: Path) -> dict:
    source, output, build_dir = source.resolve(), output.resolve(), build_dir.resolve()
    if not (source / 'CMakeLists.txt').is_file():
        prepare(source)
    for directory, revision in [(source, PORT_REVISION), (source / 'ggml', GGML_REVISION)]:
        actual = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != revision:
            raise RuntimeError(f'{directory}: expected {revision}, got {actual}')
    subprocess.run(['cmake', '-S', str(Path(__file__).parent), '-B', str(build_dir),
                    f'-DMOTIONBRICKS_SOURCE={source}', '-DCMAKE_BUILD_TYPE=Release'], check=True)
    subprocess.run(['cmake', '--build', str(build_dir), '--target', 'skeleto_motion_worker', '-j', '4'], check=True)
    executable = build_dir / ('skeleto_motion_worker.exe' if platform.system() == 'Windows' else 'skeleto_motion_worker')
    files: list[tuple[Path, Path]] = [(executable, Path(executable.name))]
    upstream = source / 'generated'
    manifest = json.loads((source / 'scripts/hf/MotionBricks-G1-GGML/MANIFEST.json').read_text())
    for entry in manifest['files']:
        relative = Path(entry['path'])
        if relative.parts[0] == 'styles' and relative.name not in {'idle.mbstyle', 'walk.mbstyle', 'slow_walk.mbstyle'}:
            continue
        original = upstream / relative
        if original.stat().st_size != entry['bytes'] or digest(original) != entry['sha256']:
            raise RuntimeError(f'model verification failed: {original}')
        files.append((original, relative))
    for name in ['UPSTREAM_LICENSE', 'NOTICE']:
        files.append((source / 'scripts/hf/MotionBricks-G1-GGML' / name, Path(name)))
    files.append((source / 'LICENSE', Path('MOTIONBRICKS_CPP_LICENSE')))
    files.append((source / 'ggml/LICENSE', Path('GGML_LICENSE')))
    files.append((Path(__file__).resolve().parents[1] / 'LICENSE', Path('SKELETO_LICENSE')))
    result = {'protocol': 'skeleto-nn/motion/1', 'platform': platform.system(),
              'architecture': platform.machine(), 'port_revision': PORT_REVISION,
              'ggml_revision': GGML_REVISION, 'files': {}}
    output.mkdir(parents=True, exist_ok=True)
    for original, relative in files:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size != original.stat().st_size or digest(target) != digest(original):
            temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.part')
            try:
                shutil.copy2(original, temporary)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        result['files'][relative.as_posix()] = {'bytes': target.stat().st_size, 'sha256': digest(target)}
    (output / '.gdignore').touch()
    (output / 'runtime.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repository = Path(__file__).resolve().parents[1]
    parser.add_argument('--source', type=Path, default=repository / 'runs/native_source')
    parser.add_argument('--output', type=Path, default=repository / 'runs/native_runtime')
    parser.add_argument('--build-dir', type=Path)
    args = parser.parse_args()
    result = build(args.source, args.output, args.build_dir or args.output.parent / 'neural_worker_build')
    print(json.dumps({'output': str(args.output.resolve()), 'files': len(result['files'])}))


if __name__ == '__main__':
    main()

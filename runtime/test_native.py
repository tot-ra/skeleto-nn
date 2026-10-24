"""Integration checks with real weights; pass --worker and --bundle explicitly."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import subprocess


def verify(worker: Path, bundle: Path) -> dict:
    process = subprocess.Popen([str(worker.resolve()), str(bundle.resolve()), '2'],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        metadata = json.loads(process.stdout.readline())
        assert metadata['protocol'] == 'skeleto-nn/motion/1'
        assert len(metadata['joints']) == 34
        timings = {}
        for index, mode in enumerate(['locomotion', 'jump_start', 'jump_air', 'jump_land', 'door', 'ladder']):
            process.stdin.write(f'plan {23+index} 8 0.7 0 1 0 1 {mode} 0.5\n')
            process.stdin.flush()
            response = json.loads(process.stdout.readline())
            assert response['mode'] == mode and response['id'] == 23 + index
            frames = response['frames']
            assert 24 <= frames <= 64 and frames % 4 == 0
            assert len(response['roots']) == frames * 3
            assert len(response['rotations']) == frames * 34 * 4
            assert all(math.isfinite(v) for v in response['roots'] + response['rotations'])
            for first in range(0, len(response['rotations']), 4):
                assert abs(sum(v*v for v in response['rotations'][first:first+4]) - 1) < 0.001
            timings[mode] = response['inference_ms']
        for invalid in ['not a command', 'plan 3 0 -2 0 1 0 1 locomotion 0',
                        'plan 3 0 1 0 1 0 1 unknown 0']:
            process.stdin.write(invalid + '\n'); process.stdin.flush()
            assert 'error' in json.loads(process.stdout.readline())
        process.stdin.write('plan 90 0 0 0 1 0 1 locomotion 0\n'); process.stdin.flush()
        assert json.loads(process.stdout.readline())['id'] == 90
        process.stdin.write('quit\n'); process.stdin.flush()
        assert process.wait(timeout=5) == 0
        return {'passed_goals': list(timings), 'inference_ms': timings, 'invalid_requests': 3,
                'worker_survives_invalid_input': True, 'clean_exit': True}
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.worker, args.bundle), indent=2))

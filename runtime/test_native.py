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


def verify_rollout(worker: Path, bundle: Path) -> dict:
    """Sustained generated context, action repetition and locomotion recovery."""
    process = subprocess.Popen([str(worker.resolve()), str(bundle.resolve()), '2'],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    phases = [('walk', 'locomotion', 0.8, 64), ('ladder', 'ladder', 0.0, 64),
              ('door', 'door', 0.0, 32), ('jump', 'jump_air', 0.0, 32),
              ('walk_after_actions', 'locomotion', 0.8, 64)]
    results = []
    try:
        metadata = json.loads(process.stdout.readline())
        indices = {joint['name']: i for i, joint in enumerate(metadata['joints'])}
        request_id = 1000
        for name, mode, speed, plans in phases:
            minimum_up, maximum_knee = 1.0, 0.0
            for i in range(plans):
                request_id += 1
                process.stdin.write(f'plan {request_id} 10 {speed} 0 1 0 1 {mode} {(i % 20) / 20}\n')
                process.stdin.flush()
                response = json.loads(process.stdout.readline())
                assert response['mode'] == mode and response['id'] == request_id
                for frame in range(min(14, response['frames'])):
                    first = frame * 34 * 4
                    x, _, z, _ = response['rotations'][first:first+4]
                    minimum_up = min(minimum_up, 1 - 2 * (x*x + z*z))
                    for side in ['left', 'right']:
                        offset = first + indices[side + '_knee_skel'] * 4
                        x, _, _, w = response['rotations'][offset:offset+4]
                        maximum_knee = max(maximum_knee, abs(2 * math.atan2(x, w)))
            assert minimum_up > 0.90, (name, 'body falls during repeated goals', minimum_up)
            assert maximum_knee < 2.4, (name, 'joint angles accumulate', maximum_knee)
            results.append({'phase': name, 'plans': plans, 'min_root_up': minimum_up,
                            'max_knee_rad': maximum_knee})
        return {'passed': True, 'plans': sum(phase[3] for phase in phases), 'phases': results}
    finally:
        process.stdin.write('quit\n'); process.stdin.flush()
        process.wait(timeout=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--long', action='store_true', help='check sustained action/context stability')
    args = parser.parse_args()
    result = verify_rollout(args.worker, args.bundle) if args.long else verify(args.worker, args.bundle)
    print(json.dumps(result, indent=2))

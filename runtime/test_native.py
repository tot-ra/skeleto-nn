"""Integration checks with real weights; pass --worker and --bundle explicitly.

Default: the v1 contract (six modes, invalid input, clean exit) plus the protocol v2
checks (handshake, styles, joint-group goals, malformed v2 lines). `--long` runs the
256-plan sustained regression instead.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess


class Worker:
    def __init__(self, worker: Path, bundle: Path) -> None:
        self.process = subprocess.Popen([str(worker.resolve()), str(bundle.resolve()), '2'],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.metadata = json.loads(self.process.stdout.readline())
        self.indices = {joint['name']: i for i, joint in enumerate(self.metadata['joints'])}

    def ask(self, line: str) -> dict:
        self.process.stdin.write(line + '\n')
        self.process.stdin.flush()
        return json.loads(self.process.stdout.readline())

    def close(self) -> int:
        if self.process.poll() is None:
            try:
                self.process.stdin.write('quit\n')
                self.process.stdin.flush()
                return self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait(timeout=5)
        return self.process.returncode

    def angle(self, response: dict, frame: int, joint: str) -> float:
        """Rotation angle of a joint in degrees (axis-independent)."""
        offset = (frame * 34 + self.indices[joint]) * 4
        x, y, z, w = response['rotations'][offset:offset + 4]
        return math.degrees(2 * math.atan2(math.sqrt(x * x + y * y + z * z), w))


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


def stride_period(series: list[float]) -> int:
    """Lag of the autocorrelation maximum in frames (8..60): one knee cycle."""
    mean = statistics.mean(series)
    centred = [value - mean for value in series]
    best, best_lag = -math.inf, 0
    for lag in range(8, 60):
        score = sum(a * b for a, b in zip(centred, centred[lag:])) / (len(centred) - lag)
        if score > best:
            best, best_lag = score, lag
    return best_lag


def verify_v2(worker: Path, bundle: Path) -> dict:
    """Protocol v2: handshake, named styles, joint-group goals, malformed lines."""
    checks = 0
    plain = Worker(worker, bundle)
    other = Worker(worker, bundle)
    try:
        metadata = plain.metadata
        assert metadata['protocol'] == 'skeleto-nn/motion/1', 'v1 clients still accept the metadata'
        assert metadata['protocol_version'] == 2
        assert {'idle', 'walk', 'slow_walk'} <= set(metadata['styles'])
        checks += 3
        hello = plain.ask('hello')
        assert hello['hello'] is True and hello['protocol_version'] == 2 and hello['v1'] is True
        assert hello['styles'] == metadata['styles']
        assert {'legs', 'arms', 'arm_l', 'arm_r', 'spine', 'head'} >= set(hello['groups']) >= {'legs', 'arm_l', 'arm_r', 'spine'}
        assert len(hello['groups']['arm_r']) == 8 and len(hello['groups']['legs']) == 14
        assert set(hello['modes']) == {'locomotion', 'jump_start', 'jump_air', 'jump_land', 'door', 'ladder'}
        assert 'protocol' not in hello, 'the handshake must not look like a metadata line'
        checks += 6
        assert plain.ask('version')['protocol_version'] == 2
        checks += 1

        # Styles: the same seeds and commands through two fresh workers, walk vs slow_walk.
        # At one explicit target speed the two exemplars share their joint poses (the
        # bundled slow_walk differs from walk only in root pace and configured speed, 0.3 vs
        # 1.0 m/s), so the knee difference is reported, not asserted. At the style pace
        # (SPEED -1) slow_walk must travel slower and take longer strides in time.
        assert hello['style_speeds']['slow_walk'] < hello['style_speeds']['walk']
        checks += 1
        knee = {'walk': [], 'slow_walk': []}
        timings = {'walk': [], 'slow_walk': []}
        for i in range(12):
            for runner, style in [(plain, 'walk'), (other, 'slow_walk')]:
                response = runner.ask(f'plan2 {i + 1} 10 0.7 0 1 0 1 locomotion 0 {style} none -')
                assert 'error' not in response, response
                assert response['style'] == style and response['masked'] == 0 and response['frames'] >= 24
                timings[style].append(response['inference_ms'])
                knee[style] += [runner.angle(response, frame, 'left_knee_skel') for frame in range(10)]
        checks += 1
        knee_difference = statistics.mean(abs(a - b) for a, b in zip(knee['walk'], knee['slow_walk']))
        pace_knee = {'walk': [], 'slow_walk': []}
        travel = {'walk': [], 'slow_walk': []}
        for i in range(16):
            for runner, style in [(plain, 'walk'), (other, 'slow_walk')]:
                response = runner.ask(f'plan2 {20 + i} 10 -1 0 1 0 1 locomotion 0 {style} none -')
                assert 'error' not in response, response
                assert response['style'] == style
                if i < 4:
                    continue  # settle into the pace
                pace_knee[style] += [runner.angle(response, frame, 'left_knee_skel') for frame in range(10)]
                roots = response['roots']
                travel[style].append(math.hypot(roots[30] - roots[0], roots[32] - roots[2]) * 3)
        periods = {style: stride_period(values) for style, values in pace_knee.items()}
        paces = {style: statistics.mean(values) for style, values in travel.items()}
        assert paces['slow_walk'] < paces['walk'] * 0.6, ('slow_walk pace is not slower', paces)
        assert periods['slow_walk'] >= periods['walk'] + 4, ('slow_walk stride timing equals walk', periods)
        assert all(10 <= period <= 60 for period in periods.values()), periods
        checks += 4
        assert 'error' in plain.ask('plan2 40 10 0.7 0 1 0 1 locomotion 0 nosuchstyle none -')
        assert 'error' in plain.ask('plan 41 10 -1 0 1 0 1 locomotion 0'), 'v1 lines keep rejecting negative speed'
        assert 'error' in plain.ask('plan2 42 10 -1 0 1 0 1 locomotion 0 - none -'), 'style pace needs a style'
        checks += 3

        # Joint-group goal: an arm_r reach over locomotion leaves hips, knees and the root
        # exactly on the unmasked plan (same seeds and commands, two fresh workers).
        plain.close(); other.close()
        plain = Worker(worker, bundle)
        other = Worker(worker, bundle)
        goal = 'right_shoulder_pitch_skel=%.6f,0,0,%.6f;right_elbow_skel=%.6f,0,0,%.6f' % (
            math.sin(-0.45), math.cos(-0.45), math.sin(0.225), math.cos(0.225))
        leg_difference = 0.0
        root_difference = 0.0
        goal_timings = []
        shoulder_goal = shoulder_plain = 0.0
        for i in range(10):
            line = f'plan2 {i + 1} 10 0.7 0 1 0 1 locomotion 0 walk %s %s'
            a = plain.ask(line % ('none', '-'))
            b = other.ask(line % ('arm_r', goal))
            assert 'error' not in a and 'error' not in b, (a, b)
            assert a['frames'] >= 24 and b['frames'] == 24 and b['masked'] == 8
            goal_timings.append(b['inference_ms'])
            for frame in range(b['frames']):
                for joint in ['left_hip_pitch_skel', 'right_hip_pitch_skel', 'left_knee_skel', 'right_knee_skel']:
                    leg_difference = max(leg_difference, abs(plain.angle(a, frame, joint) - other.angle(b, frame, joint)))
                for axis in range(3):
                    root_difference = max(root_difference, abs(a['roots'][frame * 3 + axis] - b['roots'][frame * 3 + axis]))
            shoulder_goal = other.angle(b, 23, 'right_shoulder_pitch_skel')
            shoulder_plain = plain.angle(a, 23, 'right_shoulder_pitch_skel')
        assert leg_difference < 2.0, ('masked arm goal moved the legs', leg_difference)
        assert root_difference < 1e-6, ('masked arm goal moved the root', root_difference)
        assert abs(shoulder_goal - math.degrees(0.9)) < 20.0, ('shoulder did not reach the goal', shoulder_goal)
        assert abs(shoulder_goal - shoulder_plain) > 10.0, ('goal arm equals the locomotion arm', shoulder_goal, shoulder_plain)
        checks += 4
        # Goals gone: the arm blends back over one horizon, then plain locomotion resumes.
        release = other.ask('plan2 50 10 0.7 0 1 0 1 locomotion 0 walk none -')
        assert release['frames'] == 24 and release['masked'] == 0
        assert other.ask('plan2 51 10 0.7 0 1 0 1 locomotion 0 walk none -')['frames'] >= 24
        checks += 2
        # The legacy door through v2 keeps its table but only the right arm is constrained;
        # the v1 line still constrains the whole body.
        door_v2 = other.ask('plan2 52 10 0.7 0 1 0 1 door 0 walk arm_r -')
        assert door_v2['mode'] == 'door' and door_v2['masked'] == 8 and door_v2['frames'] == 24
        door_v1 = other.ask('plan 53 10 0.7 0 1 0 1 door 0')
        assert door_v1['mode'] == 'door' and door_v1['masked'] == 0 and door_v1['frames'] == 24
        checks += 2

        # Malformed v2 lines answer an error object and the worker keeps serving.
        malformed = ['plan2 5 0 0.7 0 1 0 1 locomotion 0 walk wings -',
                     'plan2 5 0 0.7 0 1 0 1 locomotion 0 walk arm_r right_elbow_skel=1,0,0',
                     'plan2 5 0 0.7 0 1 0 1 locomotion 0 walk arm_r nosuch=0,0,0,1',
                     'plan2 5 0 0.7 0 1 0 1 locomotion 0 walk arm_r right_elbow_skel=0,0,0,2',
                     'plan2 5 0 0.7 0 1 0 1 locomotion 0 walk arm_r right_elbow_skel=a,b,c,d',
                     'plan2 5 0 0.7 0 1 0 1 locomotion 0 walk',
                     'plan2 5 0 0.7 0 1 0 1 locomotion 0 walk none - extra',
                     'hello extra']
        for line in malformed:
            assert 'error' in other.ask(line), line
        recovered = other.ask('plan2 60 0 0.7 0 1 0 1 locomotion 0 - none -')
        assert recovered['id'] == 60 and recovered['style'] == 'walk'
        checks += 2
        assert plain.close() == 0 and other.close() == 0
        checks += 1
        return {'passed': True, 'checks': checks, 'styles': metadata['styles'],
                'groups': {name: len(joints) for name, joints in hello['groups'].items()},
                'style_speeds': hello['style_speeds'],
                'explicit_speed_knee_mean_abs_difference_deg': knee_difference,
                'style_pace': {'knee_period_frames': periods, 'travel_m_per_s': paces},
                'arm_goal': {'max_leg_difference_deg': leg_difference, 'max_root_difference': root_difference,
                             'shoulder_deg': shoulder_goal, 'unmasked_shoulder_deg': shoulder_plain},
                'malformed_requests': len(malformed),
                'warmed_inference_ms': {style: statistics.mean(values[6:]) for style, values in timings.items()}
                | {'arm_r_goal': statistics.mean(goal_timings[5:])}}
    finally:
        plain.close()
        other.close()


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
    if args.long:
        result = verify_rollout(args.worker, args.bundle)
    else:
        result = {'v1': verify(args.worker, args.bundle), 'v2': verify_v2(args.worker, args.bundle)}
    print(json.dumps(result, indent=2))

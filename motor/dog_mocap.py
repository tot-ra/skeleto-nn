"""Import CC-BY-4.0 Lifelike Agility dog BVH into normalized AMP features.

Raw data are downloaded explicitly and checksum-verified, never redistributed.
Whole recordings are split before features/mirrors are built (no frame leakage).
The physics has four segments per leg; this dog recording has a reduced forepaw.
We use only comparable thigh/shank/back/neck/head/tail directions in both domains.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
from motor.amp import features, mirror, heading_frame

ARTICLE = 'https://api.figshare.com/v2/articles/24968946'
URL = 'https://ndownloader.figshare.com/files/43971117'
SHA256 = '651a4099b4660f898a5bfcd24769ba4142a0ec30daa328a2aebc3d31b51b8b34'
SEGS = [(f'{seg}{e}_{s}', f'{a}{e}_{s}', f'{b}{e}_{s}')
        for s in 'LR' for e in 'HF' for seg, a, b in
        (('thigh', 'hip', 'knee'), ('shank', 'knee', 'hock'))] + [
        ('back', 'hip_mid', 'withers'), ('neck', 'withers', 'poll'),
        ('head', 'poll', 'nose'), ('tail', 'tailbase', 'tailtip')]


def read_bvh(text):
    """Return joint metadata, frame channels and interval; supports arbitrary channel order."""
    hierarchy, motion = text.split('MOTION', 1)
    tokens = re.findall(r'[{}]|[^\s{}]+', hierarchy)
    joints = []
    pos = 0
    channel_count = 0

    def joint(parent, end=False):
        nonlocal pos, channel_count
        kind = tokens[pos]; pos += 1
        if kind == 'End':
            assert tokens[pos] == 'Site'; pos += 1
            name = joints[parent]['name'] + '__end'
        else:
            name = tokens[pos]; pos += 1
        assert tokens[pos] == '{'; pos += 1
        index = len(joints)
        entry = dict(name=name, parent=parent, offset=np.zeros(3), channels=[], start=channel_count)
        joints.append(entry)
        while tokens[pos] != '}':
            key = tokens[pos]
            if key == 'OFFSET':
                entry['offset'] = np.array(list(map(float, tokens[pos+1:pos+4]))); pos += 4
            elif key == 'CHANNELS':
                n = int(tokens[pos+1]); entry['channels'] = tokens[pos+2:pos+2+n]
                entry['start'] = channel_count; channel_count += n; pos += n+2
            elif key in ('JOINT', 'End'):
                joint(index)
            else:
                raise ValueError(f'unknown BVH token {key}')
        pos += 1
    assert tokens[pos] == 'HIERARCHY'; pos += 1
    joint(-1)
    match = re.match(r'\s*Frames:\s*(\d+)\s*Frame Time:\s*([\d.eE+-]+)\s*(.*)', motion, re.S)
    if not match:
        raise ValueError('missing BVH frame header')
    n, dt = int(match[1]), float(match[2])
    vals = np.fromstring(match[3], sep=' ')
    if vals.size != n * channel_count or not np.all(np.isfinite(vals)):
        raise ValueError('invalid BVH channel data')
    return joints, vals.reshape(n, channel_count), dt


def positions(joints, frames):
    """BVH FK in declared intrinsic Euler order. Position channels are local translations."""
    n = len(frames)
    world = np.zeros((n, len(joints), 3)); rotations = np.zeros((n, len(joints), 3, 3))
    eye = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    for i, j in enumerate(joints):
        local = eye.copy(); t = np.broadcast_to(j['offset'], (n, 3)).copy()
        for k, channel in enumerate(j['channels']):
            value = frames[:, j['start']+k]
            axis = 'XYZ'.index(channel[0].upper())
            if channel.lower().endswith('position'):
                t[:, axis] = value
            elif channel.lower().endswith('rotation'):
                angle = np.radians(value); c, s = np.cos(angle), np.sin(angle)
                rot = eye.copy(); a, b = (axis+1)%3, (axis+2)%3
                rot[:, a, a] = c; rot[:, b, b] = c; rot[:, a, b] = -s; rot[:, b, a] = s
                local = local @ rot
            else:
                raise ValueError(f'unknown channel {channel}')
        parent = j['parent']
        if parent < 0:
            rotations[:, i] = local; world[:, i] = t
        else:
            rotations[:, i] = rotations[:, parent] @ local
            world[:, i] = world[:, parent] + np.einsum('nij,nj->ni', rotations[:, parent], t)
    return world


def point_names():
    out = dict(hip_L='b_LeftLegUpper', hip_R='b_RightLegUpper', withers='b_Spine3',
               poll='b_Head', nose='Bip01_HeadNub', tailbase='b_Tail001', tailtip='b_Tail009')
    for s, side in [('L','Left'), ('R','Right')]:
        for e, names in [('H', ('LegUpper','Leg','Leg1')), ('F', ('Arm','ForeArm','Hand'))]:
            for part, name in zip(('hip','knee','hock'), names):
                out[f'{part}{e}_{s}'] = f'b_{side}{name}'
    return out


def clip_features(text, hz=30):
    joints, frames, dt = read_bvh(text)
    step = int(round(1 / (dt * hz)))
    if step < 1 or abs(step * dt - 1/hz) > 1e-5:
        raise ValueError('source rate must be an integer multiple of target rate')
    idx = {j['name']: i for i,j in enumerate(joints)}
    # Source is centimetres, Y-up; cyclic xyz -> zxy is right-handed Z-up.
    P = positions(joints, frames[::step])[:, :, [2,0,1]] * .01
    mapping = point_names()
    mapping['tailtip'] = sorted(n for n in idx if re.fullmatch(r'b_Tail[0-9]+', n))[-1]
    pts = {key: P[:,idx[name]] for key,name in mapping.items()}
    leg = np.median(np.linalg.norm(pts['hipH_L']-pts['kneeH_L'], axis=1) +
                    np.linalg.norm(pts['kneeH_L']-pts['hockH_L'], axis=1))
    if not .15 < leg < 2.:
        raise ValueError(f'implausible dog leg length {leg}')
    mid = .5*(pts['hip_L']+pts['hip_R'])
    yaw = np.unwrap([heading_frame(pts['hip_L'][t].copy(), pts['hip_R'][t].copy())[1] for t in range(len(P))])
    vel = np.gradient(mid, 1/hz, axis=0); rate = np.gradient(yaw, 1/hz)
    F = np.array([features({k:v[t].copy() for k,v in pts.items()}, vel[t], rate[t], leg, SEGS)
                  for t in range(1,len(P)-1)], dtype=np.float32)
    if len(F) < 3 or not np.all(np.isfinite(F)):
        raise ValueError('invalid feature sequence')
    return F, dict(frames=len(frames), sample_frames=len(F), source_hz=1/dt, leg_m=float(leg))


def is_holdout(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8],16) % 5 == 0


def build(archive, out):
    archive, out = Path(archive), Path(out)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError('source archive SHA-256 mismatch')
    data = {'train':[], 'holdout':[]}; clips = []
    with zipfile.ZipFile(archive) as z:
        for name in sorted(z.namelist()):
            if not name.startswith('raw_bvh_data/') or not name.endswith('.bvh'):
                continue
            # Training only level locomotion/idle; jumps, hits and play need their own goals.
            if not any(k in Path(name).name for k in ('walk','run','back','idle')):
                continue
            F, stats = clip_features(z.read(name).decode())
            split = 'holdout' if is_holdout(name) else 'train'
            for f in (F, mirror(F, SEGS)):
                data[split].append(np.concatenate([f[:-1],f[1:]],axis=1))
            clips.append(dict(name=name, split=split, **stats))
    out.mkdir(parents=True, exist_ok=True)
    for split, arrays in data.items():
        if not arrays:
            raise ValueError('empty clip split')
        np.savez_compressed(out/f'dog_{split}_amp.npz',X=np.concatenate(arrays), segments=np.array(SEGS), hz=np.array(30))
    manifest = dict(source=ARTICLE, doi='10.6084/m9.figshare.24968946.v1',
                    authors=['Lei Han', 'Qingxu Zhu', 'Jiapeng Sheng', 'Chong Zhang', 'Tingguang Li', 'Yizheng Zhang', 'He Zhang', 'Yuzhen Liu', 'Cheng Zhou', 'Rui Zhao', 'Jie Li', 'Yufeng Zhang', 'Rui Wang', 'Wanchao Chi', 'Xiong Li', 'Yonghui Zhu', 'Lingzhu Xiang', 'Xiao Teng', 'Zhengyou Zhang'], title='Lifelike Agility and Play in Quadrupedal Robots using Reinforcement Learning and Generative Pre-trained Models',
                    license='CC BY 4.0',license_url='https://creativecommons.org/licenses/by/4.0/',
                    archive_sha256=SHA256, modifications='30 Hz; Z-up metres; normalized segment directions; mirrors; whole-clip split; locomotion/idle only',
                    segments=SEGS, clips=clips,
                    samples={split:sum(len(x) for x in arrays) for split,arrays in data.items()})
    (out/'source.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--download',action='store_true')
    a=ap.parse_args()
    if a.download and not Path(a.archive).exists():
        Path(a.archive).parent.mkdir(parents=True,exist_ok=True)
        urllib.request.urlretrieve(URL,a.archive)
    build(a.archive,a.out)


if __name__=='__main__':main()

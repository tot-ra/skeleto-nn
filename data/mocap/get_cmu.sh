#!/bin/sh
# Walks, runs, turns and stops from the CMU Graphics Lab Motion Capture Database (mocap.cs.cmu.edu): free for any
# use, the data may not be resold. Then: python -m motor.amp data/mocap/cmu  (writes data/mocap/cmu_amp.npz)
cd "$(dirname "$0")" && mkdir -p cmu && cd cmu
get() { s=$1; shift; curl -sS -o $s.asf http://mocap.cs.cmu.edu/subjects/$s/$s.asf
        for t in "$@"; do f=${s}_$(printf %02d $t).amc; [ -s $f ] || curl -sS -o $f http://mocap.cs.cmu.edu/subjects/$s/$f; done; }
get 16 $(seq 11 46); get 35 $(seq 1 26); get 09 $(seq 1 11); get 07 $(seq 1 12); get 08 $(seq 1 11); get 02 1 2 3

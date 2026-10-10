"""Shot-ification: splitting long segments into edit-ready shots."""
import numpy as np
from dronesift import segment as seg
from test_segment import make_feat

def feat_with_shots(n, burst=None, acc=None):
    f = make_feat(n, jitter_burst=burst if burst is not None else np.ones(n),
                  motion_acc=acc if acc is not None else np.full(n, 10.0))
    return f

def one_seg(n, fps=10.0):
    return [dict(start_frame=0, end_frame=n - 1, start=0.0, end=n / fps, duration=n / fps, label='push_in', base='push')]

def test_disabled_by_default():
    f = feat_with_shots(100)
    assert seg.shots(one_seg(100), f, np.array(['push'] * 100), dict(seg.DEFAULT_THR), 10.0) == []

def test_long_segment_splits_under_max():
    fps = 10.0
    f = feat_with_shots(400)                                   # 40 s steady push
    thr = dict(seg.DEFAULT_THR, max_shot_s=8.0, target_shot_s=6.0, min_shot_s=4.0)
    shots = seg.shots(one_seg(400), f, np.array(['push'] * 400), thr, fps)
    assert 5 <= len(shots) <= 7                                # ~40/6
    for s in shots:
        assert s['duration'] <= 8.0 + 1.0                      # at most one merged stub beyond max
    assert abs(sum(s['duration'] for s in shots) - 40.0) < 0.2  # coverage is exact
    assert [s['shot'] for s in shots] == list(range(len(shots)))
    for a, b in zip(shots, shots[1:]):                          # contiguous in time
        assert b['start'] >= a['end'] - 1e-9

def test_cuts_avoid_shake_bursts():
    fps = 10.0
    burst = np.ones(400); burst[237:243] = 1000.0              # jolt at 24 s (a target-length cut point)
    f = feat_with_shots(400, burst=burst)
    thr = dict(seg.DEFAULT_THR, max_shot_s=8.0, target_shot_s=6.0, min_shot_s=4.0)
    shots = seg.shots(one_seg(400), f, np.array(['push'] * 400), thr, fps)
    cuts = [s['end_frame'] for s in shots[:-1]]
    assert all(abs(c - 240) > 5 for c in cuts)                 # no boundary lands inside the burst

def test_label_boundary_preferred():
    fps = 10.0
    f = feat_with_shots(240)
    labels = np.array(['push'] * 60 + ['orbit'] * 180)         # push -> orbit at 6 s
    thr = dict(seg.DEFAULT_THR, max_shot_s=8.0, target_shot_s=6.0, min_shot_s=4.0)
    shots = seg.shots(one_seg(240), f, labels, thr, fps)
    assert shots[0]['end_frame'] == 59 and shots[0]['label'] == 'push'
    assert shots[1]['label'] == 'orbit'

def test_stub_merged_into_previous():
    fps = 10.0
    f = feat_with_shots(150)                                   # 15 s: 6+6 cuts leave a 3 s stub < min_shot_s
    thr = dict(seg.DEFAULT_THR, max_shot_s=8.0, target_shot_s=6.0, min_shot_s=4.0)
    shots = seg.shots(one_seg(150), f, np.array(['push'] * 150), thr, fps)
    assert all(s['duration'] >= 4.0 for s in shots)            # no shot below min_shot_s
    assert abs(sum(s['duration'] for s in shots) - 15.0) < 0.2

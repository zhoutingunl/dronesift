"""Telemetry/flow alignment, keep-mask guards and segment assembly."""
import numpy as np
from dronesift import segment as seg

def tel_rows(pts, **cols):
    n = len(pts)
    return [dict(pts=float(p), **{k: (v[i] if isinstance(v, (list, np.ndarray)) else v) for k, v in cols.items()})
            for i, p in enumerate(pts)]

def flow_rows(n, **cols):
    base = dict(n_tracks=50, dx=0.0, dy=0.0, da=0.0, ds=0.0, inlier_ratio=1.0,
                residual=0.1, lap_var=100.0, mean_luma=128.0, mdx=0.0, mdy=0.0)
    base.update(cols)
    return [dict(base, frame=i) for i in range(n)]

def make_feat(n, **over):
    """Feature dict whose defaults pass every keep_mask threshold."""
    z = np.zeros(n); one = np.ones(n)
    f = dict(speed3=np.full(n, 3.0), speed_h=np.full(n, 3.0), v_fwd=np.full(n, 3.0), v_lat=z, track=z,
             vx=np.full(n, 3.0), vy=z, vz=z, g_yaw=z, g_pitch=np.full(n, -10.0), ac_yaw_rate=z,
             yaw_rate=z, pitch_rate=z, yaw_acc=one, pitch_acc=one, acc_h=z, acc_z=z, acc_tan=z,
             rel_alt=np.full(n, 20.0), iso=np.full(n, 100.0), ev=z, expo_jump=np.full(n, 0.1),
             jitter=one, jitter_rms=one, jitter_burst=one, hf_flip=one, motion_reversal=np.zeros(n, bool),
             motion_px=np.full(n, 100.0), motion_acc=np.full(n, 10.0),
             vmag=np.full(n, 100.0), consist=one, radius=np.full(n, 100.0), blur_drop=one,
             residual=np.full(n, 0.1), n_tracks=np.full(n, 50.0),
             dx_raw=z, dy_raw=z, da_raw=z, ds_raw=z, dx_sm=z, dy_sm=z, da_sm=z, ds_sm=z,
             flow_ok=one.astype(bool), tel_ok=one.astype(bool))
    f.update({k: np.asarray(v) for k, v in over.items()})
    return f

def test_features_resample_telemetry_by_pts():
    """A track starting 0.5 s late holds its bump at telemetry rows 15-20; on the video
    timeline it must stay at 2.0-2.5 s (frames 20-25), not slide to rows 15-20 as index-zipping would."""
    fps = 10
    pts = 0.5 + np.arange(50) / fps                     # 0.5 .. 5.4 s
    vx = np.array([5.0 if 2.0 <= p <= 2.5 else 0.0 for p in pts])
    f, n = seg.features(tel_rows(pts, vx=vx), flow_rows(60), fps=fps)
    assert n == 60
    assert 20 <= f['vx'].argmax() <= 25                 # bump at pts 2.0-2.5 = frames 20-25
    assert not f['tel_ok'][:5].any()                             # frames before the track starts
    assert f['tel_ok'][10:55].all() and not f['tel_ok'][56:].any()  # ...and after it ends (5.4 s)

def test_features_coverage_ends_with_telemetry():
    fps = 10
    pts = np.arange(50) / fps                            # 0 .. 4.9 s, video is 6 s
    f, n = seg.features(tel_rows(pts, vx=1.0), flow_rows(60), fps=fps)
    assert n == 60
    assert f['tel_ok'][:50].all() and not f['tel_ok'][50:].any()

def test_aligned_tel_drops_duplicate_pts():
    pts = [0.0, 0.1, 0.1, 0.2, 0.3, 0.4]
    cols, covered = seg.aligned_tel(tel_rows(pts, vx=1.0), 4, 10.0)
    assert len(cols['vx']) == 4 and np.allclose(cols['vx'], 1.0)
    assert covered.all()

def test_aligned_tel_without_pts_falls_back_to_index():
    rows = [dict(vx=1.0) for _ in range(5)]
    cols, covered = seg.aligned_tel(rows, 8, 10.0)
    assert len(cols['vx']) == 8
    assert np.allclose(cols['vx'][:5], 1.0) and np.allclose(cols['vx'][5:], 1.0)  # last value extended
    assert covered[:5].all() and not covered[5:].any()

def test_runs():
    runs = list(seg._runs(np.array([True, True, False, False, True])))
    assert [(a, b) for a, b, _ in runs] == [(0, 1), (2, 3), (4, 4)]

def test_keep_mask_blocks_frames_without_telemetry():
    f = make_feat(30)
    f['tel_ok'][10:20] = False
    keep = seg.keep_mask(f, dict(seg.DEFAULT_THR))
    assert keep[:10].all() and not keep[10:20].any() and keep[20:].all()

def test_keep_mask_relaxes_yaw_acc_inside_orbits():
    thr = dict(seg.DEFAULT_THR)
    f = make_feat(30, radius=np.full(30, 5.0), yaw_acc=np.full(30, 10.0))   # 10 > yaw_acc 8, < acc_h/r = 13.7 deg/s2
    assert seg.keep_mask(f, thr, np.array(['orbit'] * 30)).all()
    assert not seg.keep_mask(f, thr, np.array(['push'] * 30)).any()

def test_spike_burst_splits_segment():
    """A 0.1-0.3 s shake burst must split the segment; the 0.15 s rms metric catches what
    the 0.5 s median filter dilutes below jitter_px (the manual-dial-twitch failure mode)."""
    fps = 10
    burst = np.ones(200)
    burst[100:106] = 60.0                                  # 0.6 s of 60 px/s, rms stays ~26 < jitter_px 30
    f = make_feat(200, jitter_burst=burst)
    thr = dict(seg.DEFAULT_THR)
    spikes = seg.spike_mask(f, thr, fps)
    assert spikes[98:108].all() and not spikes[:96].any()  # eroded run padded by +-0.35 s, rest untouched
    keep = seg.keep_mask(f, thr)
    segs = seg.segments(f, keep, np.array(['push'] * 200), thr, fps)
    assert len(segs) == 2                                  # the burst breaks the segment for real

def test_spike_ignores_single_frame_glitch():
    """1-2 frame optical-flow outliers reach high burst values but must not split anything."""
    fps = 10
    glitch = np.ones(200); glitch[100] = 400.0             # one bad frame -> 0.15 s rms ~ 100+
    glitch[150:152] = 400.0
    f = make_feat(200, jitter_burst=glitch)
    thr = dict(seg.DEFAULT_THR)
    assert not seg.spike_mask(f, thr, fps).any()
    assert seg.keep_mask(f, thr).all()

def test_coherent_fast_move_is_kept():
    """A violent but never-reversing excursion (whip tilt) reads as an intentional move: no veto."""
    f = make_feat(30, jitter_rms=np.full(30, 120.0), jitter=np.full(30, 120.0), hf_flip=np.zeros(30))
    assert seg.keep_mask(f, dict(seg.DEFAULT_THR)).all()

def test_oscillating_shake_is_dropped():
    """Same amplitude: reversing motion (deviate-and-return) is dropped, coherent motion is kept."""
    thr = dict(seg.DEFAULT_THR)
    shake = make_feat(30, jitter_rms=np.full(30, 40.0), jitter=np.full(30, 40.0), hf_flip=np.full(30, 0.4))
    assert not seg.keep_mask(shake, thr).any()
    smooth = make_feat(30, jitter_rms=np.full(30, 40.0), jitter=np.full(30, 40.0), hf_flip=np.zeros(30))
    assert seg.keep_mask(smooth, thr).all()

def test_coherent_burst_does_not_split():
    """MasterShots-style whip moves trip the burst amplitude but never reverse, so no split."""
    fps = 10
    burst = np.ones(200); burst[100:106] = 200.0
    f = make_feat(200, jitter_burst=burst, hf_flip=np.zeros(200))
    assert not seg.spike_mask(f, dict(seg.DEFAULT_THR), fps).any()

def test_slow_component_reversal_is_dropped():
    """A component changing sign mid-shot (pan right... then left) reads as contradictory motion
    even though the hf_flip gate misses the slow reversal."""
    thr = dict(seg.DEFAULT_THR)
    f = make_feat(30, motion_reversal=np.full(30, True))
    assert not seg.keep_mask(f, thr).any()
    f = make_feat(30, motion_reversal=np.full(30, False))
    assert seg.keep_mask(f, thr).all()

def test_segments_merges_short_gaps_and_trims():
    fps = 10; f = make_feat(100); labels = np.array(['push'] * 100)
    keep = np.ones(100, bool); keep[50:54] = False        # 0.4 s gap < merge_gap_s
    segs = seg.segments(f, keep, labels, dict(seg.DEFAULT_THR), fps)
    assert len(segs) == 1
    s = segs[0]
    assert s['start_frame'] == 3 and s['end_frame'] == 96  # 0.3 s trimmed at both ends
    assert abs(s['duration'] - 9.4) < 1e-9 and s['label'] == 'push_in'

def test_segments_drops_runs_below_min_len():
    fps = 10; f = make_feat(100); labels = np.array(['push'] * 100)
    keep = np.zeros(100, bool); keep[40:60] = True         # 2.0 s < min_len_s 3
    assert seg.segments(f, keep, labels, dict(seg.DEFAULT_THR), fps) == []

def test_segments_labels_move_sequence():
    fps = 10; f = make_feat(100, v_lat=np.full(100, 3.0), v_fwd=np.full(100, 1.0))   # lateral > forward: orbit, not spiral
    labels = np.array(['push'] * 50 + ['orbit'] * 50)
    keep = np.ones(100, bool)
    segs = seg.segments(f, keep, labels, dict(seg.DEFAULT_THR), fps)
    assert len(segs) == 1 and segs[0]['label'] == 'push_in>orbit_CW'
    assert segs[0]['moves'] == ['push_in', 'orbit_CW']

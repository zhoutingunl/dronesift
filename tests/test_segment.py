"""Telemetry/flow alignment, keep-mask guards and segment assembly."""
import numpy as np
import pytest
from dronesift import segment as seg

def tel_rows(pts, **cols):
    n = len(pts)
    return [dict(pts=float(p), **{k: (v[i] if isinstance(v, (list, np.ndarray)) else v) for k, v in cols.items()})
            for i, p in enumerate(pts)]

def flow_rows(n, **cols):
    base = dict(n_tracks=50, dx=0.0, dy=0.0, da=0.0, ds=0.0, inlier_ratio=1.0,
                residual=0.1, lap_var=100.0, mean_luma=128.0, mdx=0.0, mdy=0.0)
    base.update(cols)
    return [dict({k: (v[i] if isinstance(v, (list, np.ndarray)) else v) for k, v in base.items()}, frame=i)
            for i in range(n)]

def synthetic_motion(fps=30, duration=12, **over):
    """Sample camera telemetry and image velocities in seconds, then run the full pipeline."""
    n = int(fps * duration)
    t = np.arange(n) / fps
    values = dict(vx=np.full(n, 3.0), vy=np.zeros(n), vz=np.zeros(n),
                  g_yaw=np.zeros(n), g_pitch=np.full(n, -20.0),
                  g_roll=np.zeros(n), image_vx=np.full(n, 80.0), image_vy=np.zeros(n),
                  image_rotation_dps=np.zeros(n))
    values.update({key: (value(t) if callable(value) else value) for key, value in over.items()})
    tel = tel_rows(t, **{key: values[key] for key in ('vx', 'vy', 'vz', 'g_yaw', 'g_pitch', 'g_roll')},
                   iso=100.0, rel_alt=20.0)
    flow = flow_rows(n, dx=np.asarray(values['image_vx']) / fps, dy=np.asarray(values['image_vy']) / fps,
                     mdx=np.asarray(values['image_vx']) / fps, mdy=np.asarray(values['image_vy']) / fps,
                     da=np.radians(values['image_rotation_dps']) / fps)
    # A decoded first frame has no previous frame from which to estimate flow.
    flow[0].update(n_tracks=0, dx=0.0, dy=0.0, mdx=0.0, mdy=0.0, da=0.0)
    f, _ = seg.features(tel, flow, fps)
    thresholds = dict(seg.DEFAULT_THR)
    keep, labels = seg.classify_frames(f, thresholds, fps)
    return f, keep, seg.segments(f, keep, labels, thresholds, fps)

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

def test_moderate_observed_shake_cannot_be_merged_back():
    fps = 30; n = fps * 12
    jitter = np.ones(n); jitter[180:186] = 35  # below burst threshold, but perceptible oscillation
    flips = np.zeros(n); flips[180:186] = 0.4
    f = make_feat(n, jitter_rms=jitter, hf_flip=flips)
    thr = dict(seg.DEFAULT_THR, disturbance_pad_s=0, merge_gap_s=2, trim_s=0)
    keep = seg.keep_mask(f, thr, fps=fps)
    cuts = seg.segments(f, keep, np.array(['push'] * n), thr, fps)
    assert len(cuts) == 2
    assert cuts[0]['end_frame'] == 179 and cuts[1]['start_frame'] == 186

def test_zero_flip_threshold_explicitly_disables_gate():
    f = make_feat(60, jitter_rms=np.full(60, 40), hf_flip=np.zeros(60))
    assert seg.keep_mask(f, dict(seg.DEFAULT_THR)).all()
    assert not seg.keep_mask(f, dict(seg.DEFAULT_THR, hf_flip=0)).any()

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

def test_slow_component_reversal_requires_explicit_legacy_gate():
    """A smooth reversal is a valid planned move; the v4 sign veto remains opt-in."""
    thr = dict(seg.DEFAULT_THR)
    f = make_feat(30, motion_reversal=np.full(30, True))
    assert seg.keep_mask(f, thr).all()
    assert not seg.keep_mask(f, {**thr, 'motion_rev_amp': 25.0}).any()
    f = make_feat(30, motion_reversal=np.full(30, False))
    assert seg.keep_mask(f, {**thr, 'motion_rev_amp': 25.0}).all()

def test_segments_merges_soft_gaps_and_preserves_source_endpoints():
    fps = 10; f = make_feat(100); labels = np.array(['push'] * 100)
    keep = np.ones(100, bool); keep[50:54] = False        # 0.4 s gap < merge_gap_s
    segs = seg.segments(f, keep, labels, dict(seg.DEFAULT_THR), fps)
    assert len(segs) == 1
    s = segs[0]
    assert s['start_frame'] == 0 and s['end_frame'] == 99
    assert abs(s['duration'] - 10.0) < 1e-9 and s['label'] == 'push_in'

def test_segments_trims_interior_boundaries():
    fps = 10; f = make_feat(100)
    keep = np.zeros(100, bool); keep[10:90] = True
    output = seg.segments(f, keep, np.array(['push'] * 100), dict(seg.DEFAULT_THR), fps)
    assert len(output) == 1
    assert (output[0]['start_frame'], output[0]['end_frame']) == (13, 86)

def test_segments_never_merges_a_short_hard_rejection():
    """A verified disturbance must survive assembly even when shorter than merge_gap_s."""
    fps = 30; n = fps * 12
    hard = np.zeros(n, bool); hard[180:183] = True
    f = make_feat(n, hard_reject=hard)
    output = seg.segments(f, ~hard, np.array(['push'] * n), dict(seg.DEFAULT_THR), fps)
    assert len(output) == 2
    assert output[0]['end_frame'] < 180 and output[1]['start_frame'] > 182
    assert output[0]['start_frame'] == 0 and output[1]['end_frame'] == n - 1

def test_short_uncovered_telemetry_gap_is_never_resurrected():
    fps = 30; n = fps * 12
    covered = np.ones(n, bool); covered[180:183] = False
    f = make_feat(n, tel_ok=covered)
    keep, labels = seg.classify_frames(f, dict(seg.DEFAULT_THR), fps)
    output = seg.segments(f, keep, labels, dict(seg.DEFAULT_THR), fps)
    assert len(output) == 2
    assert f['hard_reject'][180:183].all()
    assert all(not np.any(~covered[s['start_frame']:s['end_frame'] + 1]) for s in output)

@pytest.mark.parametrize('fps', [30, 60])
def test_smooth_fast_tilt_retains_the_complete_source(fps):
    f, keep, output = synthetic_motion(fps, g_pitch=lambda t: -85.0 + 6.0 * t,
                                      image_vy=lambda t: np.full(len(t), 100.0))
    assert f['flow_ok'][0]
    assert np.isfinite(f['jitter_burst']).all()
    assert keep.all()
    assert len(output) == 1
    assert output[0]['start'] == 0.0 and output[0]['end'] == 12.0
    assert output[0]['duration'] == 12.0

@pytest.mark.parametrize('fps', [30, 60])
def test_smooth_high_speed_pan_has_no_endpoint_disturbance(fps):
    f, keep, output = synthetic_motion(fps, g_yaw=lambda t: 20.0 * t)
    assert not any(mask.any() for mask in f['disturbances'].values())
    assert keep.all()
    assert len(output) == 1 and output[0]['duration'] == 12.0

@pytest.mark.parametrize('fps', [30, 60])
def test_smooth_accelerating_pan_is_retained(fps):
    f, keep, output = synthetic_motion(fps, g_yaw=lambda t: 5.0 * t + 4.0 * t ** 2)
    assert f['yaw_rate'][-1] > 90.0
    assert not any(mask.any() for mask in f['disturbances'].values())
    assert keep.all()
    assert len(output) == 1 and output[0]['duration'] == 12.0

@pytest.mark.parametrize('fps', [30, 60])
def test_smooth_u_turn_is_retained(fps):
    f, keep, output = synthetic_motion(fps, image_vx=lambda t: 80.0 * np.tanh((6.0 - t) / 1.5))
    assert f['motion_reversal'].any(), 'The fixture must exercise the old sign-reversal veto.'
    assert f['consist'].min() < seg.DEFAULT_THR['consist']
    assert keep.all()
    assert len(output) == 1 and output[0]['duration'] == 12.0

@pytest.mark.parametrize('fps', [30, 60])
def test_one_way_yaw_correction_is_caught_without_a_flow_flip(fps):
    def yaw(t):
        rate = 35.0 * np.exp(-0.5 * ((t - 6.0) / 0.18) ** 2)
        return np.cumsum(rate) / fps
    f, keep, output = synthetic_motion(fps, g_yaw=yaw)
    assert np.diff(f['g_yaw']).min() >= -1e-12, 'This gesture never reverses the camera.'
    assert not (f['hf_flip'] > seg.DEFAULT_THR['hf_flip']).any()
    assert f['disturbances']['yaw_correction'][6 * fps]
    assert not keep[6 * fps]
    assert len(output) == 2
    assert output[0]['end'] < 6.0 < output[1]['start']

def subtle_yaw_nudge_motion(fps, image_axis='horizontal'):
    """A coherent 2.26 degree camera nudge below the original yaw-correction threshold."""
    t = np.arange(12 * fps) / fps
    rate = 5.0 * np.exp(-0.5 * ((t - 6.0) / 0.18) ** 2)
    angle = np.cumsum(rate) / fps
    motion = dict(g_yaw=angle)
    if image_axis == 'horizontal':
        motion['image_vx'] = 80.0 + 12.0 * rate
    elif image_axis == 'vertical':
        # A fast one-way pitch move can have the same tiny yaw telemetry residual,
        # but its visible image change does not substantiate a horizontal nudge.
        motion.update(g_pitch=-60.0 + 10.0 * angle, image_vy=12.0 * rate)
    return synthetic_motion(fps, **motion)

@pytest.mark.parametrize('fps', [30, 60])
@pytest.mark.parametrize('image_axis', ['horizontal', 'vertical', 'none'])
def test_subtle_yaw_nudge_requires_horizontal_image_agreement(fps, image_axis):
    f, keep, output = subtle_yaw_nudge_motion(fps, image_axis)
    assert 2.0 < f['g_yaw'][-1] - f['g_yaw'][0] < 2.5
    assert f['yaw_correction'].max() < seg.DEFAULT_THR['yaw_correction_dps']
    assert not (f['hf_flip'] > seg.DEFAULT_THR['hf_flip']).any()
    assert not f['disturbances']['yaw_correction'].any()
    if image_axis == 'horizontal':
        assert f['disturbances']['yaw_nudge'][6 * fps]
        assert not keep[6 * fps]
        assert len(output) == 2
        assert output[0]['end'] < 6.0 < output[1]['start']
    else:
        assert not f['disturbances']['yaw_nudge'].any()
        assert keep.all()
        assert len(output) == 1 and output[0]['duration'] == 12.0

def test_subtle_yaw_nudge_boundaries_match_at_30_and_60_fps():
    boundaries = []
    for fps in (30, 60):
        f, _, output = subtle_yaw_nudge_motion(fps)
        rejected = np.flatnonzero(f['disturbances']['yaw_nudge'])
        assert len(output) == 2
        boundaries.append((rejected[0] / fps, (rejected[-1] + 1) / fps,
                           output[0]['end'], output[1]['start']))
    assert np.max(np.abs(np.asarray(boundaries[0]) - boundaries[1])) <= 2.0 / 30

@pytest.mark.parametrize('fps', [30, 60])
def test_left_right_gesture_rejects_slow_shoulders_and_its_quiet_pause(fps):
    t = np.arange(14 * fps) / fps
    rate = np.zeros(len(t))
    # The visibly changing framing starts and settles slowly around the peaks.
    # The operator briefly releases the stick before correcting the other way.
    phases = [(5.0, 5.7, 1.0), (5.7, 6.2, 5.0), (6.2, 6.5, 1.0),
              (6.65, 6.9, -1.0), (6.9, 7.4, -5.0), (7.4, 8.0, -1.0)]
    for start, end, value in phases:
        rate[(t >= start) & (t < end)] = value
    f, keep, output = synthetic_motion(fps, duration=14, g_yaw=np.cumsum(rate) / fps,
                                      image_vx=80.0 + 12.0 * rate)
    shoulders = [round(5.2 * fps), round(7.8 * fps)]
    pause = round(6.567 * fps)
    assert (f['yaw_correction'][shoulders] < seg.DEFAULT_THR['yaw_nudge_dps']).all()
    assert abs(f['yaw_rate_fast'][pause]) < seg.DEFAULT_THR['yaw_nudge_quiet_dps']
    # Disable settling padding to prove the entire gesture itself is excluded,
    # including its short quiet pause, rather than merely covered by handles.
    unpadded = seg.disturbance_masks(f, {**seg.DEFAULT_THR, 'disturbance_pad_s': 0.0}, fps)
    assert unpadded['yaw_nudge'][shoulders + [pause]].all()
    assert f['disturbances']['yaw_nudge'][round(5.1 * fps):round(7.8 * fps) + 1].all()
    assert not keep[shoulders + [pause]].any()
    assert len(output) == 2
    assert output[0]['end'] < 5.0 and output[1]['start'] > 8.0

@pytest.mark.parametrize('fps', [30, 60])
def test_one_nudge_during_a_long_pan_preserves_the_surrounding_trajectory(fps):
    t = np.arange(24 * fps) / fps
    rate = 2.0 + 7.0 * np.exp(-0.5 * ((t - 12.0) / 0.18) ** 2)
    f, keep, output = synthetic_motion(fps, duration=24, g_yaw=np.cumsum(rate) / fps,
                                      image_vx=80.0 + 12.0 * rate)
    assert (f['yaw_rate_fast'] > seg.DEFAULT_THR['yaw_nudge_quiet_dps']).all()
    assert f['disturbances']['yaw_nudge'][12 * fps]
    assert not keep[12 * fps]
    assert keep[:10 * fps].all() and keep[14 * fps:].all()
    assert f['hard_reject'].sum() / fps < 2.0
    assert len(output) == 2
    assert output[0]['start'] == 0.0 and output[1]['end'] == 24.0
    assert sum(s['duration'] for s in output) > 21.0

@pytest.mark.parametrize('fps', [30, 60])
@pytest.mark.parametrize('accelerating', [False, True])
def test_coherent_pan_with_matching_horizontal_image_motion_is_kept(fps, accelerating):
    t = np.arange(12 * fps) / fps
    rate = 5.0 + 8.0 * t if accelerating else np.full(len(t), 20.0)
    f, keep, output = synthetic_motion(fps, g_yaw=np.cumsum(rate) / fps,
                                      image_vx=80.0 + 12.0 * rate)
    assert not any(mask.any() for mask in f['disturbances'].values())
    assert keep.all()
    assert len(output) == 1 and output[0]['duration'] == 12.0

@pytest.mark.parametrize('fps', [30, 60])
def test_sequence_keeps_old_disturbances_and_missing_telemetry_rejected(fps):
    t = np.arange(24 * fps) / fps
    subtle = 5.0 * np.exp(-0.5 * ((t - 6.0) / 0.18) ** 2)
    strong = 35.0 * np.exp(-0.5 * ((t - 12.0) / 0.18) ** 2)
    pitch = -20.0 + 1.2 * np.exp(-0.5 * ((t - 18.0) / 0.25) ** 2)
    f, _, _ = synthetic_motion(fps, duration=24, g_yaw=np.cumsum(subtle + strong) / fps,
                              g_pitch=pitch, image_vx=80.0 + 12.0 * subtle,
                              image_vy=5.0 * np.gradient(pitch) * fps)
    f['tel_ok'][22 * fps:22 * fps + 2] = False
    keep, labels = seg.classify_frames(f, dict(seg.DEFAULT_THR), fps)
    output = seg.segments(f, keep, labels, dict(seg.DEFAULT_THR), fps)
    assert f['disturbances']['yaw_nudge'][6 * fps]
    assert f['disturbances']['yaw_correction'][12 * fps]
    assert f['disturbances']['pitch_return'][18 * fps - fps // 4]
    assert f['hard_reject'][22 * fps:22 * fps + 2].all()
    assert not keep[[6 * fps, 12 * fps, 18 * fps - fps // 4, 22 * fps]].any()
    assert output
    assert all(not f['hard_reject'][s['start_frame']:s['end_frame'] + 1].any() for s in output)

def pitch_return_motion(fps):
    def pitch(t):
        return -20.0 + 1.2 * np.exp(-0.5 * ((t - 6.0) / 0.25) ** 2)
    def image_velocity(t):
        return 5.0 * np.gradient(pitch(t)) * fps
    return synthetic_motion(fps, g_pitch=pitch, image_vy=image_velocity)

@pytest.mark.parametrize('fps', [30, 60])
def test_small_pitch_nudge_and_return_splits_stable_motion(fps):
    f, keep, output = pitch_return_motion(fps)
    assert not (f['hf_flip'] > seg.DEFAULT_THR['hf_flip']).any()
    assert f['disturbances']['pitch_return'][6 * fps - fps // 4]
    assert not keep[6 * fps - fps // 4]
    assert len(output) == 2
    assert output[0]['end'] < 6.0 < output[1]['start']

def test_pitch_return_boundaries_are_equivalent_at_30_and_60_fps():
    events = []
    for fps in (30, 60):
        f, _, output = pitch_return_motion(fps)
        rejected = np.flatnonzero(f['disturbances']['pitch_return'])
        assert len(output) == 2
        events.append((rejected[0] / fps, (rejected[-1] + 1) / fps,
                       output[0]['end'], output[1]['start']))
    assert np.max(np.abs(np.asarray(events[0]) - events[1])) <= 2.0 / 30

@pytest.mark.parametrize('fps', [30, 60])
@pytest.mark.parametrize('corroborated', [False, True])
def test_roll_nudge_requires_agreeing_image_rotation(fps, corroborated):
    def roll(t):
        return 1.5 * np.exp(-0.5 * ((t - 6.0) / 0.25) ** 2)
    def image_rotation(t):
        return np.gradient(roll(t)) * fps if corroborated else np.zeros(len(t))
    f, keep, output = synthetic_motion(fps, g_roll=roll, image_rotation_dps=image_rotation)
    assert f['roll_correction'].max() > seg.DEFAULT_THR['roll_correction_dps']
    if corroborated:
        assert f['disturbances']['roll_correction'].any()
        assert not keep.all()
        assert len(output) == 2
    else:
        assert not f['disturbances']['roll_correction'].any()
        assert keep.all()
        assert len(output) == 1 and output[0]['duration'] == 12.0

@pytest.mark.parametrize('image_slows', [False, True])
def test_camera_brake_requires_image_motion_to_slow(image_slows):
    fps = 30; n = fps * 12; t = np.arange(n) / fps
    progress = np.clip(t - 5.0, 0.0, 1.0)
    yaw_rate = 30.0 - 20.0 * progress
    motion = 200.0 + (-100.0 if image_slows else 100.0) * progress
    f = make_feat(n, yaw_rate=yaw_rate, yaw_acc=np.abs(np.gradient(yaw_rate) * fps),
                  motion_px=motion, motion_acc=np.abs(np.gradient(motion) * fps), hf_flip=np.zeros(n))
    keep, labels = seg.classify_frames(f, dict(seg.DEFAULT_THR), fps)
    output = seg.segments(f, keep, labels, dict(seg.DEFAULT_THR), fps)
    if image_slows:
        assert f['disturbances']['camera_brake'].any()
        assert not keep[int(5.5 * fps)]
        assert len(output) == 2
    else:
        assert not f['disturbances']['camera_brake'].any()
        assert keep.all()
        assert len(output) == 1 and output[0]['duration'] == 12.0

@pytest.mark.parametrize('fps', [30, 60])
@pytest.mark.parametrize('kind', ['telemetry', 'flow'])
def test_single_frame_measurement_outlier_keeps_the_complete_source(fps, kind):
    n = fps * 12
    if kind == 'telemetry':
        yaw = np.zeros(n); yaw[6 * fps] = 20.0
        kwargs = dict(g_yaw=yaw)
    else:
        image_vx = np.full(n, 80.0); image_vx[6 * fps] += 600.0
        kwargs = dict(image_vx=image_vx)
    f, keep, output = synthetic_motion(fps, **kwargs)
    assert not f['hard_reject'].any()
    assert keep.all()
    assert len(output) == 1 and output[0]['duration'] == 12.0

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

"""Window selection must expose idle starts even when average motion is high."""
import numpy as np
from dronesift.shot_quality import motion_window_quality, valid_motion_windows


def feat(motion):
    return {'motion_px': np.asarray(motion, dtype=float)}


def test_static_prefix_is_rejected_despite_high_average():
    f = feat(np.r_[np.ones(10), np.full(90, 40)])
    q = motion_window_quality(f, 10, 0, 10)
    assert not q['ok']
    assert 'still_or_starting_prefix' in q['reasons']


def test_regular_acceleration_is_accepted():
    q = motion_window_quality(feat(np.linspace(12, 40, 100)), 10, 0, 10)
    assert q['ok']


def test_stopped_middle_is_rejected():
    f = feat(np.r_[np.full(35, 20), np.ones(10), np.full(55, 20)])
    assert 'idle_run' in motion_window_quality(f, 10, 0, 10)['reasons']


def test_isolated_measurement_outlier_does_not_make_hover_live():
    f = feat(np.r_[np.ones(3), 500, np.ones(96)])
    assert not motion_window_quality(f, 10, 0, 10)['ok']


def test_windows_start_after_hover_and_fit_inside_segment():
    f = feat(np.r_[np.ones(20), np.full(80, 25)])
    starts = valid_motion_windows(f, 10, 0, 10, 3)
    assert starts
    assert min(starts) >= 1.75
    assert max(starts) + 3 <= 10


def test_orbit_keeps_subject_centred_but_environment_moves():
    f = {k: np.zeros(100) for k in ('dx_raw', 'dy_raw', 'da_raw', 'ds_raw')}
    f['da_raw'][:] = 0.05
    f['motion_px'] = np.zeros(100)
    assert motion_window_quality(f, 10, 0, 10)['ok']


def test_visible_yaw_turnback_is_rejected():
    f = feat(np.full(100, 30))
    f['yaw_rate_fast'] = np.r_[np.full(50, 8), np.full(50, -8)]
    f['dx_sm'] = np.r_[np.full(50, -20), np.full(50, 20)]
    q = motion_window_quality(f, 10, 0, 10)
    assert not q['ok']
    assert 'camera_turns_back' in q['reasons']


def test_tracking_yaw_return_with_one_way_background_is_accepted():
    f = feat(np.full(100, 30))
    f['yaw_rate_fast'] = np.r_[np.full(50, 8), np.full(50, -8)]
    f['dx_sm'] = np.full(100, -20)
    q = motion_window_quality(f, 10, 0, 10)
    assert q['ok']
    assert q['camera_turnback']['yaw_reverses']
    assert not q['camera_turnback']['image_reverses']


def test_small_heading_noise_is_not_a_turnback():
    f = feat(np.full(100, 30))
    f['yaw_rate_fast'] = np.r_[np.full(50, 0.5), np.full(50, -0.5)]
    f['dx_sm'] = np.r_[np.full(50, -20), np.full(50, 20)]
    assert motion_window_quality(f, 10, 0, 10)['ok']


def scale_feat(rate):
    """Scale is signed log-scale per second, as produced by features()."""
    rate = np.asarray(rate, dtype=float)
    return dict(dx_raw=np.full(len(rate), 20.0), dy_raw=np.zeros(len(rate)),
                da_raw=np.zeros(len(rate)), ds_raw=rate)


def test_recorded_pull_then_push_profile_is_rejected_despite_continuous_motion():
    # Quarter-second medians from0028 53.60355–60.95355s: the old magnitude
    # gate accepted this conspicuous pull-back/push-in because both are active.
    bins = [-.327, -.384, -.438, -.492, -.507, -.445, -.338, -.255,
            -.176, -.097, -.019, .060, .119, .162, .217, .280, .350,
            .418, .459, .496, .477, .400, .287, .175, .087, .011,
            -.028, -.037, -.079, -.121]
    f = scale_feat(np.repeat(bins, 5))
    q = motion_window_quality(f, 20, 0, len(f['ds_raw']) / 20)
    assert 'image_scale_reverses' in q['reasons']
    assert q['low_motion_fraction'] == 0
    assert [p['direction'] for p in q['image_scale']['significant_phases'][:2]] == [
        'contraction', 'expansion']


def test_shorter_real_push_pull_push_is_also_rejected():
    # The63.3–70.65 alternative has a smaller but sustained middle retreat.
    bins = [-.032, .016, .108, .198, .280, .340, .396, .454, .456,
            .379, .265, .153, .040, -.059, -.122, -.161, -.153,
            -.110, -.068, -.020, .004, .079, .182, .274, .362,
            .431, .451, .356, .193, .099]
    f = scale_feat(np.repeat(bins, 5))
    q = motion_window_quality(f, 20, 0, len(f['ds_raw']) / 20)
    assert q['image_scale']['reverses']


def test_recorded_single_push_with_tiny_scale_settling_is_accepted():
    #0028 44.6–49.5: an almost imperceptible negative wiggle at the end is
    # under1% scale change, unlike the rejected multi-second retreats.
    bins = [.268, .276, .285, .287, .296, .303, .301, .292, .272, .248,
            .222, .187, .146, .097, .051, .013, -.011, -.002, .020, .035]
    f = scale_feat(np.repeat(bins, 5))
    q = motion_window_quality(f, 20, 0, len(f['ds_raw']) / 20)
    assert q['ok']
    assert q['image_scale']['negative_log_scale'] < .01


def test_unidirectional_constant_or_smoothly_accelerating_scale_is_accepted():
    for rate in (np.full(200, .08), np.linspace(.005, .25, 200),
                 -np.linspace(.005, .25, 200)):
        assert motion_window_quality(scale_feat(rate), 20, 0, 10)['ok']


def test_scale_fit_noise_or_one_frame_outlier_does_not_create_a_retreat():
    noise = np.tile([-.04, .04], 100)
    assert motion_window_quality(scale_feat(noise), 20, 0, 10)['ok']
    rate = np.full(200, .08)
    rate[80] = -.8
    assert motion_window_quality(scale_feat(rate), 20, 0, 10)['ok']


def test_tracking_yaw_with_negligible_image_scale_is_not_push_pull():
    f = scale_feat(np.linspace(-.003, .003, 200))
    f['yaw_rate_fast'] = np.r_[np.full(100, 8), np.full(100, -8)]
    f['dx_sm'] = np.full(200, 20.0)
    f['da_raw'][:] = .05
    q = motion_window_quality(f, 20, 0, 10)
    assert q['ok']
    assert not q['image_scale']['reverses']


def test_scale_reversal_uses_only_the_selected_source_window():
    f = scale_feat(np.r_[np.full(100, -.08), np.full(100, .08)])
    assert not motion_window_quality(f, 20, 0, 10)['ok']
    assert motion_window_quality(f, 20, 0, 4.5)['ok']
    assert motion_window_quality(f, 20, 5.5, 10)['ok']

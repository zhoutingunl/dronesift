"""Window selection must expose idle starts even when average motion is high."""
import numpy as np
import pytest
from dronesift.shot_quality import (motion_window_quality, valid_motion_windows,
                                    playback_speed_quality, playback_speed_profile)


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


def speed_features(fps=30, duration=12, *, wobble=0):
    """Signed source-per-second evidence, including intentional camera yaw."""
    t = np.arange(int(fps * duration)) / fps
    n = len(t)
    return dict(dx_raw=np.full(n, 30.), dy_raw=np.zeros(n),
                da_raw=np.full(n, .01), ds_raw=np.full(n, .003),
                yaw_rate_fast=2 + wobble * np.sin(2 * np.pi * .8 * t),
                pitch_rate_fast=np.zeros(n), roll_rate_fast=np.zeros(n),
                flow_ok=np.ones(n, bool), tel_ok=np.ones(n, bool))


def test_natural_motion_passes_but_irregularity_amplified_at_1_5_fails():
    f = speed_features(wobble=1.2)
    natural = playback_speed_quality(f, 30, 0, 12, 1)
    faster = playback_speed_quality(f, 30, 0, 12, 1.5)
    assert natural['ok']
    assert not faster['ok']
    assert 'angular_jerk_deg_s3_limit' in faster['reasons']
    profile = playback_speed_profile(f, 30, 0, 12)
    assert profile['max_playback_speed'] == 1.25
    assert profile['status'] == 'assessed'
    assert profile['schema_version'] == 1
    assert profile['method'] == 'motion_speed_v1'


@pytest.mark.parametrize('fps', [20, 30, 59.94, 120])
def test_regular_velocity_and_uniform_acceleration_keep_high_cap(fps):
    f = speed_features(fps)
    t = np.arange(len(f['dx_raw'])) / fps
    f['dx_raw'] = 30 + 3 * t
    f['yaw_rate_fast'] = 2 + .2 * t
    end = len(t) / fps
    p = playback_speed_profile(f, fps, 0, end)
    assert p['max_playback_speed'] == 4
    assert p['capped_at_assessment_limit']
    q = p['assessments'][-1]
    assert q['source_metrics']['angular_jerk_deg_s3'] < .01
    assert q['source_metrics']['image_jerk_px_s3'] < .1
    assert q['analysis_fps'] <= 30.001
    assert q['analysis_sample_count'] <= 361
    # A cropped ramp must not acquire fake acceleration/jerk at its ends.
    assert playback_speed_quality(f, fps, 2.1, 6.9, 4)['ok']


def test_output_metrics_scale_in_source_and_output_time_units():
    f = speed_features(wobble=.8)
    q = playback_speed_quality(f, 30, 1, 10, 1.5, output_fps=24)
    original = q['source_metrics']
    output = q['output_metrics']
    for name, power in (('angular_irregular_velocity_deg_s', 1),
                        ('angular_irregular_acceleration_deg_s2', 2),
                        ('angular_jerk_deg_s3', 3)):
        assert output[name] == pytest.approx(original[name] * 1.5 ** power)
    assert q['source_duration_s'] == 9
    assert q['output_duration_s'] == 6
    assert q['rms_output_window_s'] == pytest.approx(q['rms_source_window_s'] / 1.5)
    assert output['image_displacement_px_per_frame'] == pytest.approx(
        original['image_speed_px_s'] * 1.5 / 24)
    assert all(1 <= p['source_start'] < p['source_end'] <= 10
               for p in q['peak_intervals'].values())


def test_output_frame_rate_limits_fast_regular_motion_without_yaw_ban():
    f = speed_features()
    f['dx_raw'][:] = 200
    assert playback_speed_quality(f, 30, 0, 12, 1.5, output_fps=30)['ok']
    q = playback_speed_quality(f, 30, 0, 12, 1.5, output_fps=24)
    assert not q['ok']
    assert q['reasons'] == ['image_displacement_px_per_frame_limit']
    assert playback_speed_quality(f, 30, 0, 12, 1.5, output_fps=60)['ok']


def test_brief_image_correction_is_not_diluted_by_a_long_regular_shot():
    f = speed_features()
    t = np.arange(len(f['dx_raw'])) / 30
    f['dx_raw'] += 18 * np.exp(-((t - 3) / .18) ** 2)
    assert playback_speed_quality(f, 30, 0, 12, 1)['ok']
    q = playback_speed_quality(f, 30, 0, 12, 1.5)
    assert not q['ok']
    assert any(reason.startswith('image_') for reason in q['reasons'])
    assert playback_speed_profile(f, 30, 7, 11)['max_playback_speed'] == 4


def test_local_clean_window_can_have_higher_cap_than_parent():
    f = speed_features(wobble=1.2)
    f['yaw_rate_fast'][150:] = 2
    parent = playback_speed_profile(f, 30, 0, 12)
    local = playback_speed_profile(f, 30, 7, 11)
    assert 0 < parent['max_playback_speed'] <= 1.25
    assert local['max_playback_speed'] == 4
    f['yaw_rate_fast'][:180] = np.nan
    assert playback_speed_profile(f, 30, 7, 11)['max_playback_speed'] == 4


def test_static_smoothness_is_not_evidence_of_activity():
    f = speed_features()
    for key in list(f):
        if key not in ('tel_ok', 'flow_ok'):
            f[key][:] = 0
    q = playback_speed_quality(f, 30, 0, 12, 4)
    assert q['ok']  # Smoothness alone; the independent live-motion gate fails.
    assert q['source_metrics']['image_speed_px_s'] == 0
    assert not motion_window_quality(f, 30, 0, 12)['ok']


def test_smooth_curved_trajectory_does_not_gain_false_acceleration_at_crop_edge():
    f = speed_features(duration=1.2)
    t = np.arange(len(f['dx_raw'])) / 30
    f['dx_raw'] = 30 + 100 * t * t
    f['da_raw'][:] = f['ds_raw'][:] = 0
    q = playback_speed_quality(f, 30, 0, 1.2, 1.5)
    assert q['ok']
    # Real, uniform jerk remains measured. A linear crop-boundary trend used
    # to invent 158 px/s² irregular acceleration and incorrectly veto this.
    assert q['output_metrics']['image_jerk_px_s3'] == pytest.approx(675)
    assert q['output_metrics']['image_irregular_acceleration_px_s2'] < 1e-8


def test_median_translation_avoids_similarity_fit_noise_but_preserves_affine_changes():
    f = speed_features()
    t = np.arange(len(f['dx_raw'])) / 30
    f['dx_raw'] += 70 * np.exp(-((t - 3) / .12) ** 2)
    assert not playback_speed_quality(f, 30, 0, 12, 1)['ok']
    f['mdx_raw'] = np.full(len(t), 30.)
    f['mdy_raw'] = np.zeros(len(t))
    q = playback_speed_quality(f, 30, 0, 12, 4)
    assert q['ok']
    assert q['image_translation_method'] == 'median_feature_displacement'
    # A real short scale change is still a problem even with quiet median XY.
    f['ds_raw'] += .4 * np.exp(-((t - 3) / .12) ** 2)
    assert not playback_speed_quality(f, 30, 0, 12, 1)['ok']


def test_partial_or_nonfinite_median_evidence_cannot_silently_fall_back():
    f = speed_features()
    f['mdx_raw'] = f['dx_raw'].copy()
    assert playback_speed_profile(f, 30, 0, 12)['max_playback_speed'] is None
    f['mdy_raw'] = f['dy_raw'].copy()
    f['mdy_raw'][150] = np.nan
    assert playback_speed_profile(f, 30, 0, 12)['max_playback_speed'] is None


@pytest.mark.parametrize('family,amplitude,units', [
    ('image', 2.0, 'px_s3'), ('angular', .2, 'deg_s3'),
])
def test_invisible_high_frequency_derivatives_do_not_invent_visible_shake(
        family, amplitude, units):
    fps, duration, frequency = 120, 10, 3
    f = speed_features(fps, duration)
    t = np.arange(len(f['dx_raw'])) / fps
    key = 'dx_raw' if family == 'image' else 'yaw_rate_fast'
    f[key] += amplitude * np.sin(2 * np.pi * frequency * t)
    natural = playback_speed_quality(f, fps, 0, duration, 1)
    faster = playback_speed_quality(f, fps, 0, duration, 1.5)
    metric = family + '_jerk_' + units
    assert faster['raw_output_metrics'][metric] > faster['limits'][metric]
    assert natural['ok'] and faster['ok']
    assert faster['output_metrics'][metric] == 0
    visible = faster['visibility'][metric]
    assert visible['raw_peak_displacement_excursion'] < visible['floor']
    # Image velocity 2 px/s at 3 Hz moves only ±0.106 px; time warp cannot
    # change that amplitude. Angular quantisation has the same invariance.
    assert visible['raw_peak_displacement_excursion'] < (
        2 * amplitude / (2 * np.pi * frequency) * 1.2)
    assert natural['visibility'] == faster['visibility']
    assert faster['raw_output_metrics'][metric] == pytest.approx(
        natural['raw_output_metrics'][metric] * 1.5 ** 3)
    assert playback_speed_profile(f, fps, 0, duration)['max_playback_speed'] == 4


def test_invisible_global_peak_cannot_hide_a_smaller_visible_correction():
    fps = 120
    f = speed_features(fps, 10)
    t = np.arange(len(f['dx_raw'])) / fps
    # The first micro-wobble has the largest raw jerk, but under half a pixel
    # excursion. The later correction is visibly larger and still must veto
    # 1.5x even though it is not the global derivative maximum.
    f['dx_raw'] = (30 + 4 * np.sin(2 * np.pi * 3 * t) * (t < 3)
                   + 12 * np.exp(-((t - 7) / .18) ** 2))
    assert playback_speed_quality(f, fps, 0, 10, 1)['ok']
    q = playback_speed_quality(f, fps, 0, 10, 1.5)
    assert not q['ok']
    assert 'image_jerk_px_s3_limit' in q['reasons']
    assert q['raw_peak_intervals']['image_jerk_px_s3']['source_end'] < 4
    assert q['peak_intervals']['image_jerk_px_s3']['source_start'] > 6
    assert q['raw_source_metrics']['image_jerk_px_s3'] > q['source_metrics']['image_jerk_px_s3']


@pytest.mark.parametrize('family', ['image', 'angular'])
def test_visible_other_axis_cannot_back_invisible_high_frequency_noise(family):
    fps = 120
    f = speed_features(fps, 10)
    t = np.arange(len(f['dx_raw'])) / fps
    if family == 'image':
        f['dx_raw'] = 40 + 20 * np.sin(2 * np.pi * .3 * t)
        f['dy_raw'] = 2 * np.sin(2 * np.pi * 3 * t)
        metric = 'image_jerk_px_s3'
    else:
        f['yaw_rate_fast'] = 2 + 5 * np.sin(2 * np.pi * .3 * t)
        f['pitch_rate_fast'] = .2 * np.sin(2 * np.pi * 3 * t)
        metric = 'angular_jerk_deg_s3'
    q = playback_speed_quality(f, fps, 0, 10, 1.5)
    assert q['raw_output_metrics'][metric] > q['limits'][metric]
    assert q['output_metrics'][metric] < q['limits'][metric]
    assert q['ok']


def test_finite_but_overflowing_measurements_are_unknown_not_invisible_zero():
    import json
    f = speed_features()
    f['dx_raw'][:] = 1e200
    p = playback_speed_profile(f, 30, 0, 12)
    assert p['status'] == 'insufficient'
    assert p['max_playback_speed'] is None
    assert all(not q['ok'] for q in p['assessments'])
    json.dumps(p, allow_nan=False)


@pytest.mark.parametrize('mutation', ['missing', 'nan', 'inf', 'outside', 'coverage', 'short'])
def test_incomplete_evidence_never_claims_supported_speed(mutation):
    f = speed_features()
    start, end = 0, 12
    if mutation == 'missing':
        del f['ds_raw']
    elif mutation in ('nan', 'inf'):
        f['dx_raw'][150] = float(mutation)
    elif mutation == 'outside':
        end += .1
    elif mutation == 'coverage':
        f['flow_ok'][150] = False
    else:
        end = .5
    p = playback_speed_profile(f, 30, start, end)
    assert p['max_playback_speed'] is None
    assert p['status'] == 'insufficient'
    assert not p['capped_at_assessment_limit']
    assert all(not q['ok'] for q in p['assessments'])
    assert p['limiting_reasons']


def test_only_large_irregularity_fails_even_the_natural_speed():
    p = playback_speed_profile(speed_features(wobble=5), 30, 0, 12)
    assert p['status'] == 'assessed'
    assert p['max_playback_speed'] == 0


@pytest.mark.parametrize('arguments', [
    (0, 0, 12, 1, 30), (float('nan'), 0, 12, 1, 30),
    (30, -1, 12, 1, 30), (30, 4, 4, 1, 30),
    (30, 0, float('inf'), 1, 30), (30, 0, 12, 0, 30),
    (30, 0, 12, float('nan'), 30), (30, 0, 12, 1, 0),
    (30, 0, 12, 1e308, 30), (30, 0, 12, True, 30),
])
def test_invalid_scalar_speed_arguments_raise(arguments):
    with pytest.raises(ValueError):
        playback_speed_quality(speed_features(), *arguments)


@pytest.mark.parametrize('speeds', [(), (1.5, 2), (1, 2, 1.5), (1, 1), (1, float('inf'))])
def test_profile_requires_increasing_assessment_grid_starting_at_natural_speed(speeds):
    with pytest.raises(ValueError):
        playback_speed_profile(speed_features(), 30, 0, 12, speeds=speeds)


def test_maximum_does_not_jump_across_a_failed_lower_assessment(monkeypatch):
    import dronesift.shot_quality as quality
    original = quality._assess_playback_speed

    def nonmonotonic(*args):
        row = original(*args)
        if row['playback_speed'] == 1.25:
            row.update(ok=False, reasons=['synthetic_sampling_failure'])
        return row

    monkeypatch.setattr(quality, '_assess_playback_speed', nonmonotonic)
    p = quality.playback_speed_profile(speed_features(), 30, 0, 12)
    assert p['assessments'][-1]['ok']
    assert p['max_playback_speed'] == 1
    assert not p['capped_at_assessment_limit']
    assert p['limiting_reasons'] == ['synthetic_sampling_failure']


def test_speed_assessment_does_not_mutate_or_use_activity_cache():
    import json
    f = speed_features(wobble=.8)
    f['_montage_motion_px_s'] = np.zeros(360)
    original = {key: value.copy() for key, value in f.items()}
    first = playback_speed_profile(f, 30, 1, 10)
    playback_speed_quality(f, 30, 1, 10, 2)
    assert playback_speed_profile(f, 30, 1, 10) == first
    assert set(f) == set(original)
    for key in f:
        np.testing.assert_array_equal(f[key], original[key])
    json.dumps(first, allow_nan=False)

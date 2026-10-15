"""Per-window motion gates and visual review for music montage selection.

Segment-level mean motion can hide a still prefix followed by a camera move.
These checks operate on quarter-second bins, including the beginning of every
proposed window. Pixel speeds are measured at the optical-flow width of 640.
Person presence is a review hint: OpenCV's bundled HOG detector does not reliably
detect tiny or overhead people in drone footage. A negative result is not proof
that a shot contains no people. Requiring visible global motion throughout a
shot also makes stationary people acceptable when the environment moves.
"""
from __future__ import annotations

import math
import os
import subprocess
from fractions import Fraction

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d
from scipy.signal import resample_poly, savgol_filter


# Advisory limits at the 640-pixel optical-flow measurement width. These are
# conservative engineering heuristics, not a guarantee of perceptual comfort.
_SPEED_LIMITS = dict(image_irregular_velocity_px_s=18.0,
                     image_irregular_acceleration_px_s2=100.0,
                     image_jerk_px_s3=800.0,
                     angular_irregular_velocity_deg_s=1.8,
                     angular_irregular_acceleration_deg_s2=10.0,
                     angular_jerk_deg_s3=80.0,
                     image_displacement_px_per_frame=12.0,
                     angular_displacement_deg_per_frame=0.8)
_SPEED_VISIBILITY_FLOORS = dict(image=0.5, angular=0.05)


def _finite_parameter(value, name, *, positive=False):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must be finite') from None
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(number):
        raise ValueError(f'{name} must be finite')
    if positive and number <= 0:
        raise ValueError(f'{name} must be positive')
    return number


def _speed_arguments(fps, start, end, output_fps):
    fps = _finite_parameter(fps, 'fps', positive=True)
    output_fps = _finite_parameter(output_fps, 'output_fps', positive=True)
    start = _finite_parameter(start, 'start')
    end = _finite_parameter(end, 'end')
    if start < 0 or end <= start:
        raise ValueError('source window must have 0 <= start < end')
    if not math.isfinite(end * fps):
        raise ValueError('source frame bounds must be finite')
    return fps, start, end, output_fps


def _speed_parameter(value):
    number = _finite_parameter(value, 'playback_speed', positive=True)
    if not math.isfinite(number * number * number):
        raise ValueError('playback_speed is too large for finite derivative scaling')
    return number


def _playback_motion_evidence(feats, fps, start, end):
    """Measure a window once in SOURCE seconds, never from whole-shot means.

    Signed image velocities already have units per second in ``features``.
    Anti-aliased resampling caps analysis at 30 Hz; polynomial edge handling
    preserves constant velocity and linear velocity ramps at crop boundaries.
    A local quadratic velocity trend removes smoothly changing acceleration.
    A linear trend alone can fit the wrong slope at a curved crop boundary and
    invent a large residual acceleration; quadratic interpolation avoids that.
    Actual jerk is measured independently, rather than removed by the trend.
    The worst 0.12-source-second RMS exposes brief corrections without allowing a
    long quiet section to dilute them. No sample is nan_to_num'd into evidence.
    """
    image_keys = ('dx_raw', 'dy_raw', 'da_raw', 'ds_raw')
    angular_keys = ('yaw_rate_fast', 'pitch_rate_fast')
    if not hasattr(feats, 'get'):
        return dict(status='insufficient', reasons=['missing_motion_evidence'])
    median_keys = ('mdx_raw', 'mdy_raw')
    has_median = [feats.get(key) is not None for key in median_keys]
    if any(has_median):
        missing = [key for key, present in zip(median_keys, has_median) if not present]
        if missing:
            return dict(status='insufficient', reasons=['missing_motion_evidence'],
                        missing_features=missing)
        # The fitted similarity's translation can jitter as foreground/background
        # parallax changes its inlier set. Signed median feature displacement is
        # the same robust observable used by the original stability gate. Keep
        # affine rotation AND scale: their real changes must not disappear just
        # because the median image translation is quiet.
        image_keys = median_keys + image_keys[2:]
    required = image_keys + angular_keys
    missing = [key for key in required if feats.get(key) is None]
    if missing:
        return dict(status='insufficient', reasons=['missing_motion_evidence'],
                    missing_features=missing)
    if feats.get('roll_rate_fast') is not None:
        angular_keys += ('roll_rate_fast',)
    keys = image_keys + angular_keys
    a = int(math.ceil(start * fps - 1e-7))
    b = int(math.ceil(end * fps - 1e-7))
    values = []
    for key in keys:
        try:
            array = np.asarray(feats[key], dtype=float)
        except (TypeError, ValueError, OverflowError):
            return dict(status='insufficient', reasons=['invalid_motion_evidence'],
                        invalid_features=[key])
        if array.ndim != 1 or b > len(array):
            return dict(status='insufficient', reasons=['outside_motion_evidence'],
                        invalid_features=[key])
        window = array[a:b]
        if not np.isfinite(window).all():
            return dict(status='insufficient', reasons=['non_finite_motion_evidence'],
                        invalid_features=[key])
        values.append(window)
    for key in ('flow_ok', 'tel_ok'):
        if feats.get(key) is None:
            continue
        coverage = np.asarray(feats[key])
        if coverage.ndim != 1 or b > len(coverage):
            return dict(status='insufficient', reasons=['outside_motion_evidence'],
                        invalid_features=[key])
        selected = coverage[a:b]
        try:
            covered = np.isfinite(selected).all() and np.all(selected == 1)
        except TypeError:
            covered = False
        if not covered:
            return dict(status='insufficient', reasons=['unreliable_motion_evidence'],
                        invalid_features=[key])
    # Flow sample a describes a-1 -> a, partly outside the chosen window.
    # Exclude it rather than borrowing an unplayed correction from before it.
    origin = (a + 1) / fps
    data = np.column_stack(values)[1:]
    if fps < 10 or len(data) < 9 or (len(data) - 1) / fps < 0.75:
        return dict(status='insufficient', reasons=['too_little_temporal_evidence'])
    # Finite input can overflow resampling, derivatives or squared RMS. Such
    # measurements are unknown, never invisible-zero evidence.
    if np.max(np.abs(data)) > np.finfo(float).max ** .25:
        return dict(status='insufficient', reasons=['motion_evidence_numeric_range'])
    ratio = Fraction(min(fps, 30.0) / fps).limit_denominator(10000)
    rate = fps * ratio.numerator / ratio.denominator
    source_count = len(data)
    if ratio.numerator != ratio.denominator:
        data = resample_poly(data, ratio.numerator, ratio.denominator,
                             axis=0, padtype='line')
        # resample_poly rounds up; do not analyse an extrapolated end sample.
        count = int(math.floor((source_count - 1) * rate / fps + 1e-8)) + 1
        data = data[:count]
    # Scale affine rotation/expansion into displacement at the image edge.
    data[:, 2:4] *= 320.0

    def smooth(x, seconds, degree=2, deriv=0):
        size = max(degree + 2, int(round(seconds * rate)))
        size += 1 - size % 2
        size = min(size, len(x) if len(x) % 2 else len(x) - 1)
        return savgol_filter(x, size, degree, deriv=deriv, delta=1 / rate,
                             axis=0, mode='interp')

    fine = smooth(data, 0.20)
    trend = smooth(fine, 0.80, degree=2)
    residual = fine - trend
    irregular_acceleration = smooth(residual, 0.20, deriv=1)
    jerk = smooth(data, 0.30, deriv=2)
    rms_frames = max(1, int(math.ceil(0.12 * rate)))
    # A time warp raises derivatives, but DOES NOT enlarge a wobble's spatial
    # excursion. Integrate signed velocities in source seconds, then remove a
    # local quadratic position trajectory (equivalently integrate its velocity
    # residual). A wider, independent position trend avoids erasing a visible
    # wobble merely because the derivative trend follows part of its curve.
    # Tiny high-frequency fit/quantisation noise
    # must not become a veto just because its second derivative is large.
    displacement = np.zeros_like(fine)
    displacement[1:] = np.cumsum((fine[1:] + fine[:-1]) / (2 * rate), axis=0)
    displacement -= smooth(displacement, 2.0, degree=2)
    support_frames = max(3, int(round(rate)) | 1)
    excursions = (maximum_filter1d(displacement, support_frames, axis=0, mode='nearest')
                  - minimum_filter1d(displacement, support_frames, axis=0, mode='nearest'))
    visible_excursions = dict(image=excursions[:, :4], angular=excursions[:, 4:])
    peaks, raw_peaks, visibility, raw_metrics = {}, {}, {}, {}

    def peak(name, vectors, family):
        power = vectors ** 2
        # Trailing windows avoid a padded edge changing a peak's meaning.
        sums = np.vstack([np.zeros(power.shape[1]), np.cumsum(power, axis=0)])
        component_power = np.maximum(0, (sums[rms_frames:] - sums[:-rms_frames])
                                     / rms_frames)
        rms = np.sqrt(np.sum(component_power, axis=1))
        centres = np.arange(len(rms)) + (rms_frames - 1) // 2
        span = visible_excursions[family][centres]
        floor = _SPEED_VISIBILITY_FLOORS[family]
        supported_components = span >= floor
        supported = np.any(supported_components, axis=1)
        # Match each component: a visible horizontal move cannot give a tiny
        # vertical/scale/roll quantisation wiggle permission to veto the shot.
        effective = np.sqrt(np.sum(component_power * supported_components, axis=1))
        raw_i = int(np.argmax(rms))
        # Search ALL derivative windows: an invisible global maximum must not
        # hide a lower, visible correction elsewhere in the consumed interval.
        i = int(np.argmax(effective)) if supported.any() else raw_i
        raw_metrics[name] = float(rms[raw_i])
        raw_peaks[name] = dict(value=float(rms[raw_i]),
                              source_start=origin + raw_i / rate,
                              source_end=min(end, origin + (raw_i + rms_frames) / rate))
        peaks[name] = dict(value=float(effective[i]),
                           source_start=origin + i / rate,
                           source_end=min(end, origin + (i + rms_frames) / rate),
                           visibility_supported=bool(supported[i]))
        visibility[name] = dict(floor=floor,
                                gating='same_component_local_displacement',
                                unit='px_at_640_width' if family == 'image' else 'deg',
                                component_names=(['horizontal', 'vertical', 'edge_rotation', 'edge_scale']
                                                 if family == 'image' else list(angular_keys)),
                                raw_peak_displacement_excursion=float(np.linalg.norm(span[raw_i])),
                                supported_peak_displacement_excursion=float(np.linalg.norm(
                                    span[i] * supported_components[i])),
                                raw_peak_component_excursions=span[raw_i].tolist(),
                                supported_peak_component_excursions=span[i].tolist(),
                                supported_components=supported_components[i].tolist(),
                                source_support_window_s=support_frames / rate,
                                position_detrend_source_window_s=2.0,
                                position_detrend_polynomial_degree=2,
                                displacement_invariant_under_time_warp=True)
        return peaks[name]['value']

    metrics = {}
    for family, sl, unit in (('image', slice(0, 4), 'px'),
                             ('angular', slice(4, None), 'deg')):
        metrics[f'{family}_irregular_velocity_{unit}_s'] = peak(
            f'{family}_irregular_velocity_{unit}_s', residual[:, sl], family)
        metrics[f'{family}_irregular_acceleration_{unit}_s2'] = peak(
            f'{family}_irregular_acceleration_{unit}_s2',
            irregular_acceleration[:, sl], family)
        metrics[f'{family}_jerk_{unit}_s3'] = peak(
            f'{family}_jerk_{unit}_s3', jerk[:, sl], family)
    # Total motion is only a sampling limit, never a blanket rotation veto.
    # Triangle inequality accounts for image translation + edge rotation/scale.
    image_speed = (np.linalg.norm(fine[:, :2], axis=1)
                   + np.abs(fine[:, 2]) + np.abs(fine[:, 3]))
    angular_speed = np.linalg.norm(fine[:, 4:], axis=1)
    metrics['image_speed_px_s'] = float(np.max(image_speed))
    metrics['angular_speed_deg_s'] = float(np.max(angular_speed))
    raw_metrics.update(image_speed_px_s=metrics['image_speed_px_s'],
                       angular_speed_deg_s=metrics['angular_speed_deg_s'])
    return dict(status='assessed', reasons=[], source_metrics=metrics,
                raw_source_metrics=raw_metrics, raw_peak_intervals=raw_peaks,
                visibility=visibility, peak_intervals=peaks, analysis_fps=rate,
                source_sample_count=source_count, analysis_sample_count=len(data),
                rms_source_window_s=rms_frames / rate,
                fine_source_window_s=0.20, trend_source_window_s=0.80,
                trend_polynomial_degree=2,
                image_translation_method=('median_feature_displacement'
                                          if all(has_median)
                                          else 'similarity_center_displacement'))


def _assess_playback_speed(evidence, fps, start, end, speed, output_fps):
    result = dict(ok=False, reasons=list(evidence['reasons']),
                  status=evidence['status'], playback_speed=speed,
                  output_fps=output_fps, source_fps=fps, source_start=start,
                  source_end=end, source_duration_s=end - start,
                  output_duration_s=(end - start) / speed,
                  method='motion_speed_v1', schema_version=1)
    if evidence['status'] != 'assessed':
        for key in ('missing_features', 'invalid_features'):
            if key in evidence:
                result[key] = evidence[key]
        return result
    source = evidence['source_metrics']
    output = {}
    for key, value in source.items():
        exponent = 3 if 'jerk_' in key else 2 if 'acceleration_' in key else 1
        output[key] = value * speed ** exponent
    output['image_displacement_px_per_frame'] = source['image_speed_px_s'] * speed / output_fps
    output['angular_displacement_deg_per_frame'] = source['angular_speed_deg_s'] * speed / output_fps
    raw_output = {key: value * speed ** (
        3 if 'jerk_' in key else 2 if 'acceleration_' in key else 1)
        for key, value in evidence['raw_source_metrics'].items()}
    if not all(math.isfinite(v) for v in (*output.values(), *raw_output.values())):
        result.update(status='insufficient', reasons=['non_finite_scaled_motion_evidence'])
        return result
    reasons = [key + '_limit' for key, limit in _SPEED_LIMITS.items()
               if output[key] > limit * (1 + 1e-9)]
    result.update(ok=not reasons, reasons=reasons,
                  source_metrics=source, output_metrics=output,
                  raw_source_metrics=evidence['raw_source_metrics'],
                  raw_output_metrics=raw_output,
                  visibility=evidence['visibility'],
                  raw_peak_intervals=evidence['raw_peak_intervals'],
                  limits=dict(_SPEED_LIMITS),
                  peak_intervals=evidence['peak_intervals'],
                  analysis_fps=evidence['analysis_fps'],
                  source_sample_count=evidence['source_sample_count'],
                  analysis_sample_count=evidence['analysis_sample_count'],
                  rms_source_window_s=evidence['rms_source_window_s'],
                  rms_output_window_s=evidence['rms_source_window_s'] / speed,
                  trend_polynomial_degree=evidence['trend_polynomial_degree'],
                  image_translation_method=evidence['image_translation_method'],
                  heuristic=True)
    return result


def playback_speed_quality(feats, fps, start, end, playback_speed, output_fps=30):
    """Advisory smoothness of this EXACT consumed source window at a speed.

    Velocity, irregular acceleration and jerk scale by k, k² and k³, but only
    constrain playback with same-family local visible displacement support.
    Displacement excursion is time-warp invariant. ``source_metrics`` and
    ``output_metrics`` expose visibility-supported peaks; ``raw_*_metrics``
    retain unmasked derivatives, including invisible high-frequency noise.
    Uniform
    velocity/acceleration survive detrending; fast regular motion can still
    exceed the per-output-frame sampling limit. This does NOT check activity,
    driving semantics, turn-backs or source stability: callers must AND their
    existing gates. Missing/non-finite/out-of-cache evidence returns ``ok=False``
    and ``status='insufficient'``; invalid scalar arguments raise ValueError.
    """
    fps, start, end, output_fps = _speed_arguments(fps, start, end, output_fps)
    speed = _speed_parameter(playback_speed)
    evidence = _playback_motion_evidence(feats, fps, start, end)
    return _assess_playback_speed(evidence, fps, start, end, speed, output_fps)


def playback_speed_profile(feats, fps, start, end, output_fps=30,
                           speeds=(1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4)):
    """Largest consecutively passing tested speed, starting with natural 1×.

    ``None`` means insufficient evidence; zero means 1× failed. A final passing
    4× means only that the assessed limit was reached, never unlimited speed.
    The profile is bound to its source window, output FPS and method version.
    """
    fps, start, end, output_fps = _speed_arguments(fps, start, end, output_fps)
    try:
        speeds = [_speed_parameter(s) for s in speeds]
    except TypeError:
        raise ValueError('speeds must be an increasing sequence starting at 1') from None
    if not speeds or speeds[0] != 1 or any(b <= a for a, b in zip(speeds, speeds[1:])):
        raise ValueError('speeds must be an increasing sequence starting at 1')
    evidence = _playback_motion_evidence(feats, fps, start, end)
    assessments = [_assess_playback_speed(evidence, fps, start, end, speed, output_fps)
                   for speed in speeds]
    maximum = None
    reasons = list(evidence['reasons'])
    insufficient = next((q for q in assessments if q['status'] == 'insufficient'), None)
    status = evidence['status']
    if insufficient is not None:
        status = 'insufficient'
        reasons = insufficient['reasons']
    elif status == 'assessed':
        maximum = 0.0
        for assessment in assessments:
            if not assessment['ok']:
                reasons = assessment['reasons']
                break
            maximum = assessment['playback_speed']
    return dict(schema_version=1, method='motion_speed_v1', source_start=start,
                source_end=end, output_fps=output_fps,
                max_playback_speed=maximum, assessed_speeds=speeds,
                assessments=assessments, limiting_reasons=reasons,
                capped_at_assessment_limit=bool(maximum is not None
                                               and maximum == speeds[-1]),
                status=status, heuristic=True)


def motion_series(feats, fps):
    """Responsive image motion, avoiding the segmentation 1-second smoothing.

    Translation at image centre plus rotation/scale at a 320-pixel radius gives
    movement across the frame even when an orbit/zoom keeps its subject centred.
    Robust quarter-second medians suppress isolated optical-flow outliers.
    """
    cached = feats.get('_montage_motion_px_s')
    if cached is not None:
        return np.asarray(cached, dtype=float)
    keys = ('dx_raw', 'dy_raw', 'da_raw', 'ds_raw')
    if all(k in feats for k in keys):
        x, y, a, s = (np.asarray(feats[k], dtype=float) for k in keys)
        motion = np.hypot(x, y) + 320 * (np.abs(a) + np.abs(s))
    else:
        motion = np.asarray(feats['motion_px'], dtype=float).copy()
    motion = np.nan_to_num(motion, nan=0.0, posinf=0.0, neginf=0.0)
    # Frame zero has no flow predecessor. The first actual motion sample is the
    # appropriate substitute; otherwise a smooth source would gain a false stop.
    if len(motion) > 1 and motion[0] == 0:
        motion[0] = motion[1]
    motion = gaussian_filter1d(motion, max(0.5, 0.06 * fps), mode='nearest')
    feats['_montage_motion_px_s'] = motion
    return motion


def motion_window_quality(feats, fps, start, end, *, floor_px_s=8.0,
                          bin_s=0.25, start_s=0.5, max_idle_s=0.5,
                          max_idle_fraction=0.10, turnback_deg=3.0,
                          turnback_px=10.0):
    """Return JSON-safe quality metrics for one original-source time window.

    A shot must already be moving in its first half second. The rest of the
    window may not contain a half-second idle run or mostly low-motion frames.
    The absolute floor deliberately excludes distant landscape hovers even if
    small amounts of sensor/vegetation motion are detected.
    """
    motion = motion_series(feats, fps)
    a = max(0, int(math.ceil(float(start) * fps - 1e-6)))
    b = min(len(motion), int(math.ceil(float(end) * fps - 1e-6)))
    reasons = []
    if end <= start or b <= a:
        return dict(ok=False, reasons=['empty_window'], start_motion_px_s=0.0,
                    low_motion_fraction=1.0, longest_idle_s=0.0,
                    motion_median_px_s=0.0, bin_motion_px_s=[])
    size = max(1, int(round(bin_s * fps)))
    ranges = [(i, min(i + size, b)) for i in range(a, b, size)]
    # Do not let a final fragment shorter than half a bin decide acceptance.
    if len(ranges) > 1 and ranges[-1][1] - ranges[-1][0] < size / 2:
        ranges.pop()
    bins = np.array([float(np.median(motion[i:j])) for i, j in ranges])
    lengths = np.array([(j - i) / fps for i, j in ranges])
    idle = bins < floor_px_s
    run = longest = 0.0
    for is_idle, length in zip(idle, lengths):
        run = run + float(length) if is_idle else 0.0
        longest = max(longest, run)
    prefix_n = min(len(bins), max(1, int(math.ceil(start_s / bin_s))))
    prefix = float(np.min(bins[:prefix_n]))
    fraction = float(lengths[idle].sum() / max(lengths.sum(), 1e-6))
    median = float(np.median(bins))
    if prefix < floor_px_s:
        reasons.append('still_or_starting_prefix')
    if longest >= max_idle_s - 1 / fps:
        reasons.append('idle_run')
    if fraction > max_idle_fraction:
        reasons.append('too_much_idle')
    if b < float(end) * fps - 1:
        reasons.append('outside_motion_cache')
    turnback = camera_turnback_quality(feats, fps, start, end,
                                      min_degrees=turnback_deg,
                                      min_pixels=turnback_px)
    if turnback['turns_back']:
        reasons.append('camera_turns_back')
    scale = image_scale_reversal_quality(feats, fps, start, end)
    if scale['reverses']:
        reasons.append('image_scale_reverses')
    return dict(ok=not reasons, reasons=reasons,
                start_motion_px_s=round(prefix, 3),
                low_motion_fraction=round(fraction, 4),
                longest_idle_s=round(longest, 3),
                motion_median_px_s=round(median, 3),
                floor_px_s=float(floor_px_s),
                camera_turnback=turnback,
                image_scale=scale,
                bin_motion_px_s=np.round(bins, 3).tolist())


def image_scale_reversal_quality(feats, fps, start, end, *, min_rate=0.02,
                                 min_run_s=0.45, min_log_change=0.06,
                                 max_gap_s=1.0):
    """Reject a substantial image contraction followed by expansion, or back.

    ``ds_raw`` is signed log image scale per second, already converted by
    ``segment.features``. The motion magnitude discards this sign, so it cannot
    expose a pull-back immediately followed by a push-in. Smooth the signed
    rate first, then require sustained runs with about6% scale change on BOTH
    sides. This ignores tiny fit fluctuations and preserves single-direction
    acceleration. Yaw/roll/translation are not zoom evidence and are not used.

    These are image-field scale changes, not lens zoom or vehicle world speed.
    The check affects montage windows, leaving the stable-cut catalog intact.
    """
    if 'ds_raw' not in feats:
        return dict(reverses=False, available=False)
    cached = feats.get('_montage_signed_scale_rate')
    if cached is None or cached[0] != fps:
        raw = np.nan_to_num(np.asarray(feats['ds_raw'], dtype=float),
                            nan=0.0, posinf=0.0, neginf=0.0)
        rate = gaussian_filter1d(raw, max(0.5, 0.12 * fps), mode='nearest')
        cached = (fps, rate)
        feats['_montage_signed_scale_rate'] = cached
    rate = cached[1]
    a = max(0, int(math.ceil(float(start) * fps - 1e-6)))
    b = min(len(rate), int(math.ceil(float(end) * fps - 1e-6)))
    if b <= a:
        return dict(reverses=False, available=True, significant_phases=[])
    window = rate[a:b]
    signs = np.where(window > min_rate, 1, np.where(window < -min_rate, -1, 0))
    boundaries = np.r_[0, np.flatnonzero(np.diff(signs)) + 1, len(signs)]
    phases = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        sign = int(signs[left])
        duration = (right - left) / fps
        change = float(sign * np.sum(window[left:right]) / fps)
        if not sign or duration < min_run_s or change < min_log_change:
            continue
        phases.append(dict(direction='expansion' if sign > 0 else 'contraction',
                           start=(a + left) / fps, end=(a + right) / fps,
                           duration_s=duration, log_scale_change=change))
    reverses = any(left['direction'] != right['direction']
                   and right['start'] - left['end'] <= max_gap_s
                   for left, right in zip(phases[:-1], phases[1:]))
    return dict(reverses=bool(reverses), available=True,
                positive_log_scale=round(float(np.maximum(window, 0).sum() / fps), 5),
                negative_log_scale=round(float(np.maximum(-window, 0).sum() / fps), 5),
                significant_phases=[{key: round(value, 5) if isinstance(value, float) else value
                                     for key, value in phase.items()} for phase in phases],
                min_rate_per_s=float(min_rate), min_run_s=float(min_run_s),
                min_log_change=float(min_log_change), max_gap_s=float(max_gap_s))


def camera_turnback_quality(feats, fps, start, end, *, min_rate_dps=2.0,
                            min_run_s=0.35, min_degrees=3.0,
                            min_pixels=10.0, min_image_speed=5.0):
    """A yaw return must also reverse the visible horizontal environment.

    A moving-person tracking shot may reverse camera yaw while its environment
    keeps drifting smoothly in one direction. Requiring corroborating image
    motion preserves that natural tracking behaviour. This montage-only check
    does not remove intentional direction changes from the stable cut catalog.
    """
    yaw = feats.get('yaw_rate_fast', feats.get('yaw_rate'))
    image = feats.get('dx_sm')
    if yaw is None or image is None:
        return dict(turns_back=False, available=False)
    n = min(len(yaw), len(image))
    a = max(0, int(math.ceil(start * fps - 1e-6)))
    b = min(n, int(math.ceil(end * fps - 1e-6)))
    yaw = np.nan_to_num(np.asarray(yaw[a:b], dtype=float))
    image = np.nan_to_num(np.asarray(image[a:b], dtype=float))

    def excursions(v, floor):
        result = []
        for sign in (1, -1):
            active = sign * v > floor
            longest = run = 0
            for on in active:
                run = run + 1 if on else 0
                longest = max(longest, run)
            distance = float(np.sum(sign * v[active]) / fps)
            result.append((distance, longest / fps))
        return result

    yaw_excursions = excursions(yaw, min_rate_dps)
    image_excursions = excursions(image, min_image_speed)
    turn = all(angle >= min_degrees and run >= min_run_s
               for angle, run in yaw_excursions)
    corroborated = all(pixels >= min_pixels and run >= min_run_s
                       for pixels, run in image_excursions)
    return dict(turns_back=bool(turn and corroborated), available=True,
                yaw_positive_deg=round(yaw_excursions[0][0], 3),
                yaw_negative_deg=round(yaw_excursions[1][0], 3),
                image_positive_px=round(image_excursions[0][0], 3),
                image_negative_px=round(image_excursions[1][0], 3),
                yaw_reverses=bool(turn), image_reverses=bool(corroborated))


def valid_motion_windows(feats, fps, start, end, duration, step_s=0.25, **kwargs):
    """Accepted source start times, frame aligned, for a desired music slot."""
    n = len(motion_series(feats, fps))
    a = max(0, int(math.ceil(start * fps - 1e-6)))
    last = min(int(math.floor((end - duration) * fps + 1e-6)),
               n - int(math.ceil(duration * fps)))
    if last < a:
        return []
    step = max(1, int(round(step_s * fps)))
    idx = list(range(a, last + 1, step))
    if idx[-1] != last:
        idx.append(last)
    return [i / fps for i in idx
            if motion_window_quality(feats, fps, i / fps,
                                     i / fps + duration, **kwargs)['ok']]


def read_frame(video, time_s, width=960):
    """Decode a source frame at a timestamp without opening a GUI."""
    cmd = ['ffmpeg', '-v', 'error', '-ss', f'{max(0, time_s):.6f}',
           '-i', os.fspath(video), '-frames:v', '1',
           '-vf', f'scale={int(width)}:-2', '-f', 'image2pipe',
           '-vcodec', 'mjpeg', '-']
    data = subprocess.run(cmd, capture_output=True, check=True).stdout
    fr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if fr is None:
        raise ValueError(f'cannot decode {video} at {time_s}s')
    return fr


def person_hints(frame, min_confidence=0.6):
    """Bundled HOG hints for normal-size upright people; tiny people need review."""
    if frame.shape[0] < 128 or frame.shape[1] < 64:
        return []
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8),
                                         padding=(8, 8), scale=1.05)
    return [dict(box=[int(v) for v in box], confidence=round(float(weight), 3))
            for box, weight in zip(boxes, weights)
            if float(weight) >= min_confidence]


def contact_sheet(timeline, out_path, width=480, frames_per_shot=3):
    """Contact sheet of actual shot starts/middles/ends for human visual audit."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Menlo.ttc', 15)
    except OSError:
        font = ImageFont.load_default()
    height = round(width * 9 / 16)
    header = 52
    page = Image.new('RGB', (width * frames_per_shot,
                             (height + header) * len(timeline)), '#151a21')
    draw = ImageDraw.Draw(page)
    for i, shot in enumerate(timeline):
        start, end = shot['start'], shot['end']
        eps = min(0.15, (end - start) / 6)
        times = np.linspace(start + eps, end - eps, frames_per_shot)
        if shot.get('reversed'):
            times = times[::-1]
        for j, t in enumerate(times):
            frame = read_frame(shot['video'], float(t), width=width)
            im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((width, height))
            x, y = j * width, i * (height + header)
            label = os.path.basename(shot['video'])
            seq = int(shot.get('seq', i + 1))
            draw.text((x + 8, y + 4), f"{seq:02d} {label} {t:.2f}s", fill='white', font=font)
            move = str(shot.get('label', shot.get('base', '')))
            max_chars = max(12, int((width - 110) / 9))
            if len(move) > max_chars:
                move = move[:max_chars - 3] + '...'
            draw.text((x + 8, y + 25), f"{move} / {end-start:.2f}s", fill='#b6cfe4', font=font)
            page.paste(im, (x, y + header))
    page.save(out_path)
    return os.fspath(out_path)

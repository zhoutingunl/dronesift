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

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


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

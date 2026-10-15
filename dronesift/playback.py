"""Enforce window-scoped, output-frame-rate-specific playback assessments."""
from __future__ import annotations

import math
from numbers import Real


def _number(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def validate_playback_speed(row, *, source_start, source_end, output_fps,
                            require_assessment=False):
    """Reject unsupported acceleration and assessments for a different window.

    Cropped windows need their own acceleration assessment: local trends and
    edge behaviour can differ from those of a containing catalog segment.
    Legacy unassessed natural-speed edits remain usable; accelerated edits must
    be assessed before rendering. This checks measurements, not subject-motion
    reviews or the semantic safety of reverse playback.
    """
    speed = row.get('playback_speed', 1.0)
    if not _number(speed) or speed <= 0:
        raise ValueError('playback_speed must be a finite positive number')
    declared = row.get('max_playback_speed')
    if declared is not None:
        if not _number(declared) or declared < 0:
            raise ValueError('max_playback_speed must be a finite nonnegative number or null')
        if speed > declared + 1e-7:
            raise ValueError(f'playback_speed {speed:g}x exceeds max_playback_speed {declared:g}x')
    profile = row.get('playback_speed_profile')
    if profile is None:
        if require_assessment and speed > 1:
            raise ValueError('accelerated playback requires a playback_speed_profile for the source window')
        return dict(status='unassessed', playback_speed=float(speed), max_playback_speed=declared)
    if (not isinstance(profile, dict) or type(profile.get('schema_version')) is not int
            or profile['schema_version'] != 1 or profile.get('method') != 'motion_speed_v1'):
        raise ValueError('unsupported playback_speed_profile; reassess the source window')
    if not _number(profile.get('output_fps')) or abs(profile['output_fps'] - output_fps) > 1e-7:
        raise ValueError('playback_speed_profile output_fps differs; reassess at the output frame rate')
    a, b = profile.get('source_start'), profile.get('source_end')
    if (not _number(a) or not _number(b) or a < 0 or b <= a
            or a > source_start + 1e-7 or b < source_end - 1e-7):
        raise ValueError('playback_speed_profile does not cover the actual consumed source window')
    exact = abs(a - source_start) <= 1e-7 and abs(b - source_end) <= 1e-7
    if speed > 1 and not exact:
        raise ValueError('accelerated playback requires a profile for the exact consumed source window')
    maximum = profile.get('max_playback_speed')
    if maximum is None:
        if declared is not None or speed > 1:
            raise ValueError('playback speed is unassessed; insufficient motion evidence for acceleration')
        return dict(status='unassessed', playback_speed=float(speed), max_playback_speed=None)
    if not _number(maximum) or maximum < 0:
        raise ValueError('invalid maximum in playback_speed_profile')
    if profile.get('status') != 'assessed':
        raise ValueError('playback_speed_profile is not an assessed motion profile')
    if declared is not None and declared > maximum + 1e-7:
        raise ValueError('declared max_playback_speed exceeds the assessed maximum')
    if speed > maximum + 1e-7:
        raise ValueError(f'playback_speed {speed:g}x exceeds assessed maximum {maximum:g}x')
    return dict(status='assessed' if exact else 'catalog_advisory',
                playback_speed=float(speed), max_playback_speed=float(maximum),
                method=profile['method'], output_fps=float(output_fps))

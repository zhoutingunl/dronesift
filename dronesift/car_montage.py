"""Reviewed driving windows and chronological transitions for montage plans.

These helpers consume cached motion measurements and explicit visual-review
scopes. Vehicle movement comes from those reviews, not inferred drone yaw.
"""
from __future__ import annotations

import copy
from datetime import timedelta
import math
from pathlib import Path

import numpy as np

from . import montage
from .shot_quality import motion_window_quality


def capture_bounds(row):
    epoch = montage.capture_datetime(row['video']).timestamp()
    return epoch + row['source_start'], epoch + row['source_end']

def enclosed(start, end, note):
    return note['start'] <= start + 1e-7 and note['end'] >= end - 1e-7

def allowed_transition(previous, row, max_scene_run=2):
    if row.get('reversed'):
        return False
    if previous:
        if capture_bounds(row)[0] < capture_bounds(previous[-1])[1] - 1e-7:
            return False
        if row['rotation'] and previous[-1]['rotation']:
            return False
    run = 1
    for old in reversed(previous):
        if old['scene_group'] != row['scene_group']:
            break
        run += 1
    return run <= max_scene_run

def straight_reviews(doc):
    result = []
    for shot in doc['shots']:
        if not shot.get('accelerate'):
            continue
        result.extend(shot.get('straight_windows', []))
        for piece in shot.get('split', {}).get('pieces', []):
            result.extend(piece.get('straight_windows', []))
    return result

def checked_row(candidate, slot, start, note, speed, straight, pan_reviews):
    duration = (slot['end_frame'] - slot['start_frame']) / 30
    end = start + duration * speed
    if not enclosed(start, end, candidate.seg) or not enclosed(start, end, note):
        return None
    quality = motion_window_quality(candidate.feats, candidate.fps, start, end)
    if not quality['ok']:
        return None
    a, b = round(start * candidate.fps), math.ceil(end * candidate.fps)
    sl, feats = slice(a, b), candidate.feats
    yaw_mean = float(np.mean(np.abs(feats['yaw_rate'][sl])))
    yaw_span = abs(float(feats['g_yaw'][b - 1] - feats['g_yaw'][a]))
    raw_rotation = yaw_mean > 1.5 and yaw_span > 8
    speed_notes = [n for n in straight if Path(n['video']).name == Path(candidate.video).name
                   and enclosed(start, end, n)]
    pan_notes = [n for n in pan_reviews if Path(n['video']).name == Path(candidate.video).name
                 and enclosed(start, end, n) and n.get('allow_regular_tracking_pan')
                 and yaw_mean <= min(float(n.get('max_yaw_mean_abs_dps', 2)), 3)]
    if speed > 1 and (not note.get('vehicle_moving') or not speed_notes):
        return None
    pan = bool(pan_notes) and yaw_span <= 20
    if speed > 1 and raw_rotation and not pan:
        return None
    rotation = raw_rotation and not pan
    horizontal_speed = float(np.median(feats['speed_h'][sl]))
    if rotation:
        move = 'orbit' if horizontal_speed > .6 else 'pan'
    elif horizontal_speed < .4:
        move = 'vertical' if np.median(np.abs(feats['vz'][sl])) > .5 else 'tilt'
    else:
        move = candidate.seg.get('base', 'push').split('+')[0]
        if move in ('orbit', 'spiral', 'pan', 'static'):
            move = ('truck' if abs(float(np.median(feats['v_lat'][sl])))
                    > abs(float(np.median(feats['v_fwd'][sl]))) else 'push')
    row = copy.deepcopy(slot)
    lon, lat = candidate.telemetry['lon'], candidate.telemetry['lat']
    mid = (a + b) // 2
    capture = candidate.capture + timedelta(seconds=(start + end) / 2)
    bearing = montage._bearing(lon[a], lat[a], lon[b-1], lat[b-1])
    row.update(video=candidate.video, fps=candidate.fps, cuts_dir=candidate.cuts_dir,
               start=start, end=end, source_start=start, source_end=end,
               stable_segment_start=candidate.seg['start'], stable_segment_end=candidate.seg['end'],
               duration=duration, slot_start=slot['start_frame']/30, output_fps=30,
               playback_speed=speed, straight_driving=speed > 1, reversed=False,
               move=move, rotation=bool(rotation), telemetry_rotation=bool(raw_rotation),
               camera_rotation_review='regular_tracking_pan' if pan else 'telemetry',
               yaw_mean_abs_dps=round(yaw_mean, 3), yaw_span_deg=round(yaw_span, 3),
               base=candidate.seg['base'], label=candidate.seg['label'], quality=quality,
               score=float(note.get('preference', 1)), scene_id=candidate.scene_id,
               capture_epoch=capture.timestamp(),
               capture_time=capture.isoformat(), capture_start_time=(candidate.capture+timedelta(seconds=start)).isoformat(),
               capture_end_time=(candidate.capture+timedelta(seconds=end)).isoformat(),
               camera_lon=float(lon[mid]), camera_lat=float(lat[mid]), shot_bearing_deg=bearing,
               camera_travel_bearing_deg=bearing, direction_error_deg=None,
               direction_reference='unknown_no_track', route_position_method='no_track_camera_gps_only',
               route_order_mode='no_track')
    for key in ('scene_group', 'scene_description', 'vehicle_present', 'vehicle_moving',
                'vehicle_screen_direction', 'vehicle_sequence', 'camera_view', 'person_present',
                'person_moving', 'story_region', 'sequence_order', 'motion_starts_immediately', 'review_note'):
        if key in note:
            row[key] = note[key]
    row['scene_description'] = note.get('scene_description', note.get('scene', ''))
    row['visual_review'] = dict(approved_source_start=note['start'], approved_source_end=note['end'],
                              annotation_scope='window', review_note=note.get('review_note', ''))
    if speed > 1:
        row['speed_review'] = copy.deepcopy(min(speed_notes, key=lambda n:n['end']-n['start']))
    if pan:
        row['regular_tracking_pan_review'] = copy.deepcopy(min(pan_notes, key=lambda n:n['end']-n['start']))
    return row

def music_grid(source):
    """Copy an already reviewed musical frame grid without mutating its source."""
    return copy.deepcopy(source['timeline'])

"""Plan natural-motion montages on a measured music grid and a timed hiking track.

This module plans only. Source windows remain inside the stable cut intervals;
output times are absolute integer frames so rendering cannot accumulate rounding
error. Nearby landscape shots can share the hiking region without lying on its
track. Reversal is opt-in and requires an explicit review of physical motion.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import glob
import json
import math
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate, find_peaks

from . import segment
from .shot_quality import motion_window_quality, valid_motion_windows, playback_speed_profile


def _spectral_flux(y, sr, hop=256, n_fft=2048):
    """Small FFT batches keep long songs from allocating an entire spectrogram."""
    count = max(0, 1 + (len(y) - n_fft) // hop)
    flux = np.zeros(count, dtype=float)
    previous = None
    win = np.hanning(n_fft)
    for a in range(0, count, 256):
        b = min(a + 256, count)
        indices = np.arange(n_fft)[None, :] + hop * np.arange(a, b)[:, None]
        spec = np.abs(np.fft.rfft(y[indices] * win, axis=1))
        if previous is not None:
            flux[a] = np.maximum(0, spec[0] - previous).sum()
        if b - a > 1:
            flux[a + 1:b] = np.maximum(0, np.diff(spec, axis=0)).sum(axis=1)
        previous = spec[-1]
    flux = gaussian_filter1d(flux, 1)
    flux /= max(float(flux.max(initial=0)), 1e-9)
    times = np.arange(count) * hop / sr + n_fft / (2 * sr)
    return times, flux


def analyse_music(path, fps=30, bpm_range=(60, 150)):
    """Estimate tempo continuously, align pulse phase, and snap beats to onsets.

    FFT lag zero is explicitly retained. Optimising period against the whole
    song removes the tempo bias from rounding autocorrelation lags to samples.
    Downbeat phase is an onset-energy estimate, recorded as such in the plan.
    """
    sr = 22050
    data = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-ac', '1',
                           '-ar', str(sr), '-f', 'f32le', '-'],
                          capture_output=True, check=True).stdout
    y = np.frombuffer(data, dtype='<f4')
    duration = len(y) / sr
    times, env = _spectral_flux(y, sr)
    dt = 256 / sr
    if len(env) < 3 or not np.any(env):
        raise ValueError('music contains too little onset information')
    centered = env - env.mean()
    ac = correlate(centered, centered, mode='full', method='fft')[len(env) - 1:]
    ac /= np.arange(len(env), 0, -1)
    lo, hi = math.ceil(60 / bpm_range[1] / dt), math.floor(60 / bpm_range[0] / dt)
    peaks, _ = find_peaks(ac[lo:hi + 1])
    peaks = peaks + lo
    if not len(peaks):
        peaks = np.array([lo + int(np.argmax(ac[lo:hi + 1]))])
    seeds = sorted(peaks, key=lambda i: ac[i], reverse=True)[:3]
    best = (-float('inf'), 0.0, 0.0)
    for lag in seeds:
        seed = 60 / (lag * dt)
        for bpm in np.arange(max(bpm_range[0], seed - 2),
                             min(bpm_range[1], seed + 2) + 0.001, 0.05):
            period = 60 / bpm
            phases = np.linspace(0, period, 64, endpoint=False)
            grids = phases[:, None] + period * np.arange(math.ceil(duration / period))[None, :]
            vals = np.interp(grids, times, env, left=np.nan, right=np.nan)
            score = np.nanmean(vals, axis=1)
            k = int(np.nanargmax(score))
            # Tiny relative autocorrelation weight prevents weak pulse aliases
            # from winning solely by selecting very few unusually loud attacks.
            objective = float(score[k]) + 0.02 * float(ac[lag] / max(ac[seeds[0]], 1e-9))
            if objective > best[0]:
                best = objective, float(bpm), float(phases[k])
    _, bpm, phase = best
    period = 60 / bpm
    beats = np.arange(phase, duration, period)
    envelope_peaks, _ = find_peaks(env, distance=max(1, int(0.10 / dt)), prominence=0.025)
    onset_times = times[envelope_peaks]
    corrected = []
    for beat in beats:
        if len(onset_times):
            k = int(np.argmin(np.abs(onset_times - beat)))
            onset = float(onset_times[k])
            if abs(onset - beat) <= min(0.065, period * 0.11):
                beat = onset
        corrected.append(float(beat))
    energy = [float(np.mean(np.interp(beats[k::4], times, env)))
              if len(beats[k::4]) else 0.0 for k in range(4)]
    offset = int(np.argmax(energy))
    return dict(music=str(Path(path).resolve()), duration=duration, fps=int(fps),
                total_frames=int(math.ceil(duration * fps - 1e-8)),
                bpm=round(bpm, 4), beat_period=period, beat_phase=phase,
                beats=corrected, bars=corrected[offset::4],
                downbeat_method='strongest_onset_phase_in_estimated_4_4',
                downbeat_phase_energy=energy)


def make_slots(music, pattern=(2, 2, 3), min_s=3.0, max_s=10.0):
    """Cover audio from frame zero to its end; internal cuts follow bar attacks."""
    fps, total = music['fps'], music['total_frames']
    bars = sorted(set(round(t * fps) for t in music['bars'] if 0 < t * fps < total))
    boundaries = [0]
    last_bar = -1
    while True:
        desired_bars = pattern[(len(boundaries) - 1) % len(pattern)]
        if last_bar < 0:
            target = boundaries[-1] + desired_bars * 4 * music['beat_period'] * fps
        else:
            target_index = last_bar + desired_bars
            if target_index >= len(bars):
                break
            target = bars[target_index]
        feasible = [(abs(frame - target), i, frame) for i, frame in enumerate(bars)
                    if min_s * fps <= frame - boundaries[-1] <= max_s * fps
                    and total - frame >= min_s * fps]
        if not feasible:
            break
        _, last_bar, frame = min(feasible)
        boundaries.append(frame)
    boundaries.append(total)
    # A shorter final phrase can be merged; all boundaries remain actual beats.
    if len(boundaries) > 2 and boundaries[-1] - boundaries[-2] < min_s * fps:
        boundaries.pop(-2)
    return [dict(slot=i, start_frame=a, end_frame=b, slot_start=a / fps,
                 duration=(b - a) / fps, output_fps=fps,
                 boundary_error_s=(min(abs(a / fps - t) for t in music['bars']) if i else 0.0))
            for i, (a, b) in enumerate(zip(boundaries, boundaries[1:]))]


def _distance(lon1, lat1, lon2, lat2):
    return 111320 * math.hypot((lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)),
                              lat2 - lat1)


def _bearing(lon1, lat1, lon2, lat2):
    if _distance(lon1, lat1, lon2, lat2) < 2:
        return None
    return math.degrees(math.atan2((lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)),
                                  lat2 - lat1)) % 360


@dataclass
class TimedTrack:
    points: np.ndarray
    times: np.ndarray
    distance: np.ndarray
    start: datetime

    def locate(self, capture):
        seconds = (capture - self.start).total_seconds()
        i = int(np.argmin(np.abs(self.times - seconds)))
        s = float(np.interp(seconds, self.times, self.distance))
        lo = max(0, int(np.searchsorted(self.distance, self.distance[i] - 25)))
        hi = min(len(self.points) - 1, int(np.searchsorted(self.distance, self.distance[i] + 25)))
        lon, lat = self.points[i]
        heading = _bearing(*self.points[lo], *self.points[hi])
        return dict(route_s=s, route_index=i, route_lon=float(lon), route_lat=float(lat),
                    hike_bearing_deg=heading, capture_offset_s=seconds,
                    capture_time=capture.isoformat(), route_time_error_s=float(abs(self.times[i] - seconds)))

    def project_camera(self, lon, lat, *, reference_s=None, region_span_m=750):
        """Project drone GPS onto the track without claiming a subject location.

        Time selects the relevant branch of an out-and-back/loop route. The
        perpendicular distance is informational, never an exclusion radius:
        cliffs or distant valleys may be filmed from well outside the path.
        """
        scale = np.array([111320 * math.cos(math.radians(lat)), 111320])
        xy = (self.points - np.array([lon, lat])) * scale
        if len(xy) < 2:
            return dict(camera_route_s=0.0, camera_route_distance_m=float(np.linalg.norm(xy[0])),
                        camera_route_lon=float(self.points[0, 0]), camera_route_lat=float(self.points[0, 1]),
                        camera_route_position_method='drone_gps_projection_with_timed_branch',
                        subject_geolocation='unknown')
        delta = np.diff(xy, axis=0)
        lengths2 = np.sum(delta * delta, axis=1)
        t = np.clip(-np.sum(xy[:-1] * delta, axis=1) / np.maximum(lengths2, 1e-9), 0, 1)
        nearest = xy[:-1] + t[:, None] * delta
        lateral = np.linalg.norm(nearest, axis=1)
        projected_s = self.distance[:-1] + t * np.diff(self.distance)
        indices = np.arange(len(delta))
        if reference_s is not None:
            local = (self.distance[1:] >= reference_s - region_span_m) & (self.distance[:-1] <= reference_s + region_span_m)
            if np.any(local):
                indices = indices[local]
        i = int(indices[np.argmin(lateral[indices])])
        point = self.points[i] + t[i] * (self.points[i + 1] - self.points[i])
        return dict(camera_route_s=float(projected_s[i]),
                    camera_route_distance_m=float(lateral[i]),
                    camera_route_lon=float(point[0]), camera_route_lat=float(point[1]),
                    camera_route_position_method='drone_gps_projection_with_timed_branch',
                    subject_geolocation='unknown')


def load_timed_track(path):
    root = ET.parse(path).getroot()
    rows = []
    for track in root.iter():
        if track.tag.rsplit('}', 1)[-1] != 'Track':
            continue
        coords = [list(map(float, e.text.split()))[:2] for e in track
                  if e.tag.rsplit('}', 1)[-1] == 'coord' and e.text]
        times = [datetime.fromisoformat(e.text.replace('Z', '+00:00')) for e in track
                 if e.tag.rsplit('}', 1)[-1] == 'when' and e.text]
        if len(coords) != len(times):
            raise ValueError('KML track coordinates and timestamps have different lengths')
        rows.extend(zip(times, coords))
    if not rows:
        raise ValueError('KML requires gx:Track points with timestamps')
    rows.sort(key=lambda r: r[0])
    start = rows[0][0]
    pts = np.asarray([r[1] for r in rows])
    seconds = np.asarray([(r[0] - start).total_seconds() for r in rows])
    lengths = [_distance(*a, *b) for a, b in zip(pts, pts[1:])]
    distance = np.concatenate(([0], np.cumsum(lengths)))
    return TimedTrack(pts, seconds, distance, start)


def capture_datetime(video, timezone='Asia/Shanghai'):
    m = re.search(r'DJI_(\d{14})', Path(video).name)
    if not m:
        raise ValueError(f'no capture timestamp in {video}')
    return datetime.strptime(m[1], '%Y%m%d%H%M%S').replace(tzinfo=ZoneInfo(timezone))


@dataclass
class SourceCandidate:
    video: str
    fps: float
    seg: dict
    feats: dict
    telemetry: dict
    cuts_dir: str
    capture: datetime
    track: TimedTrack | None
    scene_id: int = 0
    cache: dict = field(default_factory=dict)

    def windows(self, duration, annotations=None, choices=4):
        key = round(duration, 8)
        if key not in self.cache:
            starts = valid_motion_windows(self.feats, self.fps, self.seg['start'],
                                          self.seg['end'], duration, step_s=0.75)
            rows = []
            for start in starts:
                end = start + duration
                a, b = int(round(start * self.fps)), int(math.ceil(end * self.fps))
                sl = slice(a, b)
                quality = motion_window_quality(self.feats, self.fps, start, end)
                profile = playback_speed_profile(self.feats, self.fps, start, end, output_fps=30)
                maximum = profile['max_playback_speed']
                if maximum is not None and maximum < 1:
                    continue
                yaw = self.feats['yaw_rate'][sl]
                yaw_mean = float(np.mean(np.abs(yaw)))
                yaw_span = float(np.abs(self.feats['g_yaw'][b - 1] - self.feats['g_yaw'][a]))
                rotation = yaw_mean > 1.5 and yaw_span > 8.0
                # Plan the window's actual movement; a long fused segment label
                # may describe an orbit that happens outside this window.
                if rotation:
                    move = 'orbit' if np.median(self.feats['speed_h'][sl]) > 0.6 else 'pan'
                elif np.median(self.feats['speed_h'][sl]) < 0.4:
                    move = 'vertical' if np.median(np.abs(self.feats['vz'][sl])) > 0.5 else 'tilt'
                else:
                    move = self.seg.get('base', 'push').split('+')[0]
                    if move in ('orbit', 'spiral', 'pan', 'static'):
                        move = 'truck' if abs(float(np.median(self.feats['v_lat'][sl]))) > abs(float(np.median(self.feats['v_fwd'][sl]))) else 'push'
                m = (a + b) // 2
                lon, lat = self.telemetry['lon'], self.telemetry['lat']
                shot_bearing = _bearing(lon[a], lat[a], lon[b - 1], lat[b - 1])
                cap = self.capture + timedelta(seconds=(start + end) / 2)
                route = self.track.locate(cap) if self.track else dict(route_s=cap.timestamp(), capture_time=cap.isoformat())
                if self.track:
                    route.update(self.track.project_camera(float(lon[m]), float(lat[m]),
                                                           reference_s=route['route_s']))
                    route['route_position_method'] = 'timed_hiker_progress'
                hb = route.get('hike_bearing_deg')
                difference = abs((shot_bearing - hb + 180) % 360 - 180) if shot_bearing is not None and hb is not None else None
                score = (min(float(np.median(self.feats['n_tracks'][sl])) / 500, 1)
                         + min(math.log1p(quality['motion_median_px_s']) / math.log(100), 1.5)
                         - min(float(np.mean(self.feats['jitter_burst'][sl])) / 10, 2))
                rows.append(dict(video=self.video, start=start, end=end, source_start=start,
                                 source_end=end, fps=self.fps, duration=duration,
                                 cuts_dir=self.cuts_dir, base=self.seg['base'],
                                 label=self.seg['label'], move=move, rotation=bool(rotation),
                                 yaw_mean_abs_dps=round(yaw_mean, 3), yaw_span_deg=round(yaw_span, 3),
                                 reversed=False, capture_epoch=cap.timestamp(),
                                 camera_lon=float(lon[m]), camera_lat=float(lat[m]),
                                 shot_bearing_deg=shot_bearing, direction_error_deg=difference,
                                 directional=bool(move in ('push', 'pull', 'truck', 'diagonal')
                                                  and shot_bearing is not None),
                                 scene_id=self.scene_id, quality=quality, score=score,
                                 max_playback_speed=maximum, playback_speed_profile=profile,
                                 stable_segment_start=self.seg['start'],
                                 stable_segment_end=self.seg['end'], **route))
            # Keep high-quality and evenly spaced alternatives; otherwise one
            # high-motion beginning can hide a later, direction-aligned window.
            rows.sort(key=lambda r: r['score'], reverse=True)
            self.cache[key] = rows
        # Annotate the copy, not the cache: different reviews must not leak into
        # one another.
        rows = []
        for cached in self.cache[key]:
            row = dict(cached)
            if not _annotation_allowed(row, annotations):
                continue
            if _reverse_review(row, annotations):
                n = len(self.feats['yaw_rate'])
                a = max(0, int(math.ceil(row['start'] * self.fps - 1e-6)))
                b = min(n, int(math.ceil(row['end'] * self.fps - 1e-6)))
                reverse_feats = {k: np.asarray(v[a:b])[::-1]
                                 for k, v in self.feats.items()
                                 if isinstance(v, (np.ndarray, list)) and len(v) == n}
                row['reverse_quality'] = motion_window_quality(reverse_feats, self.fps, 0, (b - a) / self.fps)
            rows.append(row)
        if len(rows) <= choices:
            return rows
        chosen = [rows[0]]
        by_time = sorted(rows, key=lambda r: r['start'])
        for i in np.linspace(0, len(by_time) - 1, choices - 1).round().astype(int):
            row = by_time[i]
            if all(abs(row['start'] - k['start']) > 0.1 for k in chosen):
                chosen.append(row)
        return chosen


def _annotation_allowed(row, annotations):
    source_notes = annotations.get('sources', {}) if isinstance(annotations, dict) else {}
    source_note = source_notes.get(Path(row['video']).name, {})
    row['scene_group'] = source_note.get('scene_cluster', f'gps_{row["scene_id"]}')
    if source_note:
        row['source_visual_review'] = source_note
    for key in ('person_present', 'person_moving', 'review_note'):
        if key in source_note:
            row[key] = source_note[key]
    clips = annotations.get('clips', []) if isinstance(annotations, dict) else (annotations or [])
    for note in clips:
        if Path(note['video']).name != Path(row['video']).name:
            continue
        if min(row['end'], note.get('end', float('inf'))) <= max(row['start'], note.get('start', 0)):
            continue
        if note.get('allowed') is False:
            return False
        if note.get('scene'):
            row['scene_description'] = note['scene']
        for key in ('person_present', 'person_moving', 'review_note'):
            if key in note:
                row[key] = note[key]
    return True


def _reverse_review(row, annotations):
    """Require affirmative, complete-window review; detector absence is insufficient."""
    if not isinstance(annotations, dict):
        return False
    basename = Path(row['video']).name
    source = annotations.get('sources', {}).get(basename, {})
    reviews = [source]
    overlaps = []
    for note in annotations.get('clips', []):
        if Path(note['video']).name != basename:
            continue
        if min(row['end'], note.get('end', float('inf'))) <= max(row['start'], note.get('start', 0)):
            continue
        overlaps.append(note)
        if note.get('start', 0) <= row['start'] and note.get('end', float('inf')) >= row['end']:
            reviews.append(dict(source, **note))
    if any(n.get('allow_reverse') is False or n.get('person_moving') is True
           or n.get('physics_reversible') is False for n in [source] + overlaps):
        return False
    for review in reversed(reviews):
        if (review.get('allow_reverse') is True and review.get('physics_reversible') is True
                and (review.get('person_present') is False or review.get('person_moving') is False)):
            row['reverse_safety_review'] = {k: review[k] for k in
                                           ('allow_reverse', 'physics_reversible', 'person_present',
                                            'person_moving', 'review_note') if k in review}
            row['reverse_safety_review']['annotation_scope'] = 'window' if 'start' in review or 'end' in review else 'whole_source'
            if 'start' in review:
                row['reverse_safety_review']['approved_source_start'] = review['start']
            if 'end' in review:
                row['reverse_safety_review']['approved_source_end'] = review['end']
            return True
    return False


def _reverse_option(row, max_direction_error_deg):
    """Reverse only an otherwise misaligned reviewed translation, with a live start."""
    direction = row.get('direction_error_deg')
    ceiling = 90 if max_direction_error_deg is None else max_direction_error_deg
    if (not row.get('directional', not row['rotation']) or direction is None
            or direction <= ceiling or not row.get('reverse_safety_review')
            or not row.get('reverse_quality', {}).get('ok')):
        return None
    reverse_error = 180 - direction
    if reverse_error > ceiling:
        return None
    reversed_row = dict(row, reversed=True, direction_error_deg=reverse_error,
                        quality=row['reverse_quality'], original_direction_error_deg=direction)
    if row.get('shot_bearing_deg') is not None:
        reversed_row['shot_bearing_deg'] = (row['shot_bearing_deg'] + 180) % 360
    reversed_row['move'] = {'push': 'pull', 'pull': 'push'}.get(row['move'], row['move'])
    return reversed_row


def load_candidates(dirs, variant='_v6', track_path=None, timezone='Asia/Shanghai'):
    track = load_timed_track(track_path) if track_path else None
    candidates = []
    for directory in dirs:
        paths = glob.glob(str(Path(directory) / f'*_cuts{variant}' / '*_segments.json'))
        if Path(directory).name.endswith(f'_cuts{variant}'):
            paths += glob.glob(str(Path(directory) / '*_segments.json'))
        for path in sorted(set(paths)):
            doc = json.loads(Path(path).read_text())
            base = Path(doc['video']).stem
            cuts = Path(path).parent
            with (cuts / f'{base}_telemetry.csv').open() as fh:
                tel = list(csv.DictReader(fh))
            with (cuts / f'{base}_flow.csv').open() as fh:
                flow = list(csv.DictReader(fh))
            feats, _ = segment.features(tel, flow, doc['fps'])
            aligned, _ = segment.aligned_tel(tel, len(flow), doc['fps'])
            cap = capture_datetime(doc['video'], timezone)
            for seg in doc['segments']:
                if seg['duration'] >= 3:
                    candidates.append(SourceCandidate(doc['video'], doc['fps'], seg,
                                                      feats, aligned, str(cuts), cap, track))
    candidates.sort(key=lambda c: (c.capture.timestamp(), c.seg['start']))
    # Keep alternatives at a location, then penalise using that location again
    # in the sequence. Early deduplication would discard a better camera move.
    clusters = []
    for candidate in candidates:
        i = min(len(candidate.telemetry['lon']) - 1,
                round((candidate.seg['start'] + candidate.seg['end']) / 2 * candidate.fps))
        point = float(candidate.telemetry['lon'][i]), float(candidate.telemetry['lat'][i])
        nearby = [k for k, p in enumerate(clusters) if _distance(*point, *p) < 120]
        candidate.scene_id = nearby[0] if nearby else len(clusters)
        if not nearby:
            clusters.append(point)
    return candidates


def plan_sequence(candidates, slots, annotations=None, beam_width=100, max_per_flight=None,
                  direction_weight=1.8, max_direction_error_deg=None,
                  route_mode='strict', max_backtrack_m=250, route_progress_weight=None,
                  allow_reverse=False, reverse_penalty=2.5, max_per_visual_group=None,
                  max_consecutive_visual_group=None, visual_group_limit_slots=None):
    """Plan either strict chronology or a sequence of nearby hiking regions.

    Regional editing permits earlier/later takes in the same area, including
    off-track scenery. Its backward allowance is measured from the furthest
    selected progress, so many small backwards steps cannot unwind the hike.
    Natural playback remains the default in either ordering mode.
    """
    if not candidates or not slots:
        raise ValueError('no candidates or music slots')
    if route_mode not in ('strict', 'regional'):
        raise ValueError("route_mode must be 'strict' or 'regional'")
    if max_backtrack_m < 0:
        raise ValueError('max_backtrack_m must be nonnegative')
    if max_per_visual_group is not None and max_per_visual_group < 1:
        raise ValueError('max_per_visual_group must be positive')
    if max_consecutive_visual_group is not None and max_consecutive_visual_group < 1:
        raise ValueError('max_consecutive_visual_group must be positive')
    if route_mode == 'regional' and not all(c.track is not None for c in candidates):
        raise ValueError('regional ordering requires a timed hiking track')
    progress_weight = (13 if route_mode == 'strict' else 5) if route_progress_weight is None else route_progress_weight
    flights = {c.video for c in candidates}
    max_per_flight = max_per_flight or max(1, math.ceil(len(slots) / len(flights)) + 1)
    all_route = [c.track.locate(c.capture + timedelta(seconds=(c.seg['start'] + c.seg['end']) / 2))['route_s']
                 if c.track else c.capture.timestamp() for c in candidates]
    route_min, route_max = min(all_route), max(all_route)
    route_span = max(route_max - route_min, 1)
    options_by_slot = []
    for slot in slots:
        options = [row for c in candidates for row in c.windows(slot['duration'], annotations)]
        if allow_reverse:
            options += [reversed_row for row in list(options)
                        if (reversed_row := _reverse_option(row, max_direction_error_deg)) is not None]
        if max_direction_error_deg is not None:
            options = [r for r in options if not r.get('directional', not r['rotation'])
                       or r['direction_error_deg'] is None
                       or r['direction_error_deg'] <= max_direction_error_deg]
        options.sort(key=lambda r: (r['capture_epoch'], r['start']))
        options_by_slot.append(options)
    # Solve a relaxed suffix problem first. This stops an attractive early shot
    # near the route's end from consuming footage needed by later music slots.
    # The suffix honours time, interval overlap, rotations and physical-scene
    # runs. Whole-flight reuse counts remain part of the forward beam state.
    suffix = _suffix_feasibility(options_by_slot, route_mode, max_backtrack_m)
    option_arrays = [dict(video=np.asarray([r['video'] for r in rows]),
                          start=np.asarray([r['start'] for r in rows]),
                          end=np.asarray([r['end'] for r in rows]),
                          route=np.asarray([r['route_s'] for r in rows]),
                          group=np.asarray([r['scene_group'] for r in rows]),
                          scene=np.asarray([r['scene_id'] for r in rows]),
                          rotation=np.asarray([r['rotation'] for r in rows], dtype=bool))
                     for rows in options_by_slot]
    future_resource_cache = {}
    future_music_bounds = [(min(s['duration'] for s in slots[i + 1:]),
                            max(s['duration'] for s in slots[i + 1:]),
                            sum(s['duration'] for s in slots[i + 1:]))
                           if i + 1 < len(slots) else (0, 0, 0) for i in range(len(slots))]

    def future_resources(si, furthest_s):
        # Round the bound down: the resource check is an upper bound and must
        # never reject a feasible edit due to geographic rounding.
        floor = math.floor((furthest_s - max_backtrack_m) / 5) * 5
        key = si, floor
        if key not in future_resource_cache:
            intervals, groups = {}, set()
            for rows in options_by_slot[si + 1:]:
                for r in rows:
                    if r['route_s'] < floor:
                        continue
                    groups.add(r['scene_group'])
                    intervals.setdefault(r['video'], []).append((r['start'], r['end']))
            merged = {}
            for video, spans in intervals.items():
                union = []
                for a, b in sorted(spans):
                    if union and a <= union[-1][1]:
                        union[-1] = (union[-1][0], max(b, union[-1][1]))
                    else:
                        union.append((a, b))
                merged[video] = union, sum(b - a for a, b in union)
            future_resource_cache[key] = groups, merged
        return future_resource_cache[key]

    def enough_future_resources(si, used_intervals, row, uses, scenes, furthest_s):
        remaining = len(slots) - si - 1
        if not remaining:
            return True
        groups, intervals = future_resources(si, max(furthest_s, row['route_s']))
        if max_per_visual_group is not None:
            capacity = sum(max(0, max_per_visual_group - scenes.get(g, 0)
                               - int(g == row['scene_group'])) for g in groups)
            if capacity < remaining:
                return False
        min_duration, max_duration, music_remaining = future_music_bounds[si]
        shot_capacity, time_capacity = 0, 0.0
        for video, (spans, seconds) in intervals.items():
            quota = max_per_flight - uses.get(video, 0) - int(video == row['video'])
            if quota <= 0:
                continue
            selected = used_intervals.get(video, [])
            if video == row['video']:
                selected = selected + [(row['start'], row['end'])]
            for used_a, used_b in selected:
                seconds -= sum(max(0, min(b, used_b) - max(a, used_a)) for a, b in spans)
            seconds = max(0, seconds)
            shot_capacity += min(quota, math.floor(seconds / min_duration + 1e-8))
            time_capacity += min(seconds, quota * max_duration)
        return (shot_capacity >= remaining and
                time_capacity + 1e-7 >= music_remaining)
    # (score, timeline, flight-use counts, scene-use counts)
    beam = [(0.0, [], {}, {})]
    for si, slot in enumerate(slots):
        options = options_by_slot[si]
        target = (slot['start_frame'] + slot['end_frame']) / (2 * slots[-1]['end_frame'])
        next_beam = []
        for accumulated, timeline, uses, scenes in beam:
            previous = timeline[-1] if timeline else None
            furthest_route_s = max((r['route_s'] for r in timeline), default=-float('inf'))
            used_intervals = {}
            for used in timeline:
                used_intervals.setdefault(used['video'], []).append((used['start'], used['end']))
            future_masks = {}
            if route_mode == 'regional' and si + 1 < len(slots):
                # Always reserve an actual closing window. Near the end, also
                # inspect each remaining slot; total seconds alone can conceal
                # short fragments left around previously selected windows.
                inspected = range(si + 1, len(slots)) if len(slots) - si <= 6 else [len(slots) - 1]
                for future_si in inspected:
                    arrays = option_arrays[future_si]
                    mask = arrays['route'] + max_backtrack_m >= furthest_route_s
                    for video, count in uses.items():
                        if count >= max_per_flight:
                            mask &= arrays['video'] != video
                    if max_per_visual_group is not None:
                        for group, count in scenes.items():
                            if count >= max_per_visual_group:
                                mask &= arrays['group'] != group
                    for video, spans in used_intervals.items():
                        for a, b in spans:
                            mask &= ~((arrays['video'] == video) & (arrays['start'] < b + 0.4)
                                      & (arrays['end'] + 0.4 > a))
                    future_masks[future_si] = mask
            for oi, row in enumerate(options):
                if uses.get(row['video'], 0) >= max_per_flight:
                    continue
                if max_per_visual_group is not None and scenes.get(row['scene_group'], 0) >= max_per_visual_group:
                    continue
                visual_run = 1
                for earlier in reversed(timeline):
                    if earlier['scene_group'] != row['scene_group']:
                        break
                    visual_run += 1
                if (max_consecutive_visual_group is not None
                        and (visual_group_limit_slots is None or si < visual_group_limit_slots)
                        and visual_run > max_consecutive_visual_group):
                    continue
                # Check the entire edit, not just adjacent shots: regional
                # inserts may return to an earlier flight without repeating it.
                if any(r['video'] == row['video'] and
                       min(r['end'] + 0.4, row['end'] + 0.4) > max(r['start'], row['start'])
                       for r in timeline):
                    continue
                if previous:
                    if route_mode == 'strict':
                        if row['capture_epoch'] <= previous['capture_epoch']:
                            continue
                        if row['route_s'] + 3 < previous['route_s']:
                            continue
                        if row['video'] == previous['video'] and row['start'] < previous['end'] + 0.4:
                            continue
                    elif row['route_s'] + max_backtrack_m < furthest_route_s:
                        continue
                    if row['rotation'] and previous['rotation']:
                        continue
                    if (len(timeline) >= 2 and timeline[-2]['scene_id'] == row['scene_id']
                            and previous['scene_id'] == row['scene_id']):
                        continue
                run = 2 if previous and previous['scene_id'] == row['scene_id'] else 1
                if not suffix[si][oi, run - 1]:
                    continue
                if route_mode == 'regional' and not enough_future_resources(
                        si, used_intervals, row, uses, scenes, furthest_route_s):
                    continue
                future_ok = True
                for future_si, before in future_masks.items():
                    arrays = option_arrays[future_si]
                    mask = before & (arrays['route'] + max_backtrack_m >= max(furthest_route_s, row['route_s']))
                    same_video = arrays['video'] == row['video']
                    if uses.get(row['video'], 0) + 1 >= max_per_flight:
                        mask &= ~same_video
                    else:
                        mask &= ~(same_video & (arrays['start'] < row['end'] + 0.4)
                                  & (arrays['end'] + 0.4 > row['start']))
                    if max_per_visual_group is not None and scenes.get(row['scene_group'], 0) + 1 >= max_per_visual_group:
                        mask &= arrays['group'] != row['scene_group']
                    if sum(r['rotation'] for r in timeline) + row['rotation'] >= math.ceil(len(slots) * 0.32):
                        mask &= ~arrays['rotation']
                    if future_si == si + 1:
                        if row['rotation']:
                            mask &= ~arrays['rotation']
                        if run == 2:
                            mask &= arrays['scene'] != row['scene_id']
                        if (max_consecutive_visual_group is not None
                                and (visual_group_limit_slots is None or future_si < visual_group_limit_slots)
                                and visual_run >= max_consecutive_visual_group):
                            mask &= arrays['group'] != row['scene_group']
                    if not np.any(mask):
                        future_ok = False
                        break
                if not future_ok:
                    continue
                progress = (row['route_s'] - route_min) / route_span
                score = row['score'] - progress_weight * abs(progress - target)
                if route_mode == 'regional':
                    if row['scene_group'] not in scenes:
                        score += 0.6
                    if previous and row['route_s'] < furthest_route_s:
                        score -= 0.8 * (furthest_route_s - row['route_s']) / max(max_backtrack_m, 1)
                if row.get('reversed'):
                    score -= reverse_penalty
                direction = row['direction_error_deg']
                if direction is not None and row.get('directional', not row['rotation']):
                    score += direction_weight * math.cos(math.radians(direction))
                if row['rotation']:
                    score -= 0.8
                    if sum(r['rotation'] for r in timeline) >= math.ceil(len(slots) * 0.32):
                        continue
                score -= 0.8 * uses.get(row['video'], 0)
                score -= 0.65 * scenes.get(row['scene_group'], 0)
                if previous:
                    distance = _distance(previous['camera_lon'], previous['camera_lat'], row['camera_lon'], row['camera_lat'])
                    score += min(distance / 250, 1.0)
                    if row['scene_group'] == previous['scene_group']:
                        score -= 1.8
                    if row['move'] == previous['move']:
                        score -= 0.3
                selected = dict(row, **slot, seq=si + 1, route_progress=progress,
                                route_order_mode=route_mode,
                                route_backtrack_m=max(0.0, furthest_route_s - row['route_s']) if previous else 0.0)
                new_uses, new_scenes = dict(uses), dict(scenes)
                new_uses[row['video']] = new_uses.get(row['video'], 0) + 1
                new_scenes[row['scene_group']] = new_scenes.get(row['scene_group'], 0) + 1
                next_beam.append((accumulated + score, timeline + [selected], new_uses, new_scenes))
        if not next_beam:
            raise ValueError(f'cannot fill slot {si + 1}/{len(slots)} ({slot["duration"]:.2f}s) '
                             f'with accepted footage in {route_mode} route order')
        next_beam.sort(key=lambda state: state[0], reverse=True)
        # Preserve multiple possible endings. Near-identical windows otherwise
        # crowd the beam and make all alternatives run out together near the end.
        grouped = {}
        kept = []
        for state in next_beam:
            last = state[1][-1]
            signature = (last['video'], last['scene_id'], last['rotation'], round(last['end'] / 5))
            if max_per_visual_group is not None:
                signature += (frozenset(state[3]),)
            if grouped.get(signature, 0) >= 3:
                continue
            grouped[signature] = grouped.get(signature, 0) + 1
            kept.append(state)
            if len(kept) >= beam_width:
                break
        beam = kept
    return max(beam, key=lambda state: state[0])[1]


def _suffix_feasibility(options_by_slot, route_mode='strict', max_backtrack_m=250):
    """Whether each proposed shot can finish all remaining slots, for scene run 1/2."""
    possible = [np.zeros((len(rows), 2), dtype=bool) for rows in options_by_slot]
    if not possible:
        return possible
    possible[-1][:] = True
    for si in range(len(options_by_slot) - 2, -1, -1):
        following = options_by_slot[si + 1]
        if not following:
            continue
        cap = np.asarray([r['capture_epoch'] for r in following])
        route = np.asarray([r['route_s'] for r in following])
        rotation = np.asarray([r['rotation'] for r in following])
        scenes = np.asarray([r['scene_id'] for r in following])
        videos = np.asarray([r['video'] for r in following])
        starts = np.asarray([r['start'] for r in following])
        ends = np.asarray([r['end'] for r in following])
        for oi, row in enumerate(options_by_slot[si]):
            if route_mode == 'strict':
                compatible = ((cap > row['capture_epoch']) & (route + 3 >= row['route_s'])
                              & ~((videos == row['video']) & (starts < row['end'] + 0.4)))
            else:
                compatible = ((route + max_backtrack_m >= row['route_s'])
                              & ~((videos == row['video']) & (starts < row['end'] + 0.4)
                                  & (ends + 0.4 > row['start'])))
            if row['rotation']:
                compatible &= ~rotation
            same = scenes == row['scene_id']
            # A different location resets a run to one; same location changes
            # run one to two and is prohibited after run two.
            different_ok = compatible & ~same & possible[si + 1][:, 0]
            same_ok = compatible & same & possible[si + 1][:, 1]
            possible[si][oi, 0] = bool(np.any(different_ok | same_ok))
            possible[si][oi, 1] = bool(np.any(different_ok))
    return possible


def plan_summary(timeline):
    videos = [r for r in timeline if r.get('media_type', 'video') == 'video']
    photos = [r for r in timeline if r.get('media_type') == 'photo']
    return dict(shots=len(timeline), total_frames=timeline[-1]['end_frame'] if timeline else 0,
                video_shots=len(videos), photo_shots=len(photos),
                flights=len({r['video'] for r in videos}),
                scene_clusters=len({r['scene_id'] for r in timeline}),
                visual_scene_groups=len({r['scene_group'] for r in timeline}),
                rotations=sum(r['rotation'] for r in videos),
                consecutive_rotations=any(a.get('rotation', False) and b.get('rotation', False)
                                          for a, b in zip(timeline, timeline[1:])),
                reversed_shots=sum(r['reversed'] for r in videos),
                aligned_straight_shots=sum(r['direction_error_deg'] is not None and r['direction_error_deg'] <= 90
                                           and r.get('directional', not r['rotation']) for r in videos),
                opposed_straight_shots=sum(r['direction_error_deg'] is not None and r['direction_error_deg'] > 90
                                           and r.get('directional', not r['rotation']) for r in videos),
                neutral_shots=len(photos) + sum(not r.get('directional', not r['rotation']) for r in videos),
                reordered_capture_transitions=sum(a['capture_epoch'] >= b['capture_epoch']
                                                  for a, b in zip(timeline, timeline[1:])),
                regional_backtrack_shots=sum(r.get('route_backtrack_m', 0) > 3 for r in timeline),
                max_route_backtrack_m=max((r.get('route_backtrack_m', 0) for r in timeline), default=0),
                off_track_camera_shots=sum(r.get('camera_route_distance_m', 0) > 50 for r in timeline),
                all_motion_gates_passed=all(r['quality']['ok'] for r in videos))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('music')
    parser.add_argument('dirs', nargs='+')
    parser.add_argument('--track')
    parser.add_argument('--variant', default='_v6')
    parser.add_argument('--annotations')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--bar-pattern', default='2,2,3',
                        help='comma-separated bar counts for successive shots')
    parser.add_argument('--max-direction-error', type=float,
                        help='reject known straight travel farther than this angle from the hike')
    parser.add_argument('--route-mode', choices=('strict', 'regional'),
                        help='regional allows nearby takes in flexible order (default with a track); strict preserves capture chronology')
    parser.add_argument('--max-backtrack-m', type=float, default=250,
                        help='regional allowance behind the furthest hiking progress already selected')
    parser.add_argument('--allow-reverse', action='store_true',
                        help='allow necessary reversal only for physically reversible, manually approved windows')
    parser.add_argument('--max-per-flight', type=int,
                        help='limit selected shots from each original video')
    parser.add_argument('--max-per-visual-group', type=int,
                        help='optional scene limit, useful for a short montage with no repeated scenery')
    parser.add_argument('--max-consecutive-visual-group', type=int,
                        help='optional limit on consecutive shots of the same visual scene')
    parser.add_argument('--visual-group-limit-slots', type=int,
                        help='apply the consecutive visual-scene limit only to this many opening shots')
    parser.add_argument('-o', required=True)
    args = parser.parse_args()
    music = analyse_music(args.music, args.fps)
    pattern = tuple(int(value) for value in args.bar_pattern.split(','))
    if not pattern or any(value <= 0 for value in pattern):
        parser.error('--bar-pattern requires positive integer bar counts')
    slots = make_slots(music, pattern=pattern)
    candidates = load_candidates(args.dirs, args.variant, args.track)
    annotations = json.loads(Path(args.annotations).read_text()) if args.annotations else None
    route_mode = args.route_mode or ('regional' if args.track else 'strict')
    timeline = plan_sequence(candidates, slots, annotations,
                             max_direction_error_deg=args.max_direction_error,
                             route_mode=route_mode, max_backtrack_m=args.max_backtrack_m,
                             allow_reverse=args.allow_reverse, max_per_flight=args.max_per_flight,
                             max_per_visual_group=args.max_per_visual_group,
                             max_consecutive_visual_group=args.max_consecutive_visual_group,
                             visual_group_limit_slots=args.visual_group_limit_slots)
    report = dict(music_analysis=music, track=args.track, variant=args.variant,
                  slots=slots, timeline=timeline, summary=plan_summary(timeline),
                  route_mode=route_mode, max_backtrack_m=args.max_backtrack_m,
                  allow_reverse=args.allow_reverse,
                  policy=dict(pattern=pattern, max_direction_error_deg=args.max_direction_error,
                              route_mode=route_mode, max_backtrack_m=args.max_backtrack_m,
                              allow_reverse=args.allow_reverse, max_per_flight=args.max_per_flight,
                              max_per_visual_group=args.max_per_visual_group,
                              max_consecutive_visual_group=args.max_consecutive_visual_group,
                              visual_group_limit_slots=args.visual_group_limit_slots))
    Path(args.o).parent.mkdir(parents=True, exist_ok=True)
    Path(args.o).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report['summary'], ensure_ascii=False))


if __name__ == '__main__':
    main()

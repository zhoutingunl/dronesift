"""Plan a reviewed, music-aligned driving montage without inventing a route.

Quality rows are cached once from existing v6 flow/telemetry. The visual catalog
then supplies complete approved windows and their actual scene/vehicle context.
No video decoding, automatic vehicle detection, or reverse playback is used.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dronesift import montage, shot_quality


def gate_fingerprint():
    return hashlib.sha256((inspect.getsource(montage.SourceCandidate)
                           + Path(shot_quality.__file__).read_text(encoding='utf-8')).encode()).hexdigest()


def create_cache(source_dirs, slots, path):
    """Read only selected sources' existing v6 data; retain all accepted starts."""
    candidates = montage.load_candidates(source_dirs, track_path=None)
    durations = sorted({round(s['duration'], 8) for s in slots})
    records = [dict(video=c.video, seg=c.seg, capture=c.capture.isoformat(), windows={})
               for c in candidates]
    for i, duration in enumerate(durations):
        for c, record in zip(candidates, records):
            record['windows'][str(duration)] = c.windows(duration, choices=100000)
        print(f'Validated duration {i + 1}/{len(durations)}: {duration:.3f}s', flush=True)
    source_stats = {c.video: dict(size=Path(c.video).stat().st_size,
                                 mtime_ns=Path(c.video).stat().st_mtime_ns) for c in candidates}
    doc = dict(window_gate_hash=gate_fingerprint(), track=None, source_stats=source_stats,
               sources=sorted({c.video for c in candidates}), candidates=records)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, ensure_ascii=False), encoding='utf-8')
    return doc


def extend_cache(source_dirs, slots, path):
    """Add only new durations for selected closing sources, without a full rescan."""
    doc = json.loads(Path(path).read_text(encoding='utf-8'))
    if doc.get('window_gate_hash') != gate_fingerprint():
        raise ValueError('cannot extend a cache generated with a different quality gate')
    durations = sorted({round(s['duration'], 8) for s in slots}
                       - {float(k) for c in doc['candidates'] for k in c['windows']})
    if not durations:
        return doc
    candidates = montage.load_candidates(source_dirs, track_path=None)
    selected = {(c.video, c.seg['start'], c.seg['end']): c for c in candidates}
    for duration in durations:
        for record in doc['candidates']:
            candidate = selected.get((record['video'], record['seg']['start'], record['seg']['end']))
            record['windows'][str(duration)] = candidate.windows(duration, choices=100000) if candidate else []
        print(f'Added selected closing-source duration: {duration:.3f}s', flush=True)
    Path(path).write_text(json.dumps(doc, ensure_ascii=False), encoding='utf-8')
    return doc


def _overlaps(row, note):
    return (Path(row['video']).name == Path(note['video']).name
            and row['start'] < note['end'] - 1e-7
            and row['end'] > note['start'] + 1e-7)


def reviewed_row(cached, annotations):
    """Only accept a fully enclosed approved visual window, never partial review."""
    row = copy.deepcopy(cached)
    if row.get('reversed') or not row.get('quality', {}).get('ok'):
        return None
    source = annotations.get('sources', {}).get(Path(row['video']).name, {})
    if source.get('allowed') is False:
        return None
    clips = annotations.get('clips', annotations.get('windows', []))
    if any(_overlaps(row, n) and n.get('allowed') is False for n in clips):
        return None
    matches = [n for n in clips if Path(n['video']).name == Path(row['video']).name
               and n.get('allowed') is True and n['start'] <= row['start'] + 1e-7
               and n['end'] >= row['end'] - 1e-7]
    if not matches:
        return None
    note = min(matches, key=lambda n: n['end'] - n['start'])
    review = dict(source, **note)
    if not isinstance(review.get('vehicle_present'), bool):
        raise ValueError('approved visual windows must explicitly review vehicle presence')
    if review['vehicle_present'] and not isinstance(review.get('vehicle_moving'), bool):
        raise ValueError('vehicle windows must explicitly distinguish driving from parking')
    if review.get('motion_starts_immediately') is False:
        return None
    for key in ('scene_group', 'scene_description', 'vehicle_present', 'vehicle_moving',
                'vehicle_screen_direction', 'vehicle_sequence', 'camera_view',
                'motion_starts_immediately', 'person_present', 'person_moving',
                'story_region', 'sequence_order', 'preference', 'review_note'):
        if key in review:
            row[key] = review[key]
    row['scene_group'] = review.get('scene_group', review.get('scene_cluster', row.get('scene_group')))
    if not row.get('scene_group'):
        raise ValueError('a reviewed window needs a scene_group')
    row['scene_description'] = review.get('scene_description', review.get('scene', ''))
    row['visual_review'] = dict(approved_source_start=note['start'], approved_source_end=note['end'],
                               annotation_scope='window', review_note=review.get('review_note', ''))
    row['reversed'] = False
    row['route_position_method'] = 'no_track_camera_gps_only'
    # Epochs are timestamps, never metres or a proxy hiking track.
    for key in ('route_s', 'route_lon', 'route_lat', 'hike_bearing_deg', 'camera_route_s',
                'camera_route_distance_m', 'camera_route_lon', 'camera_route_lat',
                'reverse_safety_review', 'reverse_quality'):
        row.pop(key, None)
    row['direction_error_deg'] = None
    row['direction_reference'] = 'unknown_no_track'
    row['camera_travel_bearing_deg'] = row.get('shot_bearing_deg')
    return row


def options_from_cache(cache, slots, annotations, samples_per_window=16):
    if cache.get('window_gate_hash') != gate_fingerprint():
        raise ValueError('motion-window cache was generated with a different quality gate')
    for video, expected in cache.get('source_stats', {}).items():
        actual = Path(video).stat()
        if actual.st_size != expected['size'] or actual.st_mtime_ns != expected['mtime_ns']:
            raise ValueError(f'cached original source changed: {video}')
    approved_names = {Path(n['video']).name for n in annotations.get('clips', annotations.get('windows', []))
                      if n.get('allowed') is True}
    active_sources = [v for v in sorted(cache['sources']) if Path(v).name in approved_names
                      and annotations.get('sources', {}).get(Path(v).name, {}).get('allowed') is not False]
    source_order = {v: i for i, v in enumerate(active_sources)}
    max_end = {v: max(c['seg']['end'] for c in cache['candidates'] if c['video'] == v)
               for v in cache['sources']}
    by_duration = {}
    for duration in sorted({round(s['duration'], 8) for s in slots}):
        reviewed = []
        for candidate in cache['candidates']:
            key = str(duration)
            if key not in candidate['windows']:
                raise ValueError(f'missing cached duration {duration}')
            for cached in candidate['windows'][key]:
                row = reviewed_row(cached, annotations)
                if row is not None:
                    row['source_order_progress'] = (source_order[row['video']]
                        + (row['start'] + row['end']) / (2 * max_end[row['video']])) / max(1, len(source_order))
                    reviewed.append(row)
        # A long fused cut can contain many roads/views. Sample each reviewed
        # interval and movement independently instead of keeping four per source.
        groups = {}
        for row in reviewed:
            review = row['visual_review']
            key = (row['video'], review['approved_source_start'], review['approved_source_end'],
                   row['rotation'], row['move'], row.get('camera_view'))
            groups.setdefault(key, []).append(row)
        selected = []
        for rows in groups.values():
            rows.sort(key=lambda r: r['start'])
            indices = {round(i * (len(rows) - 1) / max(1, samples_per_window - 1))
                       for i in range(min(samples_per_window, len(rows)))}
            indices.add(max(range(len(rows)), key=lambda j: rows[j]['score']))
            selected.extend(rows[j] for j in sorted(indices))
        selected.sort(key=lambda r: (r['capture_epoch'], r['start']))
        by_duration[duration] = selected
        print(f'{duration:.3f}s: {len(reviewed)} reviewed / {len(selected)} search windows', flush=True)
    return [by_duration[round(s['duration'], 8)] for s in slots]


def _opposite_screen(a, b):
    if not (a.get('vehicle_present') and b.get('vehicle_present')):
        return False
    directions = {a.get('vehicle_screen_direction'), b.get('vehicle_screen_direction')}
    return directions in ({'left', 'right'}, {'toward', 'away'})


def _transition_allowed(row, timeline, vehicle_end, gap_s, max_visual_run):
    if any(r['video'] == row['video'] and row['start'] < r['end'] + gap_s - 1e-7
           and row['end'] + gap_s > r['start'] + 1e-7 for r in timeline):
        return False
    if row.get('vehicle_moving') and row['capture_epoch'] - row['duration'] / 2 < vehicle_end - 1e-7:
        return False
    if timeline and row['rotation'] and timeline[-1]['rotation']:
        return False
    run = 1
    for previous in reversed(timeline):
        if previous['scene_group'] != row['scene_group']:
            break
        run += 1
    if max_visual_run and run > max_visual_run:
        return False
    if timeline and _opposite_screen(timeline[-1], row):
        previous = timeline[-1]
        # A side change around the same moving car is confusing without a
        # cutaway. Different views/sequences retain a soft continuity penalty.
        if (previous.get('vehicle_sequence') == row.get('vehicle_sequence')
                and previous.get('camera_view') == row.get('camera_view')):
            return False
    return True


def plan_windows(options, slots, beam_width=120, gap_s=0.4, max_visual_run=2,
                 required_scene_groups=(), closing_scene_group=None, max_rotation_fraction=.30):
    """Keep driving chronology while nearby landscape takes may change order."""
    if len(options) != len(slots) or not all(options):
        raise ValueError('every music slot needs reviewed stable windows')
    required = set(required_scene_groups)
    if closing_scene_group:
        required.add(closing_scene_group)
        options = list(options)
        options[-1] = [r for r in options[-1] if r['scene_group'] == closing_scene_group]
    available = {r['scene_group'] for rows in options for r in rows}
    if required - available or not all(options):
        raise ValueError('required scenery does not have a reviewed stable window')
    # score, timeline, end epoch of last vehicle shot
    beam = [(0.0, [], -math.inf)]
    for si, (slot, rows) in enumerate(zip(slots, options)):
        target = (slot['start_frame'] + slot['end_frame']) / (2 * slots[-1]['end_frame'])
        expanded = []
        for accumulated, timeline, vehicle_end in beam:
            uses = Counter(r['video'] for r in timeline)
            groups = Counter(r['scene_group'] for r in timeline)
            previous = timeline[-1] if timeline else None
            for row in rows:
                if not _transition_allowed(row, timeline, vehicle_end, gap_s, max_visual_run):
                    continue
                if sum(r['rotation'] for r in timeline) + row['rotation'] > math.floor(len(slots) * max_rotation_fraction):
                    continue
                if len(required - set(groups) - {row['scene_group']}) > len(slots) - si - 1:
                    continue
                next_vehicle = max(vehicle_end, row['capture_epoch'] + row['duration'] / 2) if row.get('vehicle_moving') else vehicle_end
                if closing_scene_group and si < len(slots) - 1:
                    if not any(_transition_allowed(last, timeline + [row], next_vehicle, gap_s,
                                                   max_visual_run if si == len(slots) - 2 else None)
                               for last in options[-1]):
                        continue
                score = float(row['score']) + float(row.get('preference', 0))
                score -= 7 * abs(row.get('source_order_progress', target) - target)
                score -= .12 * uses[row['video']] + .35 * groups[row['scene_group']]
                score += .8 if not groups[row['scene_group']] else 0
                score -= .55 if row['rotation'] else 0
                if previous:
                    score -= 1.1 if previous['scene_group'] == row['scene_group'] else 0
                    score += .45 if previous.get('camera_view') != row.get('camera_view') else 0
                    score -= .7 if _opposite_screen(previous, row) else 0
                    if (previous.get('vehicle_present') and row.get('vehicle_present')
                            and not _opposite_screen(previous, row)):
                        score += .3
                    if (previous.get('vehicle_moving') and row.get('vehicle_moving')
                            and previous.get('camera_view') == row.get('camera_view')
                            and not previous['rotation'] and not row['rotation']):
                        a, b = previous.get('shot_bearing_deg'), row.get('shot_bearing_deg')
                        if a is not None and b is not None:
                            # This is observed camera travel continuity, not an
                            # estimate of a car heading or a missing KML route.
                            score += .7 * math.cos(math.radians(a - b))
                selected = dict(row, **slot, route_order_mode='no_track')
                selected['duration'] = (slot['end_frame'] - slot['start_frame']) / 30
                expanded.append((accumulated + score, timeline + [selected], next_vehicle))
        if not expanded:
            raise ValueError(f'no feasible no-track edit at slot {si + 1}/{len(slots)}')
        expanded.sort(key=lambda item: item[0], reverse=True)
        beam, seen = [], set()
        for item in expanded:
            row = item[1][-1]
            signature = (row['video'], round(row['start'] / 3), row['scene_group'],
                         round(item[2] / 5) if math.isfinite(item[2]) else None,
                         tuple(sorted({r['scene_group'] for r in item[1]})))
            if signature in seen:
                continue
            seen.add(signature); beam.append(item)
            if len(beam) == beam_width:
                break
        print(f'Planned slot {si + 1}/{len(slots)}; beam {len(beam)}', flush=True)
    return beam[0][1]


def check_plan(plan):
    """Additional no-track/visual/car checks before the generic render checks."""
    if plan.get('track') or plan.get('route_mode') != 'no_track' or plan.get('allow_reverse'):
        raise ValueError('this plan must have no track and natural playback')
    timeline = plan['timeline']
    if not timeline:
        raise ValueError('empty timeline')
    previous_frame = 0
    vehicle_end = -math.inf
    for i, row in enumerate(timeline):
        if row.get('reversed') or not row.get('quality', {}).get('ok'):
            raise ValueError('a shot is reversed or fails motion quality')
        if row.get('vehicle_present') and not isinstance(row.get('vehicle_moving'), bool):
            raise ValueError('a vehicle shot lacks its driving/parking review')
        if row.get('hike_bearing_deg') is not None or row.get('direction_error_deg') is not None or 'route_s' in row:
            raise ValueError('no-track plans cannot claim route progress/direction')
        review = row.get('visual_review', {})
        if (review.get('annotation_scope') != 'window'
                or row['start'] < review.get('approved_source_start', math.inf) - 1e-7
                or row['end'] > review.get('approved_source_end', -math.inf) + 1e-7):
            raise ValueError('a source window extends beyond visual review')
        if row['start'] < row['stable_segment_start'] - 1e-7 or row['end'] > row['stable_segment_end'] + 1e-7:
            raise ValueError('a source window extends beyond its v6 stable segment')
        if (abs(row.get('source_start', row['start']) - row['start']) > 1e-7
                or abs(row.get('source_end', row['end']) - row['end']) > 1e-7):
            raise ValueError('render source bounds differ from the reviewed bounds')
        if (row['start_frame'] != previous_frame or row['end_frame'] <= row['start_frame']
                or abs(row['end'] - row['start'] - (row['end_frame'] - row['start_frame']) / 30) > 1e-7):
            raise ValueError('source duration and continuous output frame grid differ')
        if not _transition_allowed(row, timeline[:i], vehicle_end, plan.get('source_gap_s', .4),
                                   plan.get('max_consecutive_visual_group', 2)):
            raise ValueError('source reuse, vehicle order, or transition fails')
        if row.get('motion_starts_immediately') is False:
            raise ValueError('an approved shot waits before moving')
        if row.get('vehicle_moving'):
            vehicle_end = max(vehicle_end, row['capture_epoch'] + row['duration'] / 2)
        previous_frame = row['end_frame']
    if previous_frame != plan['music_analysis']['total_frames']:
        raise ValueError('timeline does not cover the complete music frame grid')
    if set(plan.get('required_scene_groups', [])) - {r['scene_group'] for r in timeline}:
        raise ValueError('required scenery is missing')
    if plan.get('closing_scene_group') and timeline[-1]['scene_group'] != plan['closing_scene_group']:
        raise ValueError('the reviewed closing scenery is missing')
    rotation_fraction = plan.get('policy', {}).get('max_rotation_fraction')
    if rotation_fraction is not None and sum(r['rotation'] for r in timeline) > math.floor(len(timeline) * rotation_fraction):
        raise ValueError('rotation count exceeds the planned fraction')
    return dict(montage.plan_summary(timeline), no_track=True,
                vehicle_shots=sum(bool(r.get('vehicle_present')) for r in timeline),
                moving_vehicle_shots=sum(bool(r.get('vehicle_moving')) for r in timeline),
                vehicle_playback='forward', moving_vehicle_capture_order='forward',
                all_visual_windows_enclosed=True, geolocation='drone_camera_gps_only')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--music-analysis', type=Path, required=True)
    ap.add_argument('--slots', type=Path, required=True)
    ap.add_argument('--annotations', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--create-cache', action='store_true')
    ap.add_argument('--extend-cache', action='store_true', help='add new durations only for --sources')
    ap.add_argument('--sources', nargs='*', type=Path)
    ap.add_argument('--beam-width', type=int, default=120)
    ap.add_argument('--max-consecutive-visual-group', type=int, default=2)
    ap.add_argument('--require-scene-group', action='append', default=[])
    ap.add_argument('--closing-scene-group')
    ap.add_argument('-o', '--output', type=Path, required=True)
    args = ap.parse_args()
    music = json.loads(args.music_analysis.read_text(encoding='utf-8'))
    slot_doc = json.loads(args.slots.read_text(encoding='utf-8'))
    slots = slot_doc['slots'] if isinstance(slot_doc, dict) else slot_doc
    if args.create_cache:
        if not args.sources: ap.error('--create-cache requires selected --sources')
        cache = create_cache(args.sources, slots, args.cache)
    elif args.extend_cache:
        if not args.sources: ap.error('--extend-cache requires selected --sources')
        cache = extend_cache(args.sources, slots, args.cache)
    else:
        cache = json.loads(args.cache.read_text(encoding='utf-8'))
    annotations = json.loads(args.annotations.read_text(encoding='utf-8'))
    options = options_from_cache(cache, slots, annotations)
    timeline = plan_windows(options, slots, args.beam_width,
                            max_visual_run=args.max_consecutive_visual_group,
                            required_scene_groups=args.require_scene_group,
                            closing_scene_group=args.closing_scene_group)
    plan = dict(music_analysis=music, slots=slots, timeline=timeline, track=None,
                source_scope=cache['sources'],
                variant='_v6', route_mode='no_track', no_track=True, allow_reverse=False,
                annotations=str(args.annotations.resolve()), source_gap_s=.4,
                max_consecutive_visual_group=args.max_consecutive_visual_group,
                required_scene_groups=args.require_scene_group,
                closing_scene_group=args.closing_scene_group,
                selection='reviewed_landscape_and_forward_vehicle_sequence',
                policy=dict(moving_vehicle_capture_order='forward', landscape_order='editorial',
                            parked_vehicle_views='editorial_forward_playback',
                            geolocation='drone_camera_gps_only', subject_geolocation='unknown',
                            direction_reference='unknown_no_track', all_playback='forward',
                            max_rotation_fraction=.30))
    plan['summary'] = check_plan(plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(plan['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

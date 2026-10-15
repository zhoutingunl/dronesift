"""Plan and render a music-aligned montage from stable v6 cuts.

Example: .venv/bin/python scripts/build_montage.py MUSIC.mp3 SOURCE_DIR \
    --track HIKE.kml --annotations SCENES.json -o output/montage.mp4
An audited --plan JSON reuses its exact source windows and output frame grid.
"""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dronesift import montage, render
from dronesift.playback import validate_playback_speed


def route_plot(plan, path):
    if not plan.get('track'): return
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    track = montage.load_timed_track(plan['track'])
    scale = np.array([111320 * math.cos(math.radians(track.points[:, 1].mean())), 111320])
    origin = track.points[0]
    xy = (track.points - origin) * scale
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(xy[:, 0], xy[:, 1], color='#adb6c2', lw=1.5, label='KML hiking route')
    rows = plan['timeline']
    hikers = (np.array([[r['route_lon'], r['route_lat']] for r in rows]) - origin) * scale
    points = (np.array([[r.get('camera_lon', r['route_lon']),
                        r.get('camera_lat', r['route_lat'])] for r in rows]) - origin) * scale
    ax.plot(points[:, 0], points[:, 1], color='#287baf', lw=1.3, alpha=.65,
            label='Selected camera locations')
    for i, (r, p, hiker) in enumerate(zip(rows, points, hikers), 1):
        ax.plot([p[0], hiker[0]], [p[1], hiker[1]], color='#84909c', lw=.8, ls=':', alpha=.6)
        ax.scatter(*hiker, color='#84909c', s=16, marker='x',
                   label='Hiker position at capture time' if i == 1 else None)
        color = '#7a52be' if r.get('reversed') else '#d36139' if r['rotation'] else '#287baf'
        ax.scatter(*p, color=color, s=25, zorder=3)
        ax.annotate(str(i), p, xytext=(4, 5), textcoords='offset points', fontsize=8)
        heading = r.get('hike_bearing_deg')
        if heading is not None:
            arrow = np.array([math.sin(math.radians(heading)), math.cos(math.radians(heading))]) * 80
            ax.annotate('', xy=p + arrow, xytext=p, arrowprops=dict(arrowstyle='->', color='#287baf', lw=.9))
    ax.set_aspect('equal'); ax.set_xlabel('East from route start (m)'); ax.set_ylabel('North (m)')
    ax.set_title(f"{len(rows)} shots around the hiking route; arrows = hike direction")
    ax.legend(); ax.grid(alpha=.15); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def save_plan(plan, output):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    for row in plan['timeline']:
        final_cuts = Path(row['video']).parent / (Path(row['video']).stem + '_cuts_v6')
        if final_cuts.is_dir(): row['cuts_dir'] = str(final_cuts)
    plan['summary'] = montage.plan_summary(plan['timeline'])
    plan['output'] = str(output)
    edit = output.with_name(output.stem + '_edit.json')
    edit.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    route_plot(plan, output.with_name(output.stem + '_route.png'))
    return edit


def validate_plan(plan, music_path):
    audio = render.music_info(music_path)
    allow_reverse = bool(plan.get('allow_reverse', False))
    rendered_timeline = render.frame_timeline(plan['timeline'], audio['duration'], 30,
                                               allow_reverse=allow_reverse)
    capture_order = plan.get('capture_order')
    if capture_order not in (None, 'strict'):
        raise ValueError('unknown capture ordering mode')
    summary = montage.plan_summary(plan['timeline'])
    if summary['consecutive_rotations']:
        raise ValueError('rotation shots must be separated')
    if summary['reversed_shots'] and not allow_reverse:
        raise ValueError('reverse playback requires an explicit plan option')
    if not summary['all_motion_gates_passed']:
        raise ValueError('a shot fails the motion quality gate')
    if summary['opposed_straight_shots']:
        raise ValueError('a directional shot travels against the timed hike direction')
    previous = None
    route_mode = plan.get('route_mode', 'strict')
    if route_mode not in ('strict', 'regional', 'no_track'):
        raise ValueError('unknown route planning mode')
    if route_mode == 'no_track' and plan.get('track'):
        raise ValueError('a no_track plan cannot declare a hiking track')
    farthest = -math.inf
    used = []
    flight_counts = {}
    group_counts = {}
    group_run = 0
    previous_group = None
    previous_capture_end = None
    for row, rendered in zip(plan['timeline'], rendered_timeline):
        if capture_order == 'strict':
            if rendered['reversed']:
                raise ValueError('strict capture order requires forward playback')
            if (rendered['source_consumed_start'] < row['stable_segment_start'] - 1e-6
                    or rendered['source_consumed_end'] > row['stable_segment_end'] + 1e-6):
                raise ValueError('actual playback source extends beyond its stable interval')
            # Use the actual source consumed at this playback rate, including
            # parked cars and landscapes. Edited midpoint metadata cannot
            # make a later parking shot precede an earlier driving shot.
            source_epoch = montage.capture_datetime(row['video']).timestamp()
            capture_start = source_epoch + rendered['source_consumed_start']
            capture_end = source_epoch + rendered['source_consumed_end']
            if previous_capture_end is not None and capture_start < previous_capture_end - 1e-6:
                raise ValueError('shots do not follow strict source capture order')
            previous_capture_end = capture_end
        if route_mode == 'no_track' and (row.get('hike_bearing_deg') is not None
                                        or row.get('direction_error_deg') is not None):
            raise ValueError('a no_track plan cannot claim hiking direction alignment')
        if row['start'] < row['stable_segment_start'] - 1e-6 or row['end'] > row['stable_segment_end'] + 1e-6:
            raise ValueError('a selected window extends beyond its stable interval')
        if row.get('reversed'):
            review = row.get('reverse_safety_review', {})
            if not (review.get('allow_reverse') is True and review.get('physics_reversible') is True
                    and review.get('person_moving') is not True
                    and (review.get('person_present') is False or review.get('person_moving') is False)):
                raise ValueError('a reversed shot lacks its explicit visual safety review')
            if review.get('annotation_scope') == 'window':
                if (row['start'] < review.get('approved_source_start', 0) - 1e-6
                        or row['end'] > review.get('approved_source_end', math.inf) + 1e-6):
                    raise ValueError('a reversed shot extends beyond its reviewed source window')
            elif review.get('annotation_scope') != 'whole_source':
                raise ValueError('a reversed shot has no recorded visual review scope')
        if route_mode == 'strict':
            if previous and (row['capture_epoch'] <= previous['capture_epoch'] or row['route_s'] + 3 < previous['route_s']):
                raise ValueError('shots do not follow capture/route progress')
        elif route_mode == 'regional' and row['route_s'] + float(plan.get('max_backtrack_m', 250)) < farthest - 1e-6:
            raise ValueError('a regional insert falls outside the current route region')
        for other in used:
            if row['video'] == other['video'] and row['start'] < other['end'] - 1e-6 and row['end'] > other['start'] + 1e-6:
                raise ValueError('selected source windows overlap')
        if route_mode != 'no_track':
            farthest = max(farthest, row['route_s'])
        flight_counts[row['video']] = flight_counts.get(row['video'], 0) + 1
        group = row['scene_group']
        group_counts[group] = group_counts.get(group, 0) + 1
        group_run = group_run + 1 if group == previous_group else 1
        cap = plan.get('max_consecutive_visual_group')
        horizon = plan.get('visual_group_limit_slots')
        if cap and (horizon is None or len(used) < horizon) and group_run > cap:
            raise ValueError('consecutive scene repetition exceeds the planned limit')
        previous_group = group
        if plan.get('max_per_flight') and flight_counts[row['video']] > plan['max_per_flight']:
            raise ValueError('a source exceeds its planned shot limit')
        if plan.get('max_per_visual_group') and group_counts[group] > plan['max_per_visual_group']:
            raise ValueError('a scene exceeds its planned shot limit')
        used.append(row)
        previous = row
    for row, rendered in zip(plan['timeline'], rendered_timeline):
        validate_playback_speed(row, source_start=rendered['source_consumed_start'],
                                source_end=rendered['source_consumed_end'], output_fps=30,
                                require_assessment=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('music', type=Path); ap.add_argument('sources', nargs='*', type=Path)
    ap.add_argument('--track', type=Path); ap.add_argument('--annotations', type=Path)
    ap.add_argument('--plan', type=Path, help='reuse an already audited source-window timeline')
    ap.add_argument('--variant', default='_v6'); ap.add_argument('--bar-pattern')
    ap.add_argument('--route-mode', choices=['strict', 'regional'],
                    help='regional selects nearby scenery within the current route region (default for new plans)')
    ap.add_argument('--max-backtrack-m', type=float, default=250)
    ap.add_argument('--max-per-flight', type=int)
    ap.add_argument('--max-per-visual-group', type=int)
    ap.add_argument('--max-consecutive-visual-group', type=int)
    ap.add_argument('--visual-group-limit-slots', type=int)
    ap.add_argument('--allow-reverse', action='store_true',
                    help='allow direction correction only in explicitly reviewed reversible windows')
    ap.add_argument('--plan-only', action='store_true'); ap.add_argument('--workdir', type=Path)
    ap.add_argument('-o', '--output', type=Path, required=True)
    args = ap.parse_args()
    if args.plan:
        plan = json.loads(args.plan.read_text(encoding='utf-8'))
        if Path(plan['music_analysis']['music']).resolve() != args.music.resolve():
            raise ValueError('the plan was generated for a different music file')
        if args.route_mode:
            plan['route_mode'] = args.route_mode
        if args.allow_reverse:
            plan['allow_reverse'] = True
    else:
        if not args.sources or not args.track: ap.error('sources and --track are required without --plan')
        music = montage.analyse_music(args.music)
        pattern = tuple(map(int, args.bar_pattern.split(','))) if args.bar_pattern else ((3, 3, 2) if music['duration'] > 120 else (2, 2, 3))
        slots = montage.make_slots(music, pattern=pattern)
        notes = json.loads(args.annotations.read_text(encoding='utf-8')) if args.annotations else None
        candidates = montage.load_candidates(args.sources, variant=args.variant, track_path=args.track)
        route_mode = args.route_mode or 'regional'
        timeline = montage.plan_sequence(candidates, slots, notes, max_direction_error_deg=90,
                                          route_mode=route_mode, max_backtrack_m=args.max_backtrack_m,
                                          allow_reverse=args.allow_reverse,
                                          max_per_flight=args.max_per_flight,
                                          max_per_visual_group=args.max_per_visual_group,
                                          max_consecutive_visual_group=args.max_consecutive_visual_group,
                                          visual_group_limit_slots=args.visual_group_limit_slots)
        plan = dict(music_analysis=music, track=str(args.track.resolve()), variant=args.variant,
                    slots=slots, timeline=timeline, bar_pattern=pattern, route_mode=route_mode,
                    max_backtrack_m=args.max_backtrack_m, allow_reverse=args.allow_reverse,
                    max_per_flight=args.max_per_flight, max_per_visual_group=args.max_per_visual_group,
                    max_consecutive_visual_group=args.max_consecutive_visual_group,
                    visual_group_limit_slots=args.visual_group_limit_slots)
    print(json.dumps(validate_plan(plan, args.music), ensure_ascii=False), flush=True)
    save_plan(plan, args.output)
    if not args.plan_only:
        result = render.render_montage(plan['timeline'], args.music, args.output, args.workdir,
                                       allow_reverse=bool(plan.get('allow_reverse', False)))
        print(json.dumps(result['validation'], ensure_ascii=False), flush=True)


if __name__ == '__main__': main()

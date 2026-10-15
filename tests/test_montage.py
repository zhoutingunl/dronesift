"""Narrative and music timing regressions for the montage planner."""
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from dronesift.montage import (analyse_music, capture_datetime, load_timed_track,
                              make_slots, plan_sequence, TimedTrack, SourceCandidate,
                              _reverse_review)


def test_slots_cover_audio_without_beat_phase_offset_or_rounding_drift():
    music = dict(fps=30, total_frames=1569, beat_period=60 / 102,
                 bars=list(np.arange(1.45, 52.296, 4 * 60 / 102)))
    slots = make_slots(music)
    assert slots[0]['start_frame'] == 0
    assert slots[-1]['end_frame'] == 1569
    assert all(a['end_frame'] == b['start_frame'] for a, b in zip(slots, slots[1:]))
    assert sum(s['end_frame'] - s['start_frame'] for s in slots) == 1569
    assert all(s['boundary_error_s'] <= 1 / 60 + 1e-9 for s in slots)


def test_pulse_tempo_uses_lag_zero_and_continuous_period(monkeypatch):
    sr, duration, bpm = 22050, 24.0, 102.0
    y = np.zeros(round(duration * sr), dtype='<f4')
    length = round(0.025 * sr)
    pulse = np.sin(2 * np.pi * 2300 * np.arange(length) / sr) * np.hanning(length)
    for beat in np.arange(0.28, duration - 0.1, 60 / bpm):
        i = round(beat * sr)
        y[i:i + length] += pulse
    monkeypatch.setattr('dronesift.montage.subprocess.run',
                        lambda *args, **kwargs: SimpleNamespace(stdout=y.tobytes()))
    result = analyse_music('synthetic.mp3')
    assert result['bpm'] == pytest.approx(bpm, abs=0.3)
    assert result['total_frames'] == 720
    assert len(result['beats']) == pytest.approx(duration / (60 / bpm), abs=1)


def test_track_uses_timezone_and_keeps_return_trip_in_chronological_order(tmp_path):
    kml = tmp_path / 'track.kml'
    kml.write_text('''<kml xmlns="http://www.opengis.net/kml/2.2"
      xmlns:gx="http://www.google.com/kml/ext/2.2"><Document><gx:Track>
      <when>2026-10-01T01:14:06Z</when><when>2026-10-01T01:15:06Z</when>
      <when>2026-10-01T01:16:06Z</when><gx:coord>117 43 0</gx:coord>
      <gx:coord>117.001 43 0</gx:coord><gx:coord>117 43 0</gx:coord>
      </gx:Track></Document></kml>''')
    track = load_timed_track(kml)
    outward = track.locate(capture_datetime('DJI_20261001091506_001_D.MP4'))
    returned = track.locate(capture_datetime('DJI_20261001091606_001_D.MP4'))
    assert outward['capture_offset_s'] == 60
    assert returned['capture_offset_s'] == 120
    assert returned['route_s'] > outward['route_s']


class FakeCandidate:
    def __init__(self, i, rotation=False, score=1):
        self.video = f'flight_{i}.mp4'
        self.capture = datetime.fromtimestamp(i * 10)
        self.track = None
        self.seg = dict(start=0, end=9)
        self.i, self.rotation, self.score = i, rotation, score

    def windows(self, duration, annotations=None):
        return [dict(video=self.video, start=1, end=1 + duration,
                     capture_epoch=self.i * 10, route_s=self.i * 10,
                     rotation=self.rotation, reversed=False,
                     camera_lon=117 + self.i / 1000, camera_lat=43,
                     scene_id=self.i, scene_group=f'scene_{self.i}',
                     quality=dict(ok=True), move='orbit' if self.rotation else 'push',
                     direction_error_deg=0, score=self.score)]


def test_planner_avoids_adjacent_rotations_and_keeps_source_time_forward():
    candidates = [FakeCandidate(1, True, 5), FakeCandidate(2, True, 5),
                  FakeCandidate(3), FakeCandidate(4, True, 5),
                  FakeCandidate(5), FakeCandidate(6)]
    slots = [dict(slot=i, start_frame=i * 120, end_frame=(i + 1) * 120,
                  duration=4, slot_start=i * 4) for i in range(3)]
    timeline = plan_sequence(candidates, slots)
    assert len(timeline) == 3
    assert all(a['capture_epoch'] < b['capture_epoch'] for a, b in zip(timeline, timeline[1:]))
    assert not any(a['rotation'] and b['rotation'] for a, b in zip(timeline, timeline[1:]))
    assert not any(r['reversed'] for r in timeline)


def test_suffix_preserves_enough_later_footage_even_with_tiny_beam():
    candidates = [FakeCandidate(1), FakeCandidate(2), FakeCandidate(3),
                  FakeCandidate(4, score=100)]
    slots = [dict(slot=i, start_frame=i * 120, end_frame=(i + 1) * 120,
                  duration=4, slot_start=i * 4) for i in range(3)]
    timeline = plan_sequence(candidates, slots, beam_width=1)
    assert len(timeline) == 3
    assert timeline[-1]['video'] == 'flight_4.mp4'


def test_hard_direction_constraint_excludes_attractive_opposing_shot():
    opposed = FakeCandidate(1, score=100)
    original_windows = opposed.windows
    opposed.windows = lambda duration, annotations=None: [
        dict(row, directional=True, direction_error_deg=170)
        for row in original_windows(duration, annotations)]
    candidates = [opposed, FakeCandidate(2), FakeCandidate(3), FakeCandidate(4)]
    slots = [dict(slot=i, start_frame=i * 120, end_frame=(i + 1) * 120,
                  duration=4, slot_start=i * 4) for i in range(3)]
    timeline = plan_sequence(candidates, slots, max_direction_error_deg=90)
    assert all(row['video'] != 'flight_1.mp4' for row in timeline)


def regional_candidate(i, route_s, capture_epoch=None, duration=None, **row_fields):
    candidate = FakeCandidate(i)
    candidate.track = SimpleNamespace(locate=lambda capture: dict(route_s=route_s))
    original = candidate.windows

    def windows(requested, annotations=None):
        if duration is not None and requested != duration:
            return []
        return [dict(row, route_s=route_s,
                     capture_epoch=i * 10 if capture_epoch is None else capture_epoch,
                     **row_fields) for row in original(requested, annotations)]

    candidate.windows = windows
    return candidate


def slots_for(durations):
    frame = 0
    slots = []
    for i, duration in enumerate(durations):
        end = frame + round(duration * 30)
        slots.append(dict(slot=i, start_frame=frame, end_frame=end, duration=duration,
                          slot_start=frame / 30))
        frame = end
    return slots


def test_regional_inserts_reorder_takes_in_same_area_without_reversing_playback():
    candidates = [regional_candidate(1, 400, capture_epoch=20, duration=3),
                  regional_candidate(2, 300, capture_epoch=10, duration=4)]
    timeline = plan_sequence(candidates, slots_for([3, 4]), route_mode='regional')
    assert timeline[1]['capture_epoch'] < timeline[0]['capture_epoch']
    assert timeline[1]['route_backtrack_m'] == 100
    assert not any(r['reversed'] for r in timeline)
    with pytest.raises(ValueError, match='cannot fill'):
        plan_sequence(candidates, slots_for([3, 4]), route_mode='strict')


def test_regional_backtracking_is_bounded_from_furthest_progress_not_last_shot():
    candidates = [regional_candidate(1, 400, duration=3),
                  regional_candidate(2, 250, duration=4),
                  regional_candidate(3, 100, duration=5)]
    with pytest.raises(ValueError, match='cannot fill'):
        plan_sequence(candidates, slots_for([3, 4, 5]), route_mode='regional', max_backtrack_m=200)


def test_regional_keeps_off_track_camera_scenery_without_a_distance_cutoff():
    candidates = [regional_candidate(1, 0, duration=3, camera_route_distance_m=1800),
                  regional_candidate(2, 300, duration=4, camera_route_distance_m=600)]
    timeline = plan_sequence(candidates, slots_for([3, 4]), route_mode='regional')
    assert timeline[0]['camera_route_distance_m'] == 1800
    assert len(timeline) == 2


def test_regional_cannot_reuse_overlapping_footage_around_a_cutaway():
    repeated = regional_candidate(1, 100)
    original = repeated.windows
    repeated.windows = lambda duration, annotations=None: original(duration, annotations) if duration in (3, 5) else []
    cutaway = regional_candidate(2, 200, duration=4)
    with pytest.raises(ValueError, match='cannot fill'):
        plan_sequence([repeated, cutaway], slots_for([3, 4, 5]),
                      route_mode='regional', max_per_flight=2)


def test_optional_visual_group_limit_preserves_distinct_short_montage_scenery():
    candidates = [regional_candidate(1, 0, duration=3, scene_group='valley'),
                  regional_candidate(2, 100, duration=4, scene_group='ridge'),
                  regional_candidate(3, 200, duration=5, scene_group='valley', score=100),
                  regional_candidate(4, 300, duration=5, scene_group='cliff')]
    timeline = plan_sequence(candidates, slots_for([3, 4, 5]), route_mode='regional',
                             max_per_visual_group=1)
    assert [r['scene_group'] for r in timeline] == ['valley', 'ridge', 'cliff']


def test_regional_resource_guard_keeps_unique_future_scene_with_a_tiny_beam():
    tempting = regional_candidate(1, 200, score=100, scene_group='final_cliff')
    original = tempting.windows
    tempting.windows = lambda duration, annotations=None: original(duration, annotations) if duration in (3, 5) else []
    candidates = [tempting,
                  regional_candidate(2, 0, duration=3, scene_group='valley'),
                  regional_candidate(3, 100, duration=4, scene_group='ridge')]
    timeline = plan_sequence(candidates, slots_for([3, 4, 5]), route_mode='regional',
                             max_per_flight=1, max_per_visual_group=1, beam_width=1)
    assert [r['scene_group'] for r in timeline] == ['valley', 'ridge', 'final_cliff']


def test_regional_reserves_a_complete_closing_window_not_fragmented_seconds():
    fragmented = regional_candidate(1, 200, score=100)
    original = fragmented.windows

    def windows(duration, annotations=None):
        if duration == 3:
            return [dict(r, start=2, end=5) for r in original(duration, annotations)]
        if duration == 5:
            return [dict(r, start=a, end=a + 5) for r in original(duration, annotations) for a in (0, 5)]
        return []

    fragmented.windows = windows
    candidates = [fragmented, regional_candidate(2, 0, duration=3),
                  regional_candidate(3, 100, duration=4)]
    timeline = plan_sequence(candidates, slots_for([3, 4, 5]), route_mode='regional',
                             max_per_flight=2, beam_width=1)
    assert timeline[0]['video'] == 'flight_2.mp4'
    assert timeline[-1]['video'] == 'flight_1.mp4'


def test_intro_visual_run_limit_allows_distinct_views_later_in_the_film():
    candidates = [regional_candidate(i, (i - 1) * 100, duration=i + 2, scene_group='forest')
                  for i in range(1, 7)]
    candidates.append(regional_candidate(7, 200, duration=5, scene_group='ridge'))
    timeline = plan_sequence(candidates, slots_for([3, 4, 5, 6, 7, 8]), route_mode='regional',
                             max_consecutive_visual_group=2, visual_group_limit_slots=3)
    assert [r['scene_group'] for r in timeline] == ['forest', 'forest', 'ridge', 'forest', 'forest', 'forest']


def test_regional_requires_geographic_progress_instead_of_treating_seconds_as_metres():
    with pytest.raises(ValueError, match='requires a timed hiking track'):
        plan_sequence([FakeCandidate(1)], slots_for([3]), route_mode='regional')


def test_camera_projection_uses_timed_route_branch_and_does_not_locate_subject():
    points = np.array([[117.0, 43], [117.01, 43], [117.0, 43]])
    track = TimedTrack(points, np.array([0, 100, 200]), np.array([0, 800, 1600]), datetime.now())
    projected = track.project_camera(117.005, 43.001, reference_s=1200, region_span_m=250)
    assert projected['camera_route_s'] == pytest.approx(1200)
    assert projected['camera_route_distance_m'] == pytest.approx(111.32, abs=0.01)
    assert projected['subject_geolocation'] == 'unknown'


def test_reverse_requires_complete_affirmative_review_and_rejects_moving_people():
    row = dict(video='/some/landscape.mp4', start=2, end=6)
    approved = dict(video='landscape.mp4', start=2, end=6, allow_reverse=True,
                    physics_reversible=True, person_present=False)
    assert _reverse_review(dict(row), dict(clips=[approved]))
    assert not _reverse_review(dict(row), dict(clips=[dict(approved, end=5)]))
    assert not _reverse_review(dict(row), dict(clips=[dict(approved, physics_reversible=False)]))
    assert not _reverse_review(dict(row), dict(clips=[dict(approved, person_moving=True)]))
    assert not _reverse_review(dict(row), dict(clips=[dict(approved, person_present=True)]))
    assert not _reverse_review(dict(row), dict(sources={'landscape.mp4': {'person_observed': False}}))


def test_reverse_is_opt_in_and_only_supplies_needed_direction_after_review():
    candidate = FakeCandidate(1)
    original = candidate.windows
    candidate.windows = lambda duration, annotations=None: [
        dict(row, directional=True, direction_error_deg=170, shot_bearing_deg=170,
             reverse_safety_review=dict(allow_reverse=True, physics_reversible=True, person_present=False),
             reverse_quality=dict(ok=True)) for row in original(duration, annotations)]
    with pytest.raises(ValueError, match='cannot fill'):
        plan_sequence([candidate], slots_for([3]), max_direction_error_deg=90)
    row = plan_sequence([candidate], slots_for([3]), max_direction_error_deg=90, allow_reverse=True)[0]
    assert row['reversed'] is True
    assert row['direction_error_deg'] == 10
    assert row['shot_bearing_deg'] == 350
    assert row['move'] == 'pull'
    assert row['end'] - row['start'] == (row['end_frame'] - row['start_frame']) / 30


def test_reverse_quality_rechecks_original_tail_as_playback_start():
    fps, n = 30, 90
    feats = {key: np.zeros(n) for key in ('yaw_rate', 'g_yaw', 'vz', 'v_lat', 'jitter_burst')}
    feats.update(speed_h=np.ones(n), v_fwd=np.ones(n), n_tracks=np.full(n, 500),
                 motion_px=np.r_[np.full(n - 6, 20.0), np.zeros(6)])
    candidate = SourceCandidate('/some/landscape.mp4', fps,
                                dict(start=0, end=3, base='push', label='push'), feats,
                                dict(lon=np.linspace(117, 117.001, n), lat=np.full(n, 43)),
                                '/cuts', datetime.now(), None)
    notes = dict(clips=[dict(video='landscape.mp4', start=0, end=3,
                            allow_reverse=True, physics_reversible=True, person_present=False)])
    row = candidate.windows(3, notes)[0]
    assert row['quality']['ok']
    assert not row['reverse_quality']['ok']
    assert 'still_or_starting_prefix' in row['reverse_quality']['reasons']
    # Reviews applied to one call must not mutate the cached natural window.
    assert 'reverse_safety_review' not in candidate.windows(3)[0]

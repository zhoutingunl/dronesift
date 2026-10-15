import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('build_montage', Path(__file__).parents[1] / 'scripts/build_montage.py')
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def plan(tmp_path, monkeypatch, routes=(100, 150), captures=(200, 100)):
    monkeypatch.setattr(build.render, 'music_info', lambda _: {'duration': len(routes) * 2})
    rows = []
    for i, (route, capture) in enumerate(zip(routes, captures)):
        source = tmp_path / f'{i}.mp4'
        source.write_bytes(b'source placeholder')
        rows.append(dict(video=str(source), start=0, end=2, start_frame=i * 60, end_frame=(i + 1) * 60,
                         stable_segment_start=0, stable_segment_end=10, capture_epoch=capture,
                         route_s=route, scene_id=i, scene_group=str(i), rotation=False,
                         reversed=False, direction_error_deg=10, quality={'ok': True}))
    return dict(timeline=rows, route_mode='regional', max_backtrack_m=250, allow_reverse=False)


def test_regional_allows_capture_reordering(tmp_path, monkeypatch):
    edit = plan(tmp_path, monkeypatch)
    assert build.validate_plan(edit, 'music')['reordered_capture_transitions'] == 1
    edit['route_mode'] = 'strict'
    with pytest.raises(ValueError, match='capture/route progress'):
        build.validate_plan(edit, 'music')


def test_regional_limits_cumulative_backtrack(tmp_path, monkeypatch):
    edit = plan(tmp_path, monkeypatch, routes=(300, 200, 100), captures=(100, 200, 300))
    edit['max_backtrack_m'] = 150
    with pytest.raises(ValueError, match='current route region'):
        build.validate_plan(edit, 'music')


def test_reverse_requires_complete_review_scope(tmp_path, monkeypatch):
    edit = plan(tmp_path, monkeypatch, routes=(100,), captures=(100,))
    edit['allow_reverse'] = True
    row = edit['timeline'][0]
    row['reversed'] = True
    with pytest.raises(ValueError, match='visual safety review'):
        build.validate_plan(edit, 'music')
    row['reverse_safety_review'] = dict(allow_reverse=True, physics_reversible=True,
                                      person_present=False, annotation_scope='window',
                                      approved_source_start=0.5, approved_source_end=2)
    with pytest.raises(ValueError, match='reviewed source window'):
        build.validate_plan(edit, 'music')
    row['reverse_safety_review']['approved_source_start'] = 0
    assert build.validate_plan(edit, 'music')['reversed_shots'] == 1
    row['reverse_safety_review']['person_moving'] = True
    with pytest.raises(ValueError, match='visual safety review'):
        build.validate_plan(edit, 'music')


def test_regional_cannot_reuse_source_frames(tmp_path, monkeypatch):
    edit = plan(tmp_path, monkeypatch)
    edit['timeline'][1]['video'] = edit['timeline'][0]['video']
    with pytest.raises(ValueError, match='source windows overlap'):
        build.validate_plan(edit, 'music')


def test_no_track_allows_reordering_without_inventing_route_progress(tmp_path, monkeypatch):
    edit = plan(tmp_path, monkeypatch)
    edit.update(route_mode='no_track', track=None)
    for row in edit['timeline']:
        row.pop('route_s')
        row['direction_error_deg'] = None
    assert build.validate_plan(edit, 'music')['reordered_capture_transitions'] == 1
    edit['timeline'][1]['video'] = edit['timeline'][0]['video']
    with pytest.raises(ValueError, match='source windows overlap'):
        build.validate_plan(edit, 'music')


def test_no_track_rejects_claimed_hiking_direction(tmp_path, monkeypatch):
    edit = plan(tmp_path, monkeypatch)
    edit.update(route_mode='no_track', track=None)
    with pytest.raises(ValueError, match='hiking direction alignment'):
        build.validate_plan(edit, 'music')
    for row in edit['timeline']:
        row['direction_error_deg'] = None
    edit['track'] = 'unrelated_hike.kml'
    with pytest.raises(ValueError, match='cannot declare a hiking track'):
        build.validate_plan(edit, 'music')


def test_plan_cannot_authorize_acceleration_without_window_assessment(tmp_path, monkeypatch,
                                                                    playback_profile):
    edit = plan(tmp_path, monkeypatch, routes=(100,), captures=(100,))
    row = edit['timeline'][0]
    row.update(playback_speed=1.5, end=3)
    with pytest.raises(ValueError, match='requires a playback_speed_profile'):
        build.validate_plan(edit, 'music')
    row['playback_speed_profile'] = playback_profile(0, 3, maximum=1.25)
    with pytest.raises(ValueError, match='exceeds assessed maximum'):
        build.validate_plan(edit, 'music')
    row['playback_speed_profile'] = playback_profile(0, 3, maximum=1.5)
    assert build.validate_plan(edit, 'music')['shots'] == 1


def chronological_plan(tmp_path, monkeypatch):
    edit = plan(tmp_path, monkeypatch, captures=(100, 200))
    edit.update(route_mode='no_track', track=None, capture_order='strict')
    for i, row in enumerate(edit['timeline']):
        source = tmp_path / f'DJI_20260928150{i}00_000{i}_D.MP4'
        source.touch()
        row.update(video=str(source), direction_error_deg=None, vehicle_moving=i == 0)
        row.pop('route_s')
    return edit


def test_strict_capture_order_includes_parked_and_landscape_inserts(tmp_path, monkeypatch):
    edit = chronological_plan(tmp_path, monkeypatch)
    assert build.validate_plan(edit, 'music')['shots'] == 2
    # Parked/landscape footage taken later cannot be inserted before driving,
    # even when its declared midpoint metadata still appears increasing.
    a, b = edit['timeline']
    a['video'], b['video'] = b['video'], a['video']
    with pytest.raises(ValueError, match='strict source capture order'):
        build.validate_plan(edit, 'music')
    for row in edit['timeline']:
        row['vehicle_moving'] = False
        row['vehicle_present'] = False
    with pytest.raises(ValueError, match='strict source capture order'):
        build.validate_plan(edit, 'music')


def test_strict_capture_order_checks_consumed_end_at_playback_speed(tmp_path, monkeypatch, playback_profile):
    edit = chronological_plan(tmp_path, monkeypatch)
    first, second = edit['timeline']
    first.update(start=0, end=4, playback_speed=2)
    first['playback_speed_profile'] = playback_profile(0, 4)
    # A separate recording can begin inside the accelerated first window:
    # its filename is later, but the actual source intervals still overlap.
    source = tmp_path / 'DJI_20260928150003_0002_D.MP4'
    source.touch()
    second['video'] = str(source)
    with pytest.raises(ValueError, match='strict source capture order'):
        build.validate_plan(edit, 'music')
    second.update(start=1, end=3)
    assert build.validate_plan(edit, 'music')['shots'] == 2


def test_strict_capture_order_uses_render_bounds_and_disallows_reverse(tmp_path, monkeypatch):
    edit = chronological_plan(tmp_path, monkeypatch)
    row = edit['timeline'][0]
    row.update(source_start=9, source_end=11)
    with pytest.raises(ValueError, match='actual playback source'):
        build.validate_plan(edit, 'music')
    row.pop('source_start'); row.pop('source_end')
    edit['allow_reverse'] = True
    row['reversed'] = True
    with pytest.raises(ValueError, match='strict capture order requires forward'):
        build.validate_plan(edit, 'music')

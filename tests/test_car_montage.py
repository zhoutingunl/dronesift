import copy
from types import SimpleNamespace

import numpy as np

from dronesift import car_montage as planner
from dronesift.car_montage import allowed_transition, capture_bounds, music_grid, straight_reviews


def shot(start=0, end=2, video='DJI_20260928150908_0030_D.MP4', group='road', rotation=False, moving=True):
    return dict(video=video, source_start=start, source_end=end, scene_group=group,
                rotation=rotation, reversed=False, vehicle_moving=moving, capture_epoch=-1)


def test_capture_order_includes_parked_vehicles_and_landscapes():
    driving = shot(50, 60)
    assert not allowed_transition([driving], shot(10, 20, group='parked', moving=False))
    assert not allowed_transition([driving], shot(59, 70, group='landscape', moving=False))
    assert allowed_transition([driving], shot(60, 70, group='landscape', moving=False))


def test_actual_capture_datetime_precedes_window_midpoint_metadata():
    first = shot(100, 110)
    later_file = shot(0, 5, video='DJI_20260928151248_0031_D.MP4')
    first['capture_epoch'] = 999999999999
    assert allowed_transition([first], later_file)
    a, b = capture_bounds(first)
    assert b-a == 10


def test_rotation_and_scene_run_limits_apply_to_all_shots():
    first = shot(0, 2, rotation=True, moving=False)
    assert not allowed_transition([first], shot(3, 5, rotation=True))
    second = shot(3, 5)
    assert not allowed_transition([first, second], shot(6, 8))
    assert allowed_transition([first, second], shot(6, 8, group='crater'))
    assert not allowed_transition([], dict(first, reversed=True))


def test_reviewed_straight_scopes_include_splits_and_exclude_curve_decisions():
    review = {'shots': [dict(accelerate=False, straight_windows=[{'start': 1}]),
                        dict(accelerate=True, straight_windows=[{'start': 2}]),
                        dict(accelerate=True, split={'pieces': [dict(straight_windows=[{'start': 3}])]})]}
    assert straight_reviews(review) == [{'start': 2}, {'start': 3}]


def test_music_grid_keeps_every_original_cut_and_does_not_mutate_input():
    rows = [dict(start_frame=i*10, end_frame=(i+1)*10) for i in range(34)]
    rows[19]['end_frame'] = rows[20]['start_frame'] = 3374
    rows[20]['end_frame'] = rows[21]['start_frame'] = 3585
    rows[21]['end_frame'] = rows[22]['start_frame'] = 3797
    source = {'timeline': rows}
    original = copy.deepcopy(source)
    new = music_grid(source)
    assert len(new) == 34
    assert new == rows
    assert source == original


def test_regular_tracking_pan_override_requires_full_actual_source_review(monkeypatch, playback_profile):
    video = 'DJI_20260928150908_0030_D.MP4'
    candidate = SimpleNamespace(video=video, fps=10, cuts_dir='unused', scene_id=1,
        seg=dict(start=0, end=10, base='push', label='push'),
        capture=planner.montage.capture_datetime(video),
        telemetry=dict(lon=np.linspace(116, 116.001, 101), lat=np.full(101, 43.6)),
        feats=dict(yaw_rate=np.full(101, 1.6), g_yaw=np.arange(101)*.16,
                   speed_h=np.full(101, 2), vz=np.zeros(101),
                   v_lat=np.zeros(101), v_fwd=np.full(101, 2)))
    note = dict(video=video, start=0, end=10, scene_group='road',
                vehicle_present=True, vehicle_moving=True)
    slot = dict(start_frame=0, end_frame=180)
    monkeypatch.setattr(planner, 'motion_window_quality', lambda *args:dict(ok=True))
    monkeypatch.setattr(planner, 'playback_speed_profile',
                        lambda f, fps, a, b, **kw: playback_profile(a, b))
    monkeypatch.setattr(planner, 'playback_speed_quality', lambda *args, **kw:dict(ok=True))
    partial = dict(note, end=8, allow_regular_tracking_pan=True)
    assert planner.checked_row(candidate, slot, 0, note, 1.5, [note], [partial]) is None
    complete = dict(partial, end=9)
    row = planner.checked_row(candidate, slot, 0, note, 1.5, [note], [complete])
    assert row['source_end'] == 9 and row['telemetry_rotation'] and not row['rotation']
    assert row['regular_tracking_pan_review']['end'] == 9
    candidate.feats['yaw_rate'][:] = 2.05
    assert planner.checked_row(candidate, slot, 0, note, 1.5, [note], [complete]) is None
    exact_review = dict(complete, max_yaw_mean_abs_dps=2.1)
    row = planner.checked_row(candidate, slot, 0, note, 1.5, [note], [exact_review])
    assert row and row['telemetry_rotation'] and row['yaw_mean_abs_dps'] == 2.05
    # A manual review cannot authorize a fast rotation as a tracking pan.
    candidate.feats['yaw_rate'][:] = 4
    assert planner.checked_row(candidate, slot, 0, note, 1.5, [note],
                               [dict(exact_review, max_yaw_mean_abs_dps=10)]) is None
    candidate.feats['yaw_rate'][:] = 1.6
    monkeypatch.setattr(planner, 'motion_window_quality', lambda *args:dict(ok=False))
    assert planner.checked_row(candidate, slot, 0, note, 1.5, [note], [complete]) is None


def test_straight_driving_review_does_not_override_playback_smoothness_limit(monkeypatch,
                                                                          playback_profile):
    video = 'DJI_20260928150908_0030_D.MP4'
    candidate = SimpleNamespace(video=video, fps=10, cuts_dir='unused', scene_id=1,
        seg=dict(start=0, end=10, base='push', label='push'),
        capture=planner.montage.capture_datetime(video),
        telemetry=dict(lon=np.linspace(116, 116.001, 101), lat=np.full(101, 43.6)),
        feats=dict(yaw_rate=np.zeros(101), g_yaw=np.zeros(101),
                   speed_h=np.full(101, 2), vz=np.zeros(101),
                   v_lat=np.zeros(101), v_fwd=np.full(101, 2)))
    note = dict(video=video, start=0, end=10, scene_group='road', vehicle_moving=True)
    monkeypatch.setattr(planner, 'motion_window_quality', lambda *args: dict(ok=True))
    monkeypatch.setattr(planner, 'playback_speed_profile',
                        lambda f, fps, a, b, **kw: playback_profile(a, b, maximum=1.25))
    monkeypatch.setattr(planner, 'playback_speed_quality', lambda *args, **kw: dict(ok=True))
    slot = dict(start_frame=0, end_frame=90)
    assert planner.checked_row(candidate, slot, 0, note, 1.5, [note], []) is None
    row = planner.checked_row(candidate, slot, 0, note, 1, [note], [])
    assert row['max_playback_speed'] == 1.25
    assert row['playback_speed_profile']['source_end'] == 3

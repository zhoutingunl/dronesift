import copy

import pytest

from scripts.plan_car_montage import check_plan, plan_windows, reviewed_row


def shot(start=0, *, video='DJI_20260928143858_0028_D.MP4', group='volcano',
         car=False, rotation=False, score=2, epoch=100):
    return dict(video=video, start=start, end=start + 2, source_start=start,
                source_end=start + 2, duration=2, capture_epoch=epoch + start + 1,
                stable_segment_start=0, stable_segment_end=100,
                scene_id=1, scene_group=group, move='push', rotation=rotation,
                reversed=False, score=score, quality={'ok': True},
                direction_error_deg=None, directional=True, vehicle_present=car, vehicle_moving=car,
                vehicle_screen_direction='right', vehicle_sequence='road',
                camera_view='rear', visual_review=dict(annotation_scope='window',
                approved_source_start=0, approved_source_end=100),
                motion_starts_immediately=True)


def slots(n):
    return [dict(start_frame=60 * i, end_frame=60 * (i + 1), duration=2,
                 slot=i, slot_start=2 * i, output_fps=30) for i in range(n)]


def plan(rows):
    return dict(timeline=rows, music_analysis={'total_frames': rows[-1]['end_frame']},
                track=None, route_mode='no_track', allow_reverse=False,
                max_consecutive_visual_group=2, source_gap_s=.4)


def test_review_must_cover_entire_window_and_negative_overlap_vetoes():
    row = shot(5)
    name = row['video']
    partial = dict(video=name, start=5, end=6, allowed=True, scene_group='road', vehicle_present=True,
                   vehicle_moving=True)
    assert reviewed_row(row, {'clips': [partial]}) is None
    full = dict(partial, end=8)
    assert reviewed_row(row, {'clips': [full]})['scene_group'] == 'road'
    denied = dict(video=name, start=6.5, end=7.5, allowed=False)
    assert reviewed_row(row, {'clips': [full, denied]}) is None


def test_no_track_row_keeps_camera_gps_without_route_claims():
    original = dict(shot(), route_s=100, route_lon=1, route_lat=2,
                    hike_bearing_deg=90, camera_route_s=100, camera_lon=112,
                    camera_lat=42, direction_error_deg=10)
    notes = {'clips': [dict(video=original['video'], start=0, end=3,
                           allowed=True, scene_group='crater', vehicle_present=False) ]}
    row = reviewed_row(original, notes)
    assert row['camera_lon'] == 112 and row['camera_lat'] == 42
    assert 'route_s' not in row and 'hike_bearing_deg' not in row
    assert row['direction_error_deg'] is None
    assert original['route_s'] == 100


def test_landscape_insert_does_not_reset_vehicle_chronology():
    first = shot(20, car=True, group='road_1')
    cutaway = shot(0, video='scenery.MP4', epoch=0, group='crater')
    earlier_car = shot(10, car=True, group='road_2', score=100)
    later_car = shot(24, car=True, group='road_2')
    timeline = plan_windows([[first], [cutaway], [earlier_car, later_car]], slots(3))
    assert timeline[-1]['start'] == 24
    assert check_plan(plan(timeline))['moving_vehicle_capture_order'] == 'forward'


def test_no_adjacent_rotation_and_no_repeated_source_window():
    first = shot(0, rotation=True)
    repeated = shot(0, group='other', score=200)
    rotation = shot(3, rotation=True, group='other', score=100)
    valid = shot(6, group='other')
    timeline = plan_windows([[first], [repeated, rotation, valid]], slots(2), max_rotation_fraction=.5)
    assert timeline[1]['start'] == 6
    assert check_plan(plan(timeline))['consecutive_rotations'] == 0


def test_same_view_car_direction_flip_needs_cutaway():
    first = shot(0, car=True)
    flip = dict(shot(3, car=True, group='other', score=100), vehicle_screen_direction='left')
    continuing = shot(6, car=True, group='other')
    timeline = plan_windows([[first], [flip, continuing]], slots(2))
    assert timeline[1]['start'] == 6


def test_required_landmark_and_dynamic_closing_survive_higher_score():
    road = shot(0, group='road', score=100)
    crater = shot(0, video='crater.MP4', group='crater', score=1)
    cattle = shot(0, video='cattle.MP4', group='cattle')
    timeline = plan_windows([[road, crater], [cattle]], slots(2),
                            required_scene_groups=['crater'], closing_scene_group='cattle')
    assert [r['scene_group'] for r in timeline] == ['crater', 'cattle']


def test_closing_source_window_is_reserved_before_its_slot():
    early_cattle = shot(0, video='cattle.MP4', group='cattle', score=100)
    road = shot(0, group='road')
    timeline = plan_windows([[early_cattle, road], [early_cattle]], slots(2),
                            closing_scene_group='cattle')
    assert timeline[0]['scene_group'] == 'road'


def test_parked_vehicle_view_does_not_rewind_driving_order():
    moving = shot(20, car=True, group='road_1')
    parked = dict(shot(0, car=True, group='parked', video='parked.MP4', epoch=0),
                  vehicle_moving=False, vehicle_screen_direction='stationary')
    next_moving = shot(24, car=True, group='road_2')
    timeline = plan_windows([[moving], [parked], [next_moving]], slots(3))
    assert check_plan(plan(timeline))['moving_vehicle_shots'] == 2


def test_validation_rejects_reverse_unreviewed_scope_and_music_shortfall():
    timeline = plan_windows([[shot()]], slots(1))
    valid = plan(timeline)
    for mutation in ('reverse', 'scope', 'music'):
        changed = copy.deepcopy(valid)
        if mutation == 'reverse': changed['timeline'][0]['reversed'] = True
        elif mutation == 'scope': changed['timeline'][0]['visual_review']['approved_source_end'] = 1
        else: changed['music_analysis']['total_frames'] = 61
        with pytest.raises(ValueError):
            check_plan(changed)

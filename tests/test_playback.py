"""Speed evidence must remain tied to actual source consumption and output FPS."""
import copy

import pytest

from dronesift.playback import validate_playback_speed


def check(row, start=2, end=5, output_fps=30):
    return validate_playback_speed(row, source_start=start, source_end=end,
                                   output_fps=output_fps, require_assessment=True)


def test_legacy_natural_playback_is_compatible_but_acceleration_needs_evidence():
    assert check({})['status'] == 'unassessed'
    with pytest.raises(ValueError, match='requires a playback_speed_profile'):
        check(dict(playback_speed=1.5))
    # A detached scalar ceiling cannot certify an otherwise unassessed edit.
    with pytest.raises(ValueError, match='requires a playback_speed_profile'):
        check(dict(playback_speed=1.5, max_playback_speed=2))


def test_ceiling_is_enforced_even_with_a_manually_changed_speed(playback_profile):
    row = dict(playback_speed=1.5, max_playback_speed=1.5,
               playback_speed_profile=playback_profile(2, 5, maximum=1.5))
    assert check(row)['status'] == 'assessed'
    row['playback_speed'] = 2
    with pytest.raises(ValueError, match='exceeds'):
        check(row)
    row['max_playback_speed'] = 4
    with pytest.raises(ValueError, match='exceeds the assessed maximum'):
        check(row)


def test_window_ceiling_cannot_be_reused_after_a_cut_or_fps_change(playback_profile):
    row = dict(playback_speed=1.5, playback_speed_profile=playback_profile(2, 5))
    for start, end in [(1.9, 5), (2, 5.1)]:
        with pytest.raises(ValueError, match='actual consumed source window'):
            check(row, start=start, end=end)
    with pytest.raises(ValueError, match='output_fps differs'):
        check(row, output_fps=24)
    # A parent's local trend/edge metrics do not certify a smaller crop.
    with pytest.raises(ValueError, match='exact consumed source window'):
        check(row, start=2.5, end=4.5)
    row['playback_speed'] = 1
    assert check(row, start=2.5, end=4.5)['status'] == 'catalog_advisory'


def test_unknown_and_failed_profiles_do_not_authorize_fast_playback(playback_profile):
    for maximum in [None, 0]:
        row = dict(playback_speed=1.5,
                   playback_speed_profile=playback_profile(2, 5, maximum=maximum))
        with pytest.raises(ValueError, match='unassessed|exceeds'):
            check(row)


def test_inconsistent_status_or_boolean_schema_does_not_certify_acceleration(playback_profile):
    row = dict(playback_speed=1.5, playback_speed_profile=playback_profile(2, 5))
    row['playback_speed_profile']['status'] = 'insufficient'
    with pytest.raises(ValueError, match='not an assessed'):
        check(row)
    row['playback_speed_profile']['status'] = 'assessed'
    row['playback_speed_profile']['schema_version'] = True
    with pytest.raises(ValueError, match='unsupported playback_speed_profile'):
        check(row)


@pytest.mark.parametrize('value', [True, float('nan'), float('inf'), -1, '2'])
def test_malformed_ceiling_is_not_treated_as_unlimited(playback_profile, value):
    row = dict(playback_speed=1.5, playback_speed_profile=playback_profile(2, 5))
    bad = copy.deepcopy(row)
    bad['playback_speed_profile']['max_playback_speed'] = value
    with pytest.raises(ValueError, match='invalid maximum'):
        check(bad)
    row['max_playback_speed'] = value
    with pytest.raises(ValueError, match='max_playback_speed'):
        check(row)

"""Cached catalog migration never recuts media or silently changes source scope."""
import copy
import csv
import json

import pytest

from dronesift import assess_speed, segment, viz


def cached_cuts(tmp_path, *, sprite=True):
    cuts = tmp_path / '湖畔_cuts_v6'
    cuts.mkdir()
    base, fps, count = 'flight', 30, 180
    original = tmp_path / '原始.MP4'
    original.write_bytes(b'original video must never be decoded')
    clip = cuts / 'flight_01.mp4'
    clip.write_bytes(b'exported media must remain byte-identical')
    edl = cuts / 'flight.edl'
    edl.write_bytes(b'original EDL')
    row = dict(start=1.0, end=5.0, duration=4, start_frame=30, end_frame=149,
               label='push_in', moves=['push_in'], base='push', speed_mps=3,
               yaw_rate_dps=0, custom_note='保留其它字段')
    document = dict(video=str(original), fps=fps, frames=count,
                    thresholds=dict(segment.DEFAULT_THR), segments=[row], selection={'unchanged': True})
    catalog = cuts / (base + '_segments.json')
    catalog.write_text(json.dumps(document, ensure_ascii=False), encoding='utf-8')
    exports = dict(video=str(original), mode='copy', total_duration=3.2,
                   clips=[dict(path=clip.name, requested_start=1, requested_end=5,
                               source_start=1.5, source_end=4.7, frames=96, duration=3.2, verified=True)])
    (cuts / (base + '_exports.json')).write_text(json.dumps(exports), encoding='utf-8')
    with (cuts / (base + '_telemetry.csv')).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['pts', 'vx', 'vy', 'vz', 'ac_yaw', 'g_yaw', 'g_pitch', 'g_roll',
                         'lat', 'lon', 'rel_alt', 'iso', 'exposure'])
        for i in range(count): writer.writerow([i/fps, 3, 0, 0, 0, 0, -20, 0, 43, 117, 20, 100, 1/60])
    with (cuts / (base + '_flow.csv')).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['frame', 'n_tracks', 'dx', 'dy', 'da', 'ds', 'inlier_ratio',
                         'residual', 'lap_var', 'mean_luma', 'mdx', 'mdy'])
        for i in range(count): writer.writerow([i, 50, 80/fps, 0, 0, 0, 1, .1, 100, 128, 80/fps, 0])
    if sprite:
        (cuts / (base + '_thumbs.jpg')).write_bytes(b'cached image; no thumbnail decoding allowed')
    return cuts, document, exports, [original, clip, edl]


def reject_subprocess(*args, **kwargs):
    pytest.fail('migration must not launch video decoding or encoding')


def test_migration_reuses_features_and_preserves_source_media_and_metadata(tmp_path, monkeypatch):
    cuts, before, exports_before, protected = cached_cuts(tmp_path)
    protected += list(cuts.glob('*.csv')) + list(cuts.glob('*_thumbs.jpg'))
    originals = {path: path.read_bytes() for path in protected}
    calls = []
    features = segment.features

    def measured(*args, **kwargs):
        calls.append(args[2])
        return features(*args, **kwargs)

    monkeypatch.setattr(segment, 'features', measured)
    monkeypatch.setattr(viz.subprocess, 'run', reject_subprocess)
    result = assess_speed.assess_directory(cuts, output_fps=60)
    assert calls == [30]  # The refreshed visualization reuses exactly this analysis.
    updated = json.loads((cuts / 'flight_segments.json').read_text(encoding='utf-8'))
    row = updated['segments'][0]
    assert row['playback_speed_profile']['source_start'] == 1
    assert row['playback_speed_profile']['source_end'] == 5
    assert row['playback_speed_profile']['output_fps'] == 60
    assert row['playback_speed_profile']['status'] == 'assessed'
    stripped = copy.deepcopy(updated)
    stripped['segments'][0].pop('playback_speed_profile')
    stripped['segments'][0].pop('max_playback_speed')
    assert stripped == before
    exports = json.loads((cuts / 'flight_exports.json').read_text(encoding='utf-8'))
    clip = exports['clips'][0]
    assert clip['playback_speed_profile'] == row['playback_speed_profile']
    assert clip['playback_speed_profile_scope']['source_start'] == 1
    assert clip['playback_speed_profile_scope']['exported_source_start'] == 1.5
    assert clip['playback_speed_profile_scope']['reassessed'] is False
    for key in ('playback_speed_profile', 'max_playback_speed', 'playback_speed_profile_scope'):
        clip.pop(key)
    assert exports == exports_before
    report = (cuts / 'flight_viz.html').read_text(encoding='utf-8')
    meta = json.JSONDecoder().raw_decode(report.split('const M = ', 1)[1])[0]
    assert meta['segs'][0]['playback_speed_profile'] == row['playback_speed_profile']
    assert meta['segs'][0]['speed_output_fps'] == 60
    assert 'speed-cap' in report
    assert result['visualization_refreshed'] and result['media_modified'] is False
    assert {path: path.read_bytes() for path in protected} == originals
    assert not list(cuts.glob('.assess-speed-*'))


def test_missing_sprite_skips_visualization_and_does_not_decode(tmp_path, monkeypatch):
    cuts, _, _, protected = cached_cuts(tmp_path, sprite=False)
    report = cuts / 'flight_viz.html'
    report.write_text('existing report stays unchanged', encoding='utf-8')
    protected.append(report)
    snapshots = {path: path.read_bytes() for path in protected}
    monkeypatch.setattr(viz.subprocess, 'run', reject_subprocess)
    result = assess_speed.assess_directory(cuts)
    assert not result['visualization_refreshed']
    assert {path: path.read_bytes() for path in protected} == snapshots


@pytest.mark.parametrize('failure', ['missing_csv', 'ambiguous_catalog', 'invalid_export', 'report_failure'])
def test_failed_preparation_never_replaces_metadata(tmp_path, monkeypatch, failure):
    cuts, document, _, _ = cached_cuts(tmp_path)
    if failure == 'missing_csv':
        (cuts / 'flight_flow.csv').unlink()
    elif failure == 'ambiguous_catalog':
        (cuts / 'other_segments.json').write_text(json.dumps(document), encoding='utf-8')
    elif failure == 'invalid_export':
        path = cuts / 'flight_exports.json'
        exports = json.loads(path.read_text(encoding='utf-8'))
        exports['clips'][0]['source_end'] = 5.5
        path.write_text(json.dumps(exports), encoding='utf-8')
    else:
        monkeypatch.setattr(viz, 'build', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('report failed')))
    originals = {path: path.read_bytes() for path in cuts.glob('*.json')}
    with pytest.raises((ValueError, RuntimeError)):
        assess_speed.assess_directory(cuts)
    assert {path: path.read_bytes() for path in cuts.glob('*.json')} == originals
    assert not list(cuts.glob('.assess-speed-*'))


def test_cli_reports_missing_directory_and_rejects_invalid_output_fps(tmp_path, capsys):
    assert assess_speed.main([str(tmp_path / 'missing')]) == 1
    assert 'cuts directory does not exist' in capsys.readouterr().err
    with pytest.raises(SystemExit):
        assess_speed.main([str(tmp_path), '--output-fps', 'nan'])

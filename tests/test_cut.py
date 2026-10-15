"""Timecodes, encoders, and lossless cut boundary safety."""
import json
import shutil
import subprocess

import pytest
from dronesift import cut, flow
from dronesift.segment import hms

def test_tc():
    assert cut.tc(3 * 3600 + 2 * 60 + 3.25, 60) == '03:02:03:15'
    assert cut.tc(0, 59.94) == '00:00:00:00'

def test_hms():
    assert hms(75.5) == '01:15.500'

def test_encoder_args_macos(monkeypatch):
    monkeypatch.setattr(cut.platform, 'system', lambda: 'Darwin')
    assert 'hevc_videotoolbox' in cut.encoder_args('encode', '80M')
    assert 'prores_videotoolbox' in cut.encoder_args('prores', '80M')

def test_encoder_args_linux_fallback(monkeypatch):
    monkeypatch.setattr(cut.platform, 'system', lambda: 'Linux')
    args = cut.encoder_args('encode', '80M')
    assert 'libx265' in args and 'p010le' in args and 'bt709' in args
    args = cut.encoder_args('prores', '80M')
    assert 'prores_ks' in args and 'yuv422p10le' in args

def test_hwaccel_default_by_platform(monkeypatch):
    monkeypatch.setattr(flow.platform, 'system', lambda: 'Darwin')
    assert flow.hwaccel_default() == 'videotoolbox'
    monkeypatch.setattr(flow.platform, 'system', lambda: 'Linux')
    assert flow.hwaccel_default() is None

def _packet(pts, key=False, duration=0.1):
    return dict(pts=pts, dts=pts, duration=duration, keyframe=key)

def test_copy_plan_requires_an_in_range_keyframe():
    packets = [_packet(0, True), _packet(0.1), _packet(0.2)]
    assert cut.copy_plan(packets, 0.05, 0.3) == []

def test_copy_plan_keeps_video_endpoints_and_complete_frames():
    packets = [_packet(0, True), _packet(0.1), _packet(0.2)]
    assert cut.copy_plan(packets, 0, 0.3) == packets
    assert cut.copy_plan(packets, 0, 0.25) == packets[:2]

def test_copy_plan_stops_before_a_reordered_future_reference():
    packets = [_packet(0, True), _packet(0.3), _packet(0.1), _packet(0.2)]
    assert cut.copy_plan(packets, 0, 0.3) == packets[:1]

@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='requires ffmpeg/ffprobe')
def test_copy_export_reordered_source_stays_inside_requested_range(tmp_path):
    source = tmp_path / 'reordered.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    'testsrc2=size=96x64:rate=30', '-t', '2', '-c:v', 'libx264',
                    '-g', '15', '-bf', '2', '-pix_fmt', 'yuv420p', str(source)], check=True)
    packets = cut.video_packets(str(source), 30)
    assert any(abs(p['pts'] - p['dts']) > 0.001 for p in packets)
    outdir = tmp_path / 'cuts'
    profile = dict(schema_version=1, method='motion_speed_v1', source_start=.53,
                   source_end=1.8, output_fps=30, max_playback_speed=1.5,
                   assessments=[dict(speed=1.5, passed=True)], limiting_reasons=['motion_limit'])
    segs = [dict(start=0.53, end=1.8, label='pan', playback_speed_profile=profile,
                 max_playback_speed=1.5)]
    paths = cut.cut(str(source), segs, str(outdir), fps=30)
    manifest = json.loads((outdir / 'reordered_exports.json').read_text())
    clip = manifest['clips'][0]
    assert paths == [str(outdir / clip['path'])]
    assert clip['verified']
    assert clip['source_start'] >= 0.53
    assert clip['source_end'] <= 1.8 + 1e-6
    assert clip['frames'] == len(cut.video_packets(paths[0], 30))
    assert manifest['total_duration'] == clip['duration']
    assert clip['playback_speed_profile'] == profile
    assert clip['max_playback_speed'] == 1.5
    assert clip['playback_speed_profile_scope'] == dict(
        basis='complete_catalog_segment', source_start=.53, source_end=1.8,
        exported_source_start=clip['source_start'], exported_source_end=clip['source_end'],
        reassessed=False, subset_policy='inherits_catalog_advisory_ceiling_without_reassessment',
        exact_window_assessment_required_for_playback=True,
        catalog_profile_available=True)
    # Packet payloads establish that the seek/count exported the intended
    # source frames, even when several GOPs share the same timestamp pattern.
    def hashes(path):
        result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                 '-show_packets', '-show_data_hash', 'sha256',
                                 '-show_entries', 'packet=data_hash', '-of', 'json', str(path)],
                                capture_output=True, text=True, check=True)
        return [p['data_hash'] for p in json.loads(result.stdout)['packets']]
    first = next(i for i, p in enumerate(packets) if p['pts'] == clip['source_start'])
    assert hashes(paths[0]) == hashes(source)[first:first + clip['frames']]
    result = subprocess.run(['ffmpeg', '-v', 'error', '-i', paths[0], '-f', 'null', '-'],
                            capture_output=True, text=True, check=True)
    assert not result.stderr

def test_copy_export_skips_instead_of_using_an_unsafe_start(tmp_path, monkeypatch):
    monkeypatch.setattr(cut, 'video_packets', lambda *_: [_packet(0, True), _packet(0.1)])
    paths = cut.cut('no_keyframe.mp4', [dict(start=0.05, end=0.2, label='pan')], str(tmp_path), fps=10)
    assert paths == []
    manifest = json.loads((tmp_path / 'no_keyframe_exports.json').read_text())
    assert manifest['total_duration'] == 0
    assert manifest['skipped'][0]['requested_start'] == 0.05

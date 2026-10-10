"""End-to-end smoke run: synthetic MP4 + cached telemetry/flow CSVs through the full CLI.

Skipped automatically when ffmpeg/ffprobe are not installed; needs no real DJI footage
(the telemetry and optical-flow CSV caches are written by hand, so no djmd track is required).
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.skipif(
    shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None, reason='ffmpeg/ffprobe not installed')

def test_full_pipeline_synthetic_video(tmp_path):
    video = tmp_path / 'IN.MP4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=duration=6:size=640x360:rate=30',
                    '-pix_fmt', 'yuv420p', str(video)], check=True)
    out = tmp_path / 'out'; out.mkdir()
    fps, n = 30, 180
    # steady 3 m/s forward push; telemetry pts offset by 2 frames to exercise the alignment path
    with open(out / 'IN_telemetry.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['pts', 'frame', 'ts_us', 'iso', 'exposure', 'ev', 'vx', 'vy', 'vz', 'ac_pitch', 'ac_roll', 'ac_yaw',
                    'lat', 'lon', 'abs_alt', 'rel_alt', 'g_pitch', 'g_roll', 'g_yaw', 'qw', 'qx', 'qy', 'qz'])
        for i in range(n):
            w.writerow([f'{(i + 2) / fps:.6f}', i, i * 33333, 100, 1 / 60, 0, 3.0, 0.0, 0.0, 0, 0, 0,
                        37.0, -122.0, 50, 20, -10, 0, 0, 1, 0, 0, 0])
    with open(out / 'IN_flow.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['frame', 'n_tracks', 'dx', 'dy', 'da', 'ds', 'inlier_ratio', 'residual', 'lap_var', 'mean_luma', 'mdx', 'mdy'])
        for i in range(n):
            w.writerow([i, 50, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1, 100.0, 128.0, 0.0, 0.0])
    r = subprocess.run([sys.executable, '-m', 'dronesift', str(video), '--mode', 'none', '--out', str(out)],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=ROOT), cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    d = json.load(open(out / 'IN_segments.json'))
    assert d['fps'] == 30.0
    assert len(d['segments']) == 1
    s = d['segments'][0]
    assert s['label'] == 'push_in' and s['duration'] > 4.0
    assert (out / 'IN.edl').exists() and 'FROM CLIP NAME: IN.MP4' in (out / 'IN.edl').read_text()
    assert (out / 'IN_timeline.png').exists() and (out / 'IN_viz.html').exists()

"""SRT fallback parsing and GPS-derived velocities."""
import numpy as np
from dronesift import srt

SRT = """1
00:00:00,000 --> 00:00:00,033
FrameCnt: 1
[latitude: 37.441900] [longitude: -122.143000] [rel_alt: 10.500 abs_alt: 32.100] [iso: 100]
[shutter: 1/120] [ev: 0.3] [ct: 5500]

2
00:00:00,100 --> 00:00:00,133
FrameCnt: 2
[latitude: 37.441910] [longitude: -122.143000] [rel_alt: 10.600 abs_alt: 32.200] [iso: 100]
[shutter: 1/120] [ev: 0.3] [ct: 5500]
"""

def test_parse_blocks(tmp_path):
    p = tmp_path / 'x.SRT'; p.write_text(SRT)
    rows = srt.parse(str(p))
    assert len(rows) == 2
    assert rows[0]['t'] == 0.0 and rows[1]['t'] == 0.1
    assert rows[0]['frame'] == 0 and rows[1]['frame'] == 1          # FrameCnt is 1-based
    assert rows[0]['latitude'] == 37.4419 and rows[0]['longitude'] == -122.143
    assert rows[0]['rel_alt'] == 10.5 and rows[0]['abs_alt'] == 32.1
    assert abs(rows[0]['shutter'] - 1 / 120) < 1e-9
    assert rows[0]['ev'] == 0.3

def test_to_telemetry_velocity_from_gps():
    rows = [dict(t=i * 0.1, frame=i, latitude=37.0 + i * 1e-5, longitude=-122.0,
                 rel_alt=10.0, abs_alt=50.0, iso=100, shutter=0.008, ev=0.0) for i in range(40)]
    tel = srt.to_telemetry(rows, fps=10)
    assert tel[0]['pts'] == 0.0 and len(tel) == 40
    vx = np.array([r['vx'] for r in tel]); vy = np.array([r['vy'] for r in tel])
    assert 10 < vx[5:-5].max() < 12.5        # 1e-5 deg north / 0.1 s ≈ 11.1 m/s
    assert np.abs(vy[5:-5]).max() < 0.5

"""Unicode HTML export under the legacy Windows default text encoding."""
import builtins
import json

import numpy as np

from dronesift import segment, viz


def test_visualization_preserves_utf8_under_cp1252_default(tmp_path, monkeypatch):
    count = 4
    feature_keys = ('speed_h', 'rel_alt', 'vz', 'yaw_rate', 'pitch_rate',
                    'jitter_rms', 'motion_px', 'motion_acc', 'consist')
    features = {key: np.zeros(count) for key in feature_keys}
    telemetry = {key: np.zeros(count) for key in ('ac_yaw', 'g_pitch')}
    telemetry.update(lat=np.full(count, 43), lon=np.full(count, 117))
    monkeypatch.setattr(segment, 'features', lambda *a, **k: (features, count))
    monkeypatch.setattr(segment, 'classify_frames',
                        lambda *a, **k: (np.ones(count, dtype=bool), np.full(count, 'push')))
    monkeypatch.setattr(segment, 'aligned_tel', lambda *a, **k: (telemetry, np.ones(count, dtype=bool)))
    monkeypatch.setattr(viz, 'thumb_sprite', lambda *a, **k: (1, 1, 1))
    source = tmp_path / '山谷 #100%.MP4'
    (tmp_path / (source.stem + '_thumbs.jpg')).write_bytes(b'test thumbnail')
    tel, flow = tmp_path / 'telemetry.csv', tmp_path / 'flow.csv'
    tel.write_text('frame\n0\n', encoding='utf-8')
    flow.write_text('frame\n0\n', encoding='utf-8')
    segments = tmp_path / 'segments.json'
    document = dict(fps=30, thresholds={}, segments=[dict(start=0, end=count / 30,
                    duration=count / 30, label='推进', base='push', speed_mps=3,
                    yaw_rate_dps=0)])
    segments.write_text(json.dumps(document, ensure_ascii=False), encoding='utf-8')

    def cp1252_default_open(path, mode='r', *args, **kwargs):
        if 'b' not in mode and 'encoding' not in kwargs:
            kwargs['encoding'] = 'cp1252'
        return builtins.open(path, mode, *args, **kwargs)

    # Only change this module's default, preserving pytest's own file handling.
    monkeypatch.setattr(viz, 'open', cp1252_default_open, raising=False)
    output = tmp_path / '轨迹.html'
    viz.build(str(source), str(tel), str(flow), str(segments), str(output))
    text = output.read_bytes().decode('utf-8')
    assert '<meta charset="utf-8">' in text
    assert source.stem + ' 轨迹' in text
    assert '推进' in text
    assert '3D 图：拖动旋转' in text

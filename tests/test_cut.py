"""Timecode math and platform encoder selection."""
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

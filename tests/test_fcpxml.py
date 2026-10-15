"""FCPXML export structure."""
import builtins
from pathlib import PureWindowsPath
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from dronesift import fcpxml

SEGS = [dict(start=7.1, end=15.1, duration=8.0, label='push_in'),
        dict(start=20.0, end=26.0, duration=6.0, label='orbit_CW')]

def test_fcpxml_structure(tmp_path):
    p = tmp_path / 'x.fcpxml'
    video = tmp_path / 'DJI_0001.D.MP4'
    fcpxml.write(SEGS, str(video), str(p), 59.94005994)
    root = ET.parse(str(p)).getroot()
    assert root.tag == 'fcpxml' and root.attrib['version'] == '1.9'
    fmt = root.find('./resources/format')
    assert fmt.attrib['frameDuration'] == '1001/60000s'        # NTSC-ish rate gets the exact rational
    asset = root.find('./resources/asset')
    assert asset.attrib['src'] == video.resolve().as_uri()
    clips = root.findall('./library/event/project/sequence/spine/asset-clip')
    assert [c.attrib['name'] for c in clips] == ['push_in', 'orbit_CW']
    assert clips[0].attrib['offset'] == '0s'
    assert clips[0].attrib['start'] == _approx(7.1)
    assert clips[1].attrib['offset'] == clips[0].attrib['duration']   # spine is contiguous

def _approx(sec, fps=59.94005994):
    from dronesift.fcpxml import frame_rate
    fd, _ = frame_rate(fps)
    return fcpxml._t(round(sec * fps), fd)

def test_frame_rate_matching():
    from fractions import Fraction
    assert fcpxml.frame_rate(59.94005994)[0] == Fraction(1001, 60000)
    assert fcpxml.frame_rate(30.0)[0] == Fraction(1, 30)
    assert fcpxml.frame_rate(25.0)[0] == Fraction(1, 25)
    assert fcpxml.frame_rate(59.94)[1] == 'FFVideoFormat1080p5994'


def test_file_uri_escapes_unicode_and_reserved_characters(tmp_path, monkeypatch):
    def cp1252_default_open(path, mode='r', *args, **kwargs):
        if 'b' not in mode and 'encoding' not in kwargs:
            kwargs['encoding'] = 'cp1252'
        return builtins.open(path, mode, *args, **kwargs)

    monkeypatch.setattr(fcpxml, 'open', cp1252_default_open, raising=False)
    video = tmp_path / 'DJI 山谷 #100%.MP4'
    output = tmp_path / 'edit.fcpxml'
    fcpxml.write(SEGS, str(video), str(output), 30)
    asset = ET.parse(output).find('./resources/asset')
    assert asset.attrib['src'] == video.resolve().as_uri()
    assert '%20' in asset.attrib['src']
    assert '%E5%B1%B1%E8%B0%B7' in asset.attrib['src']
    assert '%23' in asset.attrib['src'] and '%25' in asset.attrib['src']
    assert asset.attrib['name'] == video.stem


@pytest.mark.parametrize(('video', 'expected'), [
    (r'C:\DJI footage\山谷 #100%.MP4',
     'file:///C:/DJI%20footage/%E5%B1%B1%E8%B0%B7%20%23100%25.MP4'),
    (r'\\drone-server\shots\山谷 #100%.MP4',
     'file://drone-server/shots/%E5%B1%B1%E8%B0%B7%20%23100%25.MP4'),
])
def test_windows_drive_and_unc_file_uri(tmp_path, monkeypatch, video, expected):
    # Resolve to a Windows path without needing a real Windows drive/network
    # share on this host. PureWindowsPath supplies the actual Windows URI rules.
    monkeypatch.setattr(fcpxml, 'Path',
                        lambda value: SimpleNamespace(resolve=lambda: PureWindowsPath(value)))
    output = tmp_path / 'windows.fcpxml'
    fcpxml.write(SEGS, video, str(output), 30)
    assert ET.parse(output).find('./resources/asset').attrib['src'] == expected

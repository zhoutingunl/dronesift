"""FCPXML export structure."""
import os
import xml.etree.ElementTree as ET

from dronesift import fcpxml

SEGS = [dict(start=7.1, end=15.1, duration=8.0, label='push_in'),
        dict(start=20.0, end=26.0, duration=6.0, label='orbit_CW')]

def test_fcpxml_structure(tmp_path):
    p = tmp_path / 'x.fcpxml'
    fcpxml.write(SEGS, '/movies/DJI_0001.D.MP4', str(p), 59.94005994)
    root = ET.parse(str(p)).getroot()
    assert root.tag == 'fcpxml' and root.attrib['version'] == '1.9'
    fmt = root.find('./resources/format')
    assert fmt.attrib['frameDuration'] == '1001/60000s'        # NTSC-ish rate gets the exact rational
    asset = root.find('./resources/asset')
    assert asset.attrib['src'] == 'file:///movies/DJI_0001.D.MP4'
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

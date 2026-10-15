"""Minimal FCPXML 1.9 spine export — imports into Final Cut Pro and DaVinci Resolve.

Asset clips reference the ORIGINAL video file with source in/out, so nothing is re-encoded
and the NLE treats dronesift output like a normal edit.
"""
import os
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.dom import minidom

# exact rational frame durations for common camera rates (tolerance-matched)
_RATES = ((59.94, 1001, 60000, 'FFVideoFormat1080p5994'), (29.97, 1001, 30000, 'FFVideoFormat1080p2997'),
          (23.976, 1001, 24000, 'FFVideoFormat1080p2398'), (119.88, 1001, 120000, None),
          (24.0, 1, 24, 'FFVideoFormat1080p24'), (25.0, 1, 25, 'FFVideoFormat1080p25'),
          (30.0, 1, 30, 'FFVideoFormat1080p30'), (50.0, 1, 50, 'FFVideoFormat1080p50'),
          (60.0, 1, 60, 'FFVideoFormat1080p60'))

def frame_rate(fps):
    """(frameDuration fraction, format name) for a float fps."""
    for target, num, den, name in _RATES:
        if abs(fps - target) < 0.02:
            return Fraction(num, den), name
    n = round(fps)
    return Fraction(1, n), None

def _t(frames, fd):
    v = Fraction(frames) * fd
    return f"{v.numerator}/{v.denominator}s" if v.denominator != 1 else f"{v.numerator}s"

def write(items, video, path, fps):
    """items: segments or shots dicts (start/end in seconds). Writes one spine in list order."""
    fd, fmt_name = frame_rate(fps)
    root = ET.Element('fcpxml', version='1.9')
    res = ET.SubElement(root, 'resources')
    fmt_attrs = dict(id='r1', frameDuration=_t(1, fd))
    if fmt_name: fmt_attrs['name'] = fmt_name
    ET.SubElement(res, 'format', fmt_attrs)
    name = os.path.splitext(os.path.basename(video))[0]
    src = Path(video).resolve().as_uri()
    ET.SubElement(res, 'asset', dict(id='a1', name=name, src=src, format='r1', hasVideo='1',
                                     audioRate='48000', audioChannels='2'))
    lib = ET.SubElement(root, 'library')
    event = ET.SubElement(lib, 'event', name='dronesift')
    proj = ET.SubElement(event, 'project', name=name)
    total = sum(Fraction(round(i['duration'] * fps)) * fd for i in items)
    seq = ET.SubElement(proj, 'sequence', format='r1', duration=_t(0, fd) if not items else _t(round(total / fd), fd))
    spine = ET.SubElement(seq, 'spine')
    off = 0
    for i in items:
        nfr = round(i['duration'] * fps)
        clip = ET.SubElement(spine, 'asset-clip', ref='a1', name=f"{i.get('label', 'shot')}",
                             offset=_t(off, fd), start=_t(round(i['start'] * fps), fd), duration=_t(nfr, fd),
                             note=f"{i.get('label', '')} {i['duration']:.1f}s")
        off += nfr
    xml = minidom.parseString(ET.tostring(root, encoding='unicode')).toprettyxml(indent='  ')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(xml)
    return path

if __name__ == '__main__':
    import json, sys
    d = json.load(open(sys.argv[1]))
    print(write(d.get('shots') or d['segments'], d['video'], sys.argv[2], d['fps']))

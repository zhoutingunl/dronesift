"""Parse DJI djmd metadata track (Air 3S / dvtm_Air3s.proto) into per-frame records.

Field numbers were inferred by schema-less decoding plus dvtm_library.proto:
  3 frame_meta
    1 header: 1 frame_index, 2 timestamp_us
    2 camera_frame_meta: 9 iso, 10 exposure_time(packed [num,den]), 31 ev, 32 ct
    3 fc_frame_meta: 2 Velocity(x,y,z m/s), 3 EulerAngle(pitch,roll,yaw deci-deg),
                     4 GpsBasic(1 coords{2 lat_rad,3 lon_rad}, 2 alt_mm), 5 RelativeAltitude(1 mm)
    4 gimbal_frame_meta: 3 EulerAngle(pitch,roll,yaw deci-deg), 4 Quaternion(w,x,y,z)
"""
import csv, math, struct, subprocess, json
from .pbraw import read_varint

def _fields(b):
    """Yield (field, wiretype, value) for one message."""
    i, n = 0, len(b)
    while i < n:
        key, i = read_varint(b, i); f, wt = key >> 3, key & 7
        if wt == 0: v, i = read_varint(b, i)
        elif wt == 1: v = b[i:i+8]; i += 8
        elif wt == 5: v = b[i:i+4]; i += 4
        elif wt == 2: ln, i = read_varint(b, i); v = b[i:i+ln]; i += ln
        else: raise ValueError('bad wiretype')
        yield f, wt, v

def _msg(b):
    d = {}
    for f, wt, v in _fields(b): d.setdefault(f, []).append((wt, v))
    return d

def _f32(d, f, default=0.0):
    if f not in d: return default
    return struct.unpack('<f', d[f][0][1])[0]
def _f64(d, f, default=0.0):
    if f not in d: return default
    return struct.unpack('<d', d[f][0][1])[0]
def _sint(d, f, default=0):
    if f not in d: return default
    v = d[f][0][1]
    return v - (1 << 64) if v >= (1 << 63) else v
def _sub(d, f):
    return _msg(d[f][0][1]) if f in d else {}

def packets(mp4, stream_index=1):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', str(stream_index),
                          '-show_entries', 'packet=pts_time,size,pos', '-of', 'json', mp4],
                         capture_output=True, text=True, check=True).stdout
    with open(mp4, 'rb') as fh:
        for p in json.loads(out)['packets']:
            fh.seek(int(p['pos'])); yield float(p['pts_time']), fh.read(int(p['size']))

def parse(mp4):
    rows = []
    for pts, b in packets(mp4):
        row = _decode(b, pts)
        if row is not None: rows.append(row)
    return rows

def _decode(b, pts):
    """One djmd packet -> telemetry row dict, or None if the packet is not a frame_meta."""
    top = _msg(b)
    if 3 not in top: return None
    fm = _sub(top, 3)
    hdr = _sub(fm, 1); cam = _sub(fm, 2); fc = _sub(fm, 3); gb = _sub(fm, 4)
    vel = _sub(fc, 2); fce = _sub(fc, 3); gps = _sub(fc, 4); coords = _sub(gps, 1); ralt = _sub(fc, 5)
    ge = _sub(gb, 3); gq = _sub(gb, 4)
    exp = None
    if 10 in cam:
        e = _sub(cam, 10)
        if 1 in e:
            raw = e[1][0][1]; vals = []; i = 0
            while i < len(raw): v, i = read_varint(raw, i); vals.append(v)
            if len(vals) == 2 and vals[1]: exp = vals[0] / vals[1]
    return dict(
        pts=pts, frame=_sint(hdr, 1), ts_us=_sint(hdr, 2),
        iso=_f32(_sub(cam, 9), 1), exposure=exp, ev=_f32(_sub(cam, 31), 1),
        vx=_f32(vel, 1), vy=_f32(vel, 2), vz=_f32(vel, 3),
        ac_pitch=_sint(fce, 1) / 10, ac_roll=_sint(fce, 2) / 10, ac_yaw=_sint(fce, 3) / 10,
        lat=math.degrees(_f64(coords, 2)), lon=math.degrees(_f64(coords, 3)),
        abs_alt=_sint(gps, 2) / 1000, rel_alt=_f32(ralt, 1) / 1000,
        g_pitch=_sint(ge, 1) / 10, g_roll=_sint(ge, 2) / 10, g_yaw=_sint(ge, 3) / 10,
        qw=_f32(gq, 1), qx=_f32(gq, 2), qy=_f32(gq, 3), qz=_f32(gq, 4),
    )

def write_csv(rows, path):
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

if __name__ == '__main__':
    import sys
    rows = parse(sys.argv[1]); write_csv(rows, sys.argv[2]); print(len(rows), 'frames')

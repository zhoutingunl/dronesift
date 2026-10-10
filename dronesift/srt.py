"""Parse DJI SRT telemetry (per-frame GPS/altitude/exposure) into records."""
import re, csv
_TS = re.compile(r'(\d+):(\d+):(\d+),(\d+) --> ')
_KV = re.compile(r'\[([a-z_]+)\s*:\s*([^\]]+?)\s*\]')

def parse(path):
    rows = []
    with open(path, encoding='utf-8', errors='replace') as fh:
        block = []
        for line in fh:
            line = line.strip()
            if line: block.append(line); continue
            if block: rows.append(_block(block)); block = []
        if block: rows.append(_block(block))
    return [r for r in rows if r]

def _block(lines):
    m = _TS.search(' '.join(lines))
    if not m: return None
    t = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1000
    text = ' '.join(lines)
    d = {'t': t}
    fc = re.search(r'FrameCnt:\s*(\d+)', text)
    if fc: d['frame'] = int(fc[1]) - 1
    for k, v in _KV.findall(text):
        if k == 'rel_alt':
            m2 = re.match(r'([-\d.]+)\s+abs_alt:\s*([-\d.]+)', v)
            if m2: d['rel_alt'] = float(m2[1]); d['abs_alt'] = float(m2[2]); continue
        if k == 'shutter':
            v = v.replace('1/', ''); 
            try: d['shutter'] = 1 / float(v)
            except ValueError: pass
            continue
        try: d[k] = float(v)
        except ValueError: d[k] = v
    return d

if __name__ == '__main__':
    import sys
    rows = parse(sys.argv[1])
    with open(sys.argv[2], 'w', newline='') as fh:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != 't', k != 'frame', k))
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(len(rows), 'records; keys:', keys)

def to_telemetry(rows, fps=59.94):
    """Build djmd-like telemetry rows from SRT GPS (velocity by differentiating position; no gimbal data)."""
    import math
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    lat = np.array([r.get('latitude', 0.0) for r in rows]); lon = np.array([r.get('longitude', 0.0) for r in rows])
    alt = np.array([r.get('rel_alt', 0.0) for r in rows]); t = np.array([r['t'] for r in rows])
    lat0 = np.radians(lat.mean()); R = 6371000.0
    x = np.radians(lat - lat.mean()) * R; y = np.radians(lon - lon.mean()) * R * math.cos(lat0)   # north, east (m)
    sig = 0.5 * fps
    x, y, alt_s = (gaussian_filter1d(v, sig) for v in (x, y, alt))
    vx, vy = np.gradient(x) * fps, np.gradient(y) * fps; vz = -np.gradient(alt_s) * fps
    track = np.degrees(np.arctan2(vy, vx))
    out = []
    for i, r in enumerate(rows):
        out.append(dict(pts=t[i], frame=i, ts_us=int(t[i] * 1e6), iso=r.get('iso', 0), exposure=r.get('shutter', 0), ev=r.get('ev', 0),
                        vx=vx[i], vy=vy[i], vz=vz[i], ac_pitch=0, ac_roll=0, ac_yaw=track[i],
                        lat=lat[i], lon=lon[i], abs_alt=r.get('abs_alt', 0), rel_alt=alt[i],
                        g_pitch=0, g_roll=0, g_yaw=0, qw=0, qx=0, qy=0, qz=0))
    return out

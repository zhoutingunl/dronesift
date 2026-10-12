"""Music-aware draft assembly: beat-quantised, richness-weighted sequence of shots.

Pipeline:  shots -> richness -> desired duration -> quantised to 1/2/4-bar slots on the
song's beat grid -> greedy slot filling -> draft.mp4 (uniform 1080p30, hard cuts on bar
lines, music mixed in) + edit.json (the timeline) + draft EDL/FCPXML.

usage: python -m dronesift.edit MUSIC.mp3 DIR... [-o draft.mp4]
       DIRs hold <video>_cuts_v3/ results; any *_cuts* dir with a _segments.json works.
"""
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

FPS = 30.0
H, W = 1080, 1920
BAR_UNITS = (1, 2, 4)          # candidate slot lengths in bars
DUR_MIN, DUR_MAX = 2.5, 10.0   # hard clamps on shot length, s
DUR_LOW, DUR_HIGH = 3.0, 9.0   # richness maps into this range

# ---------------------------------------------------------------- music / beats

def _decode_mono(path, sr=22050):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tf:
        wav = tf.name
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', path, '-ac', '1', '-ar', str(sr), wav], check=True)
    import wave
    with wave.open(wav) as w:
        n = w.getnframes()
        y = np.frombuffer(w.readframes(n), np.int16).astype(np.float32) / 32768.0
    os.unlink(wav)
    return y, sr

def beat_grid(music_path, bpm_range=(60, 150)):
    """(bars, bpm, duration): bar lines from spectral-flux onset envelope; tempo by autocorrelation
    with octave correction (prefer the slower doubling while it stays strong), downbeat phase by
    onset energy mod 4. Good enough for bar-quantised cutting (~±50 ms)."""
    y, sr = _decode_mono(music_path)
    hop = 512
    S = _stft_mag(y, hop)                            # magnitude spectrogram via FFT frames
    flux = np.maximum(0.0, np.diff(S, axis=1)).sum(axis=0)
    flux = gaussian_filter1d(flux, 2)
    env = flux / (flux.max() + 1e-9)
    dt = hop / sr
    # tempo: strongest autocorrelation lag in the BPM range
    ac = np.correlate(env, env, 'full')[len(env):]
    lo, hi = int(60 / bpm_range[1] / dt), int(60 / bpm_range[0] / dt)
    lag = lo + int(np.argmax(ac[lo:hi]))
    while 2 * lag <= hi and ac[2 * lag] >= 0.8 * ac[lag]:    # octave fix: 184 detected -> real 92
        lag *= 2
    bpm = 60 / (lag * dt)
    period = lag * dt
    # beat phase: align beat grid to onset envelope by energy at grid points (try 16 sub-phases)
    best = (-1, 0.0)
    for ph in np.linspace(0, period, 16, endpoint=False):
        idx = (np.arange(ph, env.shape[0], period) / dt).astype(int)
        idx = idx[idx < env.shape[0]]
        e = env[idx].mean()
        if e > best[0]: best = (e, ph)
    dur = len(y) / sr
    beats = np.arange(best[1], dur, period)
    # bar lines: 4/4; pick downbeat phase among the 4 beats by onset energy
    db_energy = []
    for k in range(4):
        idx = (beats[k::4] / dt).astype(int)
        idx = idx[idx < env.shape[0]]
        db_energy.append(env[idx].mean() if len(idx) else 0)
    off = int(np.argmax(db_energy))
    bars = list(beats[off::4])
    return bars, bpm, dur

def _stft_mag(y, hop, n_fft=2048):
    win = np.hanning(n_fft)
    nf = 1 + (len(y) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(nf)[:, None]
    frames = y[idx] * win
    return np.abs(np.fft.rfft(frames, axis=1)).T

# ---------------------------------------------------------------- richness & windows

LABEL_PRIOR = dict(orbit=1.0, spiral=0.95, truck=0.8, diagonal=0.8, push=0.7, pull=0.7,
                   vertical=0.5, pan=0.4, tilt=0.4, static=0.35)

def richness(f, sl, base):
    """0..1 content-richness of a shot window: featureful image + interesting move + motion variety."""
    prior = LABEL_PRIOR.get(str(base).partition('+')[0], 0.6)
    tracks = float(np.nanmean(f['n_tracks'][sl]))
    sharp = float(np.nanmean(f['residual'][sl]))
    mvar = float(np.nanstd(f['motion_px'][sl]) / (np.nanmean(f['motion_px'][sl]) + 1e-6))
    return float(np.clip(0.4 * prior + 0.3 * min(tracks / 400, 1) + 0.15 * min(mvar, 0.6) / 0.6
                         + 0.15 * min(sharp / 0.6, 1), 0, 1))

def load_track(path):
    """Hiking track from a KML as (pts Nx2 lon/lat, t seconds from start, s metres along track).
    t/s are None when the KML carries no <when> timestamps. On out-and-back trails s(t) rises to
    the turnaround then falls; pass 0 = before the turnaround (outbound), pass 1 = after."""
    import re
    from datetime import datetime
    text = open(path, encoding='utf-8', errors='replace').read()
    pts, whens = [], []
    for m in re.finditer(r'<gx:coord>([^<]+)</gx:coord>', text):
        v = m.group(1).split()
        if len(v) >= 2:
            pts.append((float(v[0]), float(v[1])))
    if not pts:
        for m in re.finditer(r'<coordinates>([^<]+)</coordinates>', text):
            for tup in m.group(1).split():
                v = tup.split(',')
                if len(v) >= 2:
                    pts.append((float(v[0]), float(v[1])))
    for m in re.finditer(r'<when>([^<]+)</when>', text):
        try:
            whens.append(datetime.fromisoformat(m.group(1).strip().replace('Z', '+00:00')))
        except ValueError:
            pass
    if not pts:
        return None
    pts = np.array(pts)
    if len(whens) != len(pts):
        return pts, None, None
    # axis = seconds from the first track point; both this and 'video filename time minus hike
    # start' share the same clock origin (the KML <name> is the local rendering of whens[0]),
    # so no timezone arithmetic is needed anywhere
    t = np.array([(w - whens[0]).total_seconds() for w in whens])
    import math
    scale = np.array([111320.0 * math.cos(math.radians(pts[:, 1].mean())), 111320.0])
    seg = np.diff(pts, axis=0) * scale                  # degrees -> metres
    s = np.concatenate(([0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))))
    return pts, t, s

def capture_time_s(video, t0_local):
    """Seconds from the hike start for a DJI filename (DJI_YYYYMMDDHHMMSS...), or None."""
    import re
    from datetime import datetime
    m = re.search(r'DJI_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})', os.path.basename(video))
    if not m:
        return None
    y, mo, d, h, mi, s = map(int, m.groups())
    try:
        dt = datetime(y, mo, d, h, mi, s)
    except ValueError:
        return None
    return (dt - t0_local).total_seconds()

def track_progress(pts, t, s_axis, capture_s=None, lon=None, lat=None):
    """(arc position s in metres, pass index) for a shot. With a timed track and a capture time,
    the matching track point is the one closest IN TIME; pass 0 = outbound (before the turnaround,
    order by s ascending), pass 1 = inbound (order by s descending). Without timing: nearest space,
    always pass 0. The arc position is refined onto the neighbouring polyline segments, so shots a
    few metres apart order correctly."""
    import math
    if t is not None and capture_s is not None:
        i = int(np.argmin(np.abs(t - capture_s)))
    elif lon is not None:
        i = int(np.argmin((pts[:, 0] - lon) ** 2 + (pts[:, 1] - lat) ** 2))
    else:
        return 0.0, 0
    scale = np.array([111320.0 * math.cos(math.radians(pts[:, 1].mean())), 111320.0])
    q = np.array([lon, lat]) * scale if lon is not None else pts[i] * scale
    p = pts * scale
    best = (float(np.hypot(*(p[i] - q))), float(s_axis[i]))
    for j in (i - 1, i):                               # refine onto segments (j, j+1)
        if 0 <= j < len(pts) - 1:
            a, b = p[j], p[j + 1]
            d = b - a
            L2 = float(d @ d)
            tt = 0.0 if L2 == 0 else float(np.clip((q - a) @ d / L2, 0, 1))
            proj = a + tt * d
            dist = float(np.hypot(*(proj - q)))
            if dist < best[0]:
                best = (dist, float(s_axis[j] + tt * math.sqrt(L2)))
    turn = int(np.argmax(s_axis))
    return best[1], (0 if i <= turn or s_axis[turn] == 0 else 1)

class Candidate:
    """One segment that can donate windows; windows are cut from its front, best-quality first strip."""

    def __init__(self, video, fps, seg, feats, cuts_dir):
        self.video, self.fps, self.seg = video, fps, seg
        self.cuts_dir = cuts_dir
        self.base = os.path.basename(video)
        a, b = int(seg['start'] * fps), int(seg['end'] * fps)
        sl = slice(a, b + 1)
        self.rich = richness(feats, sl, seg['base'])
        self.jit_mean = float(np.nanmean(feats['jitter_burst'][sl]))
        self.cursor = a
        self.end = b
        self.move = seg['base'].partition('+')[0]

    @property
    def order_key(self):
        # video names embed the shooting timestamp: DJI_YYYYMMDDHHMMSS...
        return (os.path.basename(self.video), self.seg['start'])

    def desired_dur(self):
        d = DUR_LOW + (DUR_HIGH - DUR_LOW) * self.rich
        return float(np.clip(d, DUR_MIN, DUR_MAX))

    def take(self, dur):
        """Pop the next window of `dur` seconds from this segment, or None."""
        if self.end - self.cursor < dur * self.fps:
            return None
        w = dict(video=self.video, start=self.cursor / self.fps, end=(self.cursor + dur * self.fps) / self.fps,
                 duration=dur, rich=self.rich, base=self.seg['base'], label=self.seg['label'],
                 source=self.seg, fps=self.fps)
        self.cursor += int(dur * self.fps) + int(0.3 * self.fps)   # small gap between windows
        return w

# ---------------------------------------------------------------- assembly

def make_candidates(dirs, variant='_v3', track=None, track_path=None, dedup_r=60.0):
    from . import segment as seg
    cands = []
    for d in dirs:
        for jp in sorted(glob_jsons(d, variant)):
            base = os.path.basename(jp)[:-14]
            dj = json.load(open(jp))
            video = dj['video']
            tel_p = os.path.join(os.path.dirname(jp), base + '_telemetry.csv')
            flow_p = os.path.join(os.path.dirname(jp), base + '_flow.csv')
            if not (os.path.exists(tel_p) and os.path.exists(flow_p)):
                print(f'skip (no caches): {jp}', file=sys.stderr); continue
            tel = list(csv_rows(tel_p)); flow = list(csv_rows(flow_p))
            feats, _ = seg.features(tel, flow, dj['fps'])
            lat = np.array([float(r['lat']) for r in tel])
            lon = np.array([float(r['lon']) for r in tel])
            mlon, mlat = float(np.mean(lon)), float(np.mean(lat))
            for s in dj['segments']:
                if s['duration'] >= DUR_MIN + 0.5:
                    c = Candidate(video, dj['fps'], s, feats, os.path.dirname(jp))
                    c.mlon, c.mlat = mlon, mlat
                    cands.append(c)
    # calibrate richness against the candidate pool itself: percentile-scale to [0.15, 0.95]
    raw = sorted(c.rich for c in cands)
    for c in cands:
        rank = raw.index(c.rich) / max(1, len(raw) - 1)
        c.rich = 0.15 + 0.8 * rank
    # montage order = narrative order along the hike: pass 0 (outbound, by track position
    # ascending), then pass 1 (inbound, descending). Within a pass the track position corrects
    # flights that were flown out of walking sequence. Nearby same-subject flights are deduped.
    if track is not None and track[1] is not None:
        pts, t, s_axis = track
        t0_local = _t0_local(track_path)
        kept = []
        for c in sorted(cands, key=lambda c: c.order_key):
            cap = capture_time_s(c.video, t0_local) if t0_local else None
            c.s, c.pass_id = track_progress(pts, t, s_axis, cap, c.mlon, c.mlat)
            dup = False
            for k in kept:
                if abs(k.pass_id - c.pass_id) == 0 and _dist_m(k.mlon, k.mlat, c.mlon, c.mlat) < dedup_r:
                    dup = True
                    break
            if dup:
                print(f'dedup (same spot as an earlier flight): {os.path.basename(c.video)}', file=sys.stderr)
                continue
            kept.append(c)
        cands = kept
        sgn = np.where(np.array([c.pass_id for c in cands]) == 0, 1, -1)
        # bucket track position at 100 m: inside a bucket the trail projection cannot resolve
        # a few metres of GPS noise, so capture time decides; across buckets position wins
        cands.sort(key=lambda c: (c.pass_id, sgn[c.pass_id] * round(c.s / 100.0), c.order_key))
    else:
        # no timed track: dedup by position still applies, ordered by capture time
        kept = []
        for c in sorted(cands, key=lambda c: c.order_key):
            if any(_dist_m(k.mlon, k.mlat, c.mlon, c.mlat) < dedup_r for k in kept):
                print(f'dedup (same spot as an earlier flight): {os.path.basename(c.video)}', file=sys.stderr)
                continue
            kept.append(c)
        cands = kept
    return cands

def _dist_m(lon1, lat1, lon2, lat2):
    import math
    return math.hypot((lon2 - lon1) * math.cos(math.radians(lat1)), lat2 - lat1) * 111320

def _t0_local(track_path):
    """Hike start as a local naive datetime, parsed from the KML <name> ('2026-10-01 09:14:06')."""
    import re
    from datetime import datetime
    if not track_path:
        return None
    text = open(track_path, encoding='utf-8', errors='replace').read()
    nm = re.search(r'<name>([^<]+)</name>', text)
    if not nm:
        return None
    try:
        return datetime.fromisoformat(nm.group(1).strip())
    except ValueError:
        return None

def glob_jsons(d, variant='_v3'):
    import glob
    if variant:                                        # explicit variant: no silent fallback
        return sorted(glob.glob(os.path.join(d, f'*_cuts{variant}', '*_segments.json')))
    for suffix in ('_v3', ''):
        out = sorted(glob.glob(os.path.join(d, f'*_cuts{suffix}', '*_segments.json')))
        if out:
            return out
    return []

def csv_rows(path):
    import csv
    with open(path) as fh:
        return list(csv.DictReader(fh))

MOVE_CLASS = dict(orbit='rotate', spiral='rotate', push='straight', pull='straight',
                  truck='straight', diagonal='straight', vertical='other', static='other',
                  pan='other', tilt='other')

def max_per_flight(n_flights, music_dur, bars):
    """Windows per flight so the song is filled with as many distinct scenes as possible."""
    bar = bars[1] - bars[0] if len(bars) > 1 else 2.0
    need = max(1, int(music_dur / (2 * bar)))              # ~2-bar average shot
    return max(1, min(3, math.ceil(need / max(1, n_flights))))

def plan(cands, bars, max_slots=200, max_per_flight=2, music_end=None, max_flights=None, max_unit=None):
    """Location-block montage: candidates arrive in narrative order (hike pass + track position),
    flights become contiguous blocks (A-A-B-B), each flight donates up to max_per_flight windows.
    max_flights evenly samples across the blocks when the song is too short for every scene (and
    then max_unit caps slot length so the sampled scenes actually all fit). music_end caps the
    timeline so the silent tail after the song never exceeds ~1 s."""
    timeline = []
    i = 0
    bar = bars[1] - bars[0] if len(bars) > 1 else 2.0
    blocks = {}
    for c in cands:
        blocks.setdefault(c.video, []).append(c)
    if max_flights and len(blocks) > max_flights:
        keep = sorted(set(np.linspace(0, len(blocks) - 1, max_flights).round().astype(int)))
        names = list(blocks)
        blocks = {n: blocks[n] for n in (names[k] for k in keep)}
        max_unit = min(max_unit or 4, 2)
        print(f'sampling {len(blocks)} of {len(names)} flights to span the whole hike (slots <= {max_unit} bars)', file=sys.stderr)

    def alternate(segs):
        """Order a flight's segments so consecutive move classes differ when possible."""
        out, last = [], None
        pool = list(segs)
        while pool:
            pick = next((c for c in pool if MOVE_CLASS.get(c.move) != last), pool[0])
            out.append(pool.pop(pool.index(pick)))
            last = MOVE_CLASS.get(out[-1].move)
        return out

    for pass_no in range(2):                           # pass 2 allows a 3rd window if music remains
        for video, segs in blocks.items():
            if pass_no == 0:
                segs = alternate(sorted(segs, key=lambda c: c.seg['start']))
            else:
                segs = sorted(segs, key=lambda c: c.seg['start'])
            placed = 0
            for c in segs:
                if placed >= max_per_flight or i >= len(bars) - 1 or len(timeline) >= max_slots:
                    break
                units = [u for u in BAR_UNITS if u * bar >= DUR_MIN and u * bar <= DUR_MAX
                         and (max_unit is None or u <= max_unit)
                         and i + u <= len(bars) - 1
                         and (music_end is None or bars[i] + u * bar <= music_end + 1.0)]
                fit = [u for u in units if c.end - c.cursor >= u * bar * c.fps]
                if not fit:
                    continue
                u = min(fit, key=lambda u: abs(u * bar - c.desired_dur()))
                w = c.take(u * bar)
                if w is None:
                    continue
                w.update(slot=i, bar_unit=u, slot_start=bars[i], bar_s=bar, seq=len(timeline) + 1,
                         cuts_dir=c.cuts_dir, move=c.move)
                timeline.append(w)
                placed += 1
                i += u
            if i >= len(bars) - 1:
                break
        if i >= len(bars) - 1 or not any(c.end - c.cursor >= DUR_MIN * c.fps for c in cands):
            break
    return timeline

def _venc():
    import platform
    return (['-c:v', 'h264_videotoolbox', '-b:v', '12M'] if platform.system() == 'Darwin'
            else ['-c:v', 'libx264', '-preset', 'fast', '-crf', '20'])

def assemble(timeline, music_path, out_mp4, workdir):
    """Cut every window exactly (re-encoded to uniform 1080p30), concat, mix music with fade-out."""
    os.makedirs(workdir, exist_ok=True)
    parts = []
    for k, w in enumerate(timeline):
        part = os.path.join(workdir, f'part_{k:03d}.mp4')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f"{w['start']:.3f}", '-to', f"{w['end']:.3f}",
                        '-i', w['video'], '-vf', f'scale={W}:{H},fps={FPS},format=yuv420p',
                        '-an'] + _venc() + [part], check=True)
        parts.append(part)
        print(f"  [{k + 1}/{len(timeline)}] {os.path.basename(w['video'])} {w['start']:7.2f}-{w['end']:7.2f}s", file=sys.stderr)
    lst = os.path.join(workdir, 'concat.txt')
    with open(lst, 'w') as fh:
        for p in parts:
            fh.write(f"file '{p}'\n")
    silent = os.path.join(workdir, 'video.mp4')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', silent], check=True)
    total = sum(w['duration'] for w in timeline)
    fade = max(0.0, total - 2.5)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', silent, '-i', music_path,
                    '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                    '-af', f'afade=t=out:st={fade:.2f}:d=2.5', '-shortest', out_mp4], check=True)
    return out_mp4

def gps_map(timeline, out_png, track=None):
    """Flight tracks (grey) with the windows used in the draft overlaid in colour and numbered
    by their position in the edit; the hiking track, if given, is the dashed blue line."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tracks = {}                                        # cuts_dir -> (lat, lon)
    for w in timeline:
        key = w['cuts_dir']
        if key not in tracks:
            tel_p = os.path.join(key, os.path.splitext(os.path.basename(w['video']))[0] + '_telemetry.csv')
            if not os.path.exists(tel_p):
                tracks[key] = None
                continue
            rows = csv_rows(tel_p)
            lat = np.array([float(r['lat']) for r in rows])
            lon = np.array([float(r['lon']) for r in rows])
            tracks[key] = (lat, lon)
    lat0 = np.mean([np.mean(t[0]) for t in tracks.values() if t is not None])
    kx = math.cos(math.radians(lat0))
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.get_cmap('tab20')
    if track is not None and len(track):
        ax.plot(track[:, 0] * kx, track[:, 1], '--', color='tab:blue', lw=2.2, zorder=2, label='hike track')
    for t in tracks.values():
        if t is None: continue
        lat, lon = t
        ax.plot(lon[::8] * kx, lat[::8], color='0.75', lw=0.8, zorder=1)
    for w in timeline:
        t = tracks.get(w['cuts_dir'])
        if t is None: continue
        lat, lon = t
        fps = w.get('fps', 59.94)
        a, b = int(w['start'] * fps), int(w['end'] * fps)
        a, b = max(0, a), min(len(lat) - 1, b)
        if b <= a: continue
        col = cmap(w['seq'] % 20)
        ax.plot(lon[a:b:4] * kx, lat[a:b:4], color=col, lw=3, zorder=2)
        ax.annotate(str(w['seq']), (np.mean(lon[a:b]) * kx, np.mean(lat[a:b])),
                    fontsize=9, fontweight='bold', color=col, ha='center', va='center', zorder=3)
    ax.set_aspect(1.0)
    ax.set_title(f"{len(timeline)} shots used in draft (numbered) — grey: flights, dashed: hike track")
    ax.set_xlabel('lon (scaled)'); ax.set_ylabel('lat')
    if track is not None and len(track):
        ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png

def build(music_path, dirs, out_mp4, workdir=None, track_path=None):
    bars, bpm, dur = beat_grid(music_path)
    print(f'music: {bpm:.1f} BPM, {len(bars)} bars, {dur:.0f}s', file=sys.stderr)
    track = load_track(track_path) if track_path else None
    if track is not None:
        print(f'hike track: {len(track[0])} points, timed={track[1] is not None} — montage follows track progress', file=sys.stderr)
    cands = make_candidates(dirs, track=track, track_path=track_path)
    print(f'{len(cands)} candidate segments (from _cuts_v3)', file=sys.stderr)
    n_flights = len({c.video for c in cands})
    mpf = max_per_flight(n_flights, dur, bars)
    budget = max(1, round(dur / (2 * (bars[1] - bars[0]))))   # slots at ~2-bar shots
    timeline = plan(cands, bars, music_end=dur, max_per_flight=mpf,
                    max_flights=max(1, min(n_flights, budget // mpf)))
    total = sum(w['duration'] for w in timeline)
    vids = {w['video'] for w in timeline}
    order = 'shooting order (track drawn on the GPS map)'
    print(f'plan: {len(timeline)} shots, {total:.0f}s of video (music {dur:.0f}s), {len(vids)} flights, order={order}', file=sys.stderr)
    workdir = workdir or tempfile.mkdtemp(prefix='dronesift_draft_')
    assemble(timeline, music_path, out_mp4, workdir)
    ej = os.path.splitext(out_mp4)[0] + '_edit.json'
    json.dump(dict(music=music_path, bpm=bpm, timeline=timeline,
                   total_s=total, shots=len(timeline)), open(ej, 'w'), indent=1, ensure_ascii=False)
    try:
        gps_map(timeline, os.path.splitext(out_mp4)[0] + '_gps.png', track[0] if track else None)
    except Exception as e:
        print(f'gps map failed: {e}', file=sys.stderr)
    return out_mp4, timeline

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('music'); ap.add_argument('dirs', nargs='+')
    ap.add_argument('-o', default='draft.mp4')
    ap.add_argument('--workdir', default=None)
    ap.add_argument('--track', default=None, help='KML hiking track: order scenes by track progress')
    a = ap.parse_args()
    out, tl = build(a.music, a.dirs, a.o, a.workdir, a.track)
    print(out)

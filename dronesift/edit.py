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
        self.a, self.b = a, b                           # full span, for the GPS map
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
    """Returns (selected, all): `all` is every stable segment in narrative order — drawn on the
    GPS map; `selected` is the deduped, slot-length-capable subset used for planning."""
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
                c = Candidate(video, dj['fps'], s, feats, os.path.dirname(jp))
                c.mlon, c.mlat = mlon, mlat
                cands.append(c)
    if not cands:
        return [], []
    # calibrate richness against the candidate pool itself: percentile-scale to [0.15, 0.95]
    raw = sorted(c.rich for c in cands)
    for c in cands:
        rank = raw.index(c.rich) / max(1, len(raw) - 1)
        c.rich = 0.15 + 0.8 * rank
    # montage order = narrative order along the hike: pass 0 (outbound, by track position
    # ascending), then pass 1 (inbound, descending). Within a pass the track position corrects
    # flights that were flown out of walking sequence. Nearby same-subject flights are deduped
    # from PLANNING only — the map still shows every segment.
    if track is not None and track[1] is not None:
        pts, t, s_axis = track
        t0_local = _t0_local(track_path)
        for c in cands:
            cap = capture_time_s(c.video, t0_local) if t0_local else None
            c.s, c.pass_id = track_progress(pts, t, s_axis, cap, c.mlon, c.mlat)
        sgn = np.where(np.array([c.pass_id for c in cands]) == 0, 1, -1)
        # bucket track position at 100 m: inside a bucket the trail projection cannot resolve
        # a few metres of GPS noise, so capture time decides; across buckets position wins
        cands.sort(key=lambda c: (c.pass_id, sgn[c.pass_id] * round(c.s / 100.0), c.order_key))
    else:
        cands.sort(key=lambda c: c.order_key)
    for c in cands:
        c.pass_id = getattr(c, 'pass_id', 0)
    selected = []
    for c in cands:
        if c.seg['duration'] < DUR_MIN + 0.5:
            continue
        if any(k.pass_id == c.pass_id and _dist_m(k.mlon, k.mlat, c.mlon, c.mlat) < dedup_r
               for k in selected):
            print(f'dedup (same spot as an earlier flight): {os.path.basename(c.video)}', file=sys.stderr)
            continue
        selected.append(c)
    return selected, cands

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

def music_slots(bars, music_dur, pattern=(2, 2, 4), min_dur=DUR_MIN):
    """Slot list (start_s, duration_s) from the beat grid — the music decides where every clip
    begins and how long it is; footage is then cut to fit each slot. The bar-count pattern
    repeats over the song's bar lines and the final slot is trimmed to the song end, so the
    silent tail after the music never exceeds ~1 s."""
    slots = []
    i = 0
    while i < len(bars) - 1:
        u = pattern[len(slots) % len(pattern)]
        t0 = bars[i]
        rem = music_dur - t0
        if i + u > len(bars) - 1 or t0 + u * (bars[1] - bars[0] if len(bars) > 1 else 2.0) > music_dur:
            if rem >= min_dur:
                slots.append((t0, rem))                # final partial slot, cut at the song end
            break
        t1 = bars[i + u]
        slots.append((t0, t1 - t0))
        i += u
    return slots

def plan_slots(cands, slots, max_per_flight=2):
    """Fill the music slots in narrative order: each slot cuts its exact duration from the next
    candidate in line that still has enough footage (slots first, footage second — not 'segments
    that happen to match the length'). When there are more flights than slots, flights are
    sampled evenly along the hike so every part of the trail is represented, one window each."""
    timeline = []
    used = {}
    queue = list(cands)
    flights = list(dict.fromkeys(c.video for c in queue))
    if len(flights) > len(slots):
        n = min(len(flights), len(slots) + 3)          # buffer: some sampled flights are too short
        keep = sorted(set(np.linspace(0, len(flights) - 1, n).round().astype(int)))
        keepv = {flights[k] for k in keep}
        queue = [c for c in queue if c.video in keepv]
        max_per_flight = 1
        print(f'sampling {len(keepv)} of {len(flights)} flights to span the whole hike', file=sys.stderr)
    for si, (t0, dur) in enumerate(slots):
        pick = None
        for pos, c in enumerate(queue):
            if used.get(c.video, 0) >= max_per_flight:
                continue
            if c.end - c.cursor >= dur * c.fps:
                pick = pos
                break
        if pick is None:
            break
        c = queue[pick]
        w = c.take(dur)
        if w is None:
            queue.pop(pick)                            # exhausted: drop and retry this slot
            continue
        w.update(slot=si, slot_start=t0, seq=len(timeline) + 1, cuts_dir=c.cuts_dir, move=c.move)
        used[c.video] = used.get(c.video, 0) + 1
        timeline.append(w)
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

CLASS_COLOR = {'rotate': '#e8710a', 'straight': '#2a78d6', 'other': '#1baf7a'}

def gps_map(timeline, cands, out_png, track=None):
    """Every _cuts_v3 stable segment coloured by move class; the shots used in the draft drawn
    bold with their sequence number and an arrow showing the camera's travel direction (straight
    arrow for line moves, arc arrow for rotations)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    for fname in ('PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS', 'STHeiti',
                  'Microsoft YaHei', 'Noto Sans CJK SC'):
        try:
            if font_manager.findfont(fname, fallback_to_default=False):
                plt.rcParams['font.sans-serif'] = [fname]
                break
        except Exception:
            continue
    plt.rcParams['axes.unicode_minus'] = False
    tracks = {}                                        # cuts_dir -> (lat, lon)

    def get_track(cuts_dir, video):
        if cuts_dir not in tracks:
            p = os.path.join(cuts_dir, os.path.splitext(os.path.basename(video))[0] + '_telemetry.csv')
            if os.path.exists(p):
                rows = csv_rows(p)
                tracks[cuts_dir] = (np.array([float(r['lat']) for r in rows]),
                                    np.array([float(r['lon']) for r in rows]))
            else:
                tracks[cuts_dir] = None
        return tracks[cuts_dir]

    lats = [c.mlat for c in cands] or [43.0]
    kx = math.cos(math.radians(float(np.mean(lats))))
    fig, ax = plt.subplots(figsize=(12, 10))
    if track is not None and len(track):
        ax.plot(track[:, 0] * kx, track[:, 1], '--', color='tab:blue', lw=2.2, zorder=2)
    fps0 = cands[0].fps if cands else 59.94
    for c in cands:                                    # the whole stable-footage pool
        tr = get_track(c.cuts_dir, c.video)
        if tr is None: continue
        lat, lon = tr
        a, b = max(0, c.a), min(len(lat) - 1, c.b)
        if b <= a: continue
        col = CLASS_COLOR.get(MOVE_CLASS.get(c.move, 'other'), '#1baf7a')
        ax.plot(lon[a:b:2] * kx, lat[a:b:2], color=col, lw=1.4, alpha=0.8, zorder=3)
    for w in timeline:                                 # the shots in the final cut
        tr = get_track(w['cuts_dir'], w['video'])
        if tr is None: continue
        lat, lon = tr
        fps = w.get('fps', fps0)
        a, b = int(w['start'] * fps), int(w['end'] * fps)
        a, b = max(0, a), min(len(lat) - 1, b)
        if b <= a: continue
        ax.plot(lon[a:b] * kx, lat[a:b], color='crimson', lw=4, zorder=4)
        i1, i2 = a + (b - a) // 4, a + 3 * (b - a) // 4
        x1, y1 = lon[i1] * kx, lat[i1]
        x2, y2 = lon[i2] * kx, lat[i2]
        conn = 'arc3,rad=0.4' if MOVE_CLASS.get(w['move']) == 'rotate' else 'arc3,rad=0'
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), zorder=5,
                    arrowprops=dict(arrowstyle='-|>', color='crimson', lw=2,
                                    mutation_scale=22, connectionstyle=conn))
        ax.annotate(str(w['seq']), (lon[(a + b) // 2] * kx, lat[(a + b) // 2]), zorder=6,
                    fontsize=14, fontweight='bold', color='crimson', ha='center', va='center',
                    path_effects=[pe.withStroke(linewidth=4, foreground='white')])
    handles = [Line2D([], [], color=c, lw=2.5, label=l) for l, c in
               (('旋转 orbit/spiral', CLASS_COLOR['rotate']), ('直线 push/pull/truck', CLASS_COLOR['straight']),
                ('其他 升降/悬停/摇', CLASS_COLOR['other']))]
    handles.append(Line2D([], [], color='crimson', lw=3.5, label='成片片段(序号+方向)'))
    if track is not None and len(track):
        handles.append(Line2D([], [], color='tab:blue', ls='--', lw=2.2, label='徒步轨迹'))
    ax.legend(handles=handles, loc='best', fontsize=10)
    ax.set_aspect(1.0)
    ax.set_title(f"稳定运镜分布 + 成片 {len(timeline)} 段(序号 = 成片位置,箭头 = 运镜方向)")
    ax.set_xlabel('lon (scaled)'); ax.set_ylabel('lat')
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png

def build(music_path, dirs, out_mp4, workdir=None, track_path=None):
    bars, bpm, dur = beat_grid(music_path)
    print(f'music: {bpm:.1f} BPM, {len(bars)} bars, {dur:.0f}s', file=sys.stderr)
    track = load_track(track_path) if track_path else None
    if track is not None:
        print(f'hike track: {len(track[0])} points — drawn on the GPS map', file=sys.stderr)
    cands, cands_all = make_candidates(dirs, track=track, track_path=track_path)
    print(f'{len(cands)} planning candidates, {len(cands_all)} stable segments on the map (from _cuts_v3)', file=sys.stderr)
    slots = music_slots(bars, dur)
    print(f'music slots: {len(slots)} ({", ".join(f"{d:.1f}s" for _, d in slots)})', file=sys.stderr)
    timeline = plan_slots(cands, slots)
    total = sum(w['duration'] for w in timeline)
    vids = {w['video'] for w in timeline}
    print(f'plan: {len(timeline)} shots, {total:.1f}s of video (music {dur:.1f}s), {len(vids)} flights', file=sys.stderr)
    workdir = workdir or tempfile.mkdtemp(prefix='dronesift_draft_')
    assemble(timeline, music_path, out_mp4, workdir)
    ej = os.path.splitext(out_mp4)[0] + '_edit.json'
    json.dump(dict(music=music_path, bpm=bpm, slots=slots, timeline=timeline,
                   total_s=total, shots=len(timeline)), open(ej, 'w'), indent=1, ensure_ascii=False)
    try:
        gps_map(timeline, cands_all, os.path.splitext(out_mp4)[0] + '_gps.png', track[0] if track else None)
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

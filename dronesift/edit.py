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

def make_candidates(dirs, variant='_v3'):
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
            for s in dj['segments']:
                if s['duration'] >= DUR_MIN + 0.5:
                    cands.append(Candidate(video, dj['fps'], s, feats, os.path.dirname(jp)))
    # calibrate richness against the candidate pool itself: percentile-scale to [0.15, 0.95]
    raw = sorted(c.rich for c in cands)
    for c in cands:
        rank = raw.index(c.rich) / max(1, len(raw) - 1)
        c.rich = 0.15 + 0.8 * rank
    # montage order = shooting order; quality only feeds the richness->duration map
    cands.sort(key=lambda c: c.order_key)
    return cands

def glob_jsons(d, variant='_v3'):
    import glob
    for suffix in (variant, ''):
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

def plan(cands, bars, max_slots=200, max_per_video=2):
    """Chronological montage: candidates are consumed in shooting order, so the edit follows the
    day as it was flown. Two soft rules bend the order without violating it:
    - rotate/straight interleave: if the next candidate repeats the previous shot's move CLASS
      (rotation vs straight line vs other), look ahead for a different class; the skipped
      candidate stays in line and surfaces later;
    - scene coverage: one window per segment, at most max_per_video per flight, so as many distinct
      scenes as possible appear."""
    timeline = []
    i = 0
    prev_class = None
    used = {}
    bar = bars[1] - bars[0] if len(bars) > 1 else 2.0
    queue = list(cands)
    while i < len(bars) - 1 and len(timeline) < max_slots and queue:
        t0 = bars[i]
        units = [u for u in BAR_UNITS if u * bar >= DUR_MIN and u * bar <= DUR_MAX
                 and i + u <= len(bars) - 1]
        if not units:
            break
        # next candidate in shooting order; if it repeats the previous move class, look ahead
        # (bounded) for a different class; the deferred candidate surfaces later, chronology bends
        # only within the lookahead depth
        pick = None
        viable = 0
        for pos, c in enumerate(queue):
            if used.get(c.video, 0) >= max_per_video:
                continue
            cls = MOVE_CLASS.get(c.move, 'other')
            viable += 1
            if pos == 0 or prev_class is None or cls != prev_class or viable >= 8:
                pick = pos
                break
        if pick is None:                                   # everything left is capped
            break
        c = queue.pop(pick)
        want = c.desired_dur()
        fit = [u for u in units if c.end - c.cursor >= u * bar * c.fps]
        if not fit:
            continue
        u = min(fit, key=lambda u: abs(u * bar - want))
        dur = u * bar
        w = c.take(dur)
        if w is None:
            continue
        w.update(slot=i, bar_unit=u, slot_start=t0, bar_s=bar, seq=len(timeline) + 1,
                 cuts_dir=c.cuts_dir, move=c.move)
        timeline.append(w)
        used[c.video] = used.get(c.video, 0) + 1
        prev_class = MOVE_CLASS.get(c.move, 'other')
        i += u
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

def gps_map(timeline, out_png):
    """Flight tracks (grey) with the windows used in the draft overlaid in colour and numbered
    by their position in the edit."""
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
    ax.set_title(f"{len(timeline)} shots used in draft (numbered) — grey: all flight tracks")
    ax.set_xlabel('lon (scaled)'); ax.set_ylabel('lat')
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png

def build(music_path, dirs, out_mp4, workdir=None):
    bars, bpm, dur = beat_grid(music_path)
    print(f'music: {bpm:.1f} BPM, {len(bars)} bars, {dur:.0f}s', file=sys.stderr)
    cands = make_candidates(dirs)
    print(f'{len(cands)} candidate segments', file=sys.stderr)
    timeline = plan(cands, bars)
    total = sum(w['duration'] for w in timeline)
    vids = {w['video'] for w in timeline}
    print(f'plan: {len(timeline)} shots, {total:.0f}s of video, {len(vids)} flights, order=shooting order', file=sys.stderr)
    workdir = workdir or tempfile.mkdtemp(prefix='dronesift_draft_')
    assemble(timeline, music_path, out_mp4, workdir)
    ej = os.path.splitext(out_mp4)[0] + '_edit.json'
    json.dump(dict(music=music_path, bpm=bpm, timeline=timeline,
                   total_s=total, shots=len(timeline)), open(ej, 'w'), indent=1, ensure_ascii=False)
    try:
        gps_map(timeline, os.path.splitext(out_mp4)[0] + '_gps.png')
    except Exception as e:
        print(f'gps map failed: {e}', file=sys.stderr)
    return out_mp4, timeline

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('music'); ap.add_argument('dirs', nargs='+')
    ap.add_argument('-o', default='draft.mp4')
    ap.add_argument('--workdir', default=None)
    a = ap.parse_args()
    out, tl = build(a.music, a.dirs, a.o, a.workdir)
    print(out)

"""dronesift: find stable / continuous-move segments in drone footage and cut them.

usage: python -m dronesift VIDEO.MP4 [--out DIR] [--mode copy|encode|prores|none] [--hwaccel auto] ...
"""
import argparse, csv, json, os, sys, shutil
from . import djmd, flow as flowmod, segment as seg, cut as cutmod, plot, srt as srtmod

def telemetry_rows(video, srt_path, cache):
    if os.path.exists(cache): return list(csv.DictReader(open(cache)))
    rows = []
    try: rows = djmd.parse(video)
    except Exception as e: print('djmd parse failed:', e, file=sys.stderr)
    if rows and any(float(r['qw'] or 0) != 0 for r in rows[:200]):
        djmd.write_csv(rows, cache); return list(csv.DictReader(open(cache)))
    if not srt_path or not os.path.exists(srt_path):
        sys.exit('no usable djmd track and no SRT file found')
    print('no djmd attitude data; falling back to SRT (no gimbal angles: pan/orbit detection disabled)', file=sys.stderr)
    rows = srtmod.to_telemetry(srtmod.parse(srt_path)); djmd.write_csv(rows, cache); return list(csv.DictReader(open(cache)))

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('video'); ap.add_argument('--srt', help='SRT path (default: same name as video)')
    ap.add_argument('--out', help='output dir (default: <video>_cuts next to the video)')
    ap.add_argument('--cache-dir', help='read existing telemetry/flow/thumbnail caches from this directory')
    ap.add_argument('--mode', choices=['copy', 'encode', 'prores', 'none'], default='copy', help='copy=keyframe-snapped lossless; encode=frame-exact HEVC 10-bit; prores=ProRes 422 HQ exact; none=EDL/JSON only')
    ap.add_argument('--bitrate', default='80M')
    ap.add_argument('--hwaccel', default='auto', choices=['auto', 'none', 'videotoolbox', 'vaapi', 'cuda'],
                    help='hardware decoder for the analysis/thumbnail passes (auto: videotoolbox on macOS, software elsewhere)')
    for k, v in seg.DEFAULT_THR.items():
        ap.add_argument('--' + k.replace('_', '-'), type=float, default=v, help=f'(default {v})')
    a = ap.parse_args()
    video = os.path.abspath(a.video); base = os.path.splitext(os.path.basename(video))[0]
    out = a.out or os.path.join(os.path.dirname(video), base + '_cuts'); os.makedirs(out, exist_ok=True)
    srt_path = a.srt or os.path.splitext(video)[0] + '.SRT'
    if not os.path.exists(srt_path): srt_path = os.path.splitext(video)[0] + '.srt'
    thr = {k: getattr(a, k) for k in seg.DEFAULT_THR}
    # Each result directory remains self-contained; reusable raw measurements
    # are copied, never modified when trying a new segmentation policy.
    if a.cache_dir:
        for suffix in ('_telemetry.csv', '_flow.csv', '_thumbs.jpg'):
            src = os.path.join(a.cache_dir, base + suffix); dst = os.path.join(out, base + suffix)
            if os.path.isfile(src) and not os.path.exists(dst): shutil.copy2(src, dst)
    tel = telemetry_rows(video, srt_path, os.path.join(out, base + '_telemetry.csv'))
    flow_csv = os.path.join(out, base + '_flow.csv')
    if not os.path.exists(flow_csv):
        print('analysing image motion (optical flow)...', file=sys.stderr); flowmod.analyze(video, flow_csv, hwaccel=a.hwaccel)
    flow = list(csv.DictReader(open(flow_csv)))
    fps = seg.FPS
    try:
        import subprocess
        r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', video], capture_output=True, text=True).stdout.strip()
        num, den = r.split('/'); fps = float(num) / float(den)
    except Exception: pass
    f, n = seg.features(tel, flow, fps, rev_amp=thr['motion_rev_amp'])
    keep, labels = seg.classify_frames(f, thr, fps)
    segs = seg.segments(f, keep, labels, thr, fps)
    js = os.path.join(out, base + '_segments.json')
    retained = sum(s['duration'] for s in segs)
    selection = dict(source_duration=n / fps, retained_duration=retained,
                     retention=retained / (n / fps), kept_frame_fraction=float(keep.mean()))
    events = [dict(reason=name, start=i / fps, end=(j + 1) / fps)
              for name, mask in f['disturbances'].items() for i, j, value in seg._runs(mask) if value]
    json.dump(dict(video=video, fps=fps, frames=n, thresholds=thr, segments=segs,
                   selection=selection, disturbances=events), open(js, 'w'), indent=1)
    cutmod.write_edl(segs, video, os.path.join(out, base + '.edl'), fps)
    # edit-ready shots (<= max_shot_s): own json/EDL + clips in <out>/shots, and the FCPXML uses them
    shots = seg.shots(segs, f, labels, thr, fps) if thr.get('max_shot_s') else []
    if shots:
        json.dump(dict(video=video, fps=fps, frames=n, thresholds=thr, shots=shots),
                  open(os.path.join(out, base + '_shots.json'), 'w'), indent=1)
        cutmod.write_edl(shots, video, os.path.join(out, base + '_shots.edl'), fps, digits=3)
    from . import fcpxml
    fcpxml.write(shots or segs, video, os.path.join(out, base + '.fcpxml'), fps)
    plot.timeline(f, keep, segs, thr, os.path.join(out, base + '_timeline.png'), fps, title=base)
    try:
        from . import viz; viz.build(video, os.path.join(out, base + '_telemetry.csv'), flow_csv, js, os.path.join(out, base + '_viz.html'), hwaccel=a.hwaccel)
    except Exception as e: print('viz failed:', e, file=sys.stderr)
    print(f"kept frames {keep.mean() * 100:.1f}%  ->  {len(segs)} segments, {sum(s['duration'] for s in segs):.1f}s"
          + (f"; {len(shots)} shots" if shots else ''))
    for s in segs:
        print(f"  {seg.hms(s['start'])} - {seg.hms(s['end'])}  {s['duration']:5.1f}s  {s['label']:28s} jitter={s['jitter_px']:.1f}px/s v={s['speed_mps']:.1f}m/s yaw={s['yaw_rate_dps']:+.1f}°/s climb={s['climb_mps']:+.1f}m/s" + (f" r≈{s['radius_m']:.0f}m" if s.get('radius_m') else ''))
    if a.mode != 'none':
        cutmod.cut(video, segs, out, a.mode, fps, bitrate=a.bitrate)
        exported = json.load(open(os.path.join(out, base + '_exports.json')))
        print(f"actual exported {exported['total_duration']:.3f}s / {n / fps:.3f}s = "
              f"{exported['total_duration'] / (n / fps) * 100:.2f}% (verified clip durations)")
        if shots:
            cutmod.cut(video, shots, os.path.join(out, 'shots'), a.mode, fps, bitrate=a.bitrate, digits=3)
    print('outputs in', out)

if __name__ == '__main__':
    main()

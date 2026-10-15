"""Write clips / EDL from a segments JSON."""
import copy, json, os, platform, subprocess, sys

_TIME_EPS = 1e-6  # ffprobe prints timestamps to six decimal places

def video_packets(mp4, fps=59.94):
    """Video packets in decode order, including their presentation intervals."""
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                          '-show_entries', 'packet=pts_time,dts_time,duration_time,flags',
                          '-of', 'json', mp4], capture_output=True, text=True, check=True)
    packets = []
    for p in json.loads(out.stdout).get('packets', []):
        if 'pts_time' not in p:
            raise ValueError(f'video packet has no presentation timestamp: {mp4}')
        pts = float(p['pts_time'])
        duration = float(p.get('duration_time', 0)) or 1 / fps
        packets.append(dict(pts=pts, dts=float(p.get('dts_time', pts)),
                            duration=duration, keyframe='K' in p.get('flags', '')))
    return packets

def copy_plan(packets, start, end):
    """The first decodable packet prefix wholly inside [start, end).

    Counting packets instead of relying on ffmpeg's timestamp stop rule also
    protects the end of a cut when B frames are presented out of decode order.
    It may shorten such a cut, since a future reference frame can precede the
    last requested frames in the bitstream.
    """
    first = next((i for i, p in enumerate(packets)
                  if p['keyframe'] and p['pts'] >= start - _TIME_EPS
                  and p['pts'] + p['duration'] <= end + _TIME_EPS), None)
    if first is None:
        return []
    selected = []
    source_start = packets[first]['pts']
    for p in packets[first:]:
        if p['pts'] < source_start - _TIME_EPS or p['pts'] + p['duration'] > end + _TIME_EPS:
            break
        selected.append(p)
    return selected

def _verify_copy(expected, actual):
    """A remux must preserve every chosen packet's presentation time and length."""
    if len(expected) != len(actual) or not actual:
        return False
    shift = actual[0]['pts'] - expected[0]['pts']
    return all(abs((q['pts'] - p['pts']) - shift) <= 2 * _TIME_EPS
               and abs(q['duration'] - p['duration']) <= 2 * _TIME_EPS
               for p, q in zip(expected, actual))

def keyframes(mp4):
    return sorted(p['pts'] for p in video_packets(mp4) if p['keyframe'])

def tc(t, fps):
    fr = int(round(t * fps)); ff = int(round(fps))
    return f"{fr // (3600 * ff):02d}:{fr // (60 * ff) % 60:02d}:{fr // ff % 60:02d}:{fr % ff:02d}"

def write_edl(segs, mp4, path, fps, digits=2):
    name = os.path.basename(mp4); rec = 0.0
    with open(path, 'w') as fh:
        fh.write(f"TITLE: {os.path.splitext(name)[0]}_cuts\nFCM: NON-DROP FRAME\n\n")
        for i, s in enumerate(segs, 1):
            fh.write(f"{i:0{digits}d}  AX       V     C        {tc(s['start'], fps)} {tc(s['end'], fps)} {tc(rec, fps)} {tc(rec + s['duration'], fps)}\n")
            fh.write(f"* FROM CLIP NAME: {name}\n* COMMENT: {s['label']} {s['duration']:.1f}s\n\n")
            rec += s['duration']

def encoder_args(mode, bitrate):
    """Encoder for the exact-cut modes: videotoolbox on macOS, portable ffmpeg encoders elsewhere."""
    if mode != 'prores':
        enc = (['-c:v', 'hevc_videotoolbox', '-profile:v', 'main10'] if platform.system() == 'Darwin'
               else ['-c:v', 'libx265', '-preset', 'slow'])
        return enc + ['-pix_fmt', 'p010le', '-b:v', bitrate, '-tag:v', 'hvc1',
                      '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709']
    return (['-c:v', 'prores_videotoolbox', '-profile:v', '3', '-pix_fmt', 'p210le'] if platform.system() == 'Darwin'
            else ['-c:v', 'prores_ks', '-profile:v', '3', '-pix_fmt', 'yuv422p10le'])   # ProRes 422 HQ 10-bit

def playback_speed_metadata(segment, source_start, source_end):
    """Copy a catalog advisory without claiming to reassess an exported subset."""
    profile = copy.deepcopy(segment.get('playback_speed_profile'))
    return dict(playback_speed_profile=profile,
                max_playback_speed=(profile or {}).get('max_playback_speed'),
                playback_speed_profile_scope=dict(
                    basis='complete_catalog_segment',
                    source_start=(profile or {}).get('source_start', segment['start']),
                    source_end=(profile or {}).get('source_end', segment['end']),
                    exported_source_start=source_start, exported_source_end=source_end,
                    reassessed=False,
                    subset_policy='inherits_catalog_advisory_ceiling_without_reassessment',
                    exact_window_assessment_required_for_playback=True,
                    catalog_profile_available=isinstance(profile, dict)))

def cut(mp4, segs, outdir, mode='copy', fps=59.94, bitrate='80M', digits=2):
    os.makedirs(outdir, exist_ok=True); base = os.path.splitext(os.path.basename(mp4))[0]
    packets = video_packets(mp4, fps) if mode == 'copy' and segs else None
    paths = []; clips = []; skipped = []
    for i, s in enumerate(segs, 1):
        start, end = s['start'], s['end']
        selected = None
        if mode == 'copy':
            selected = copy_plan(packets, start, end)
            if not selected:
                reason = 'no complete frame beginning at an in-range keyframe'
                skipped.append(dict(segment=i, requested_start=start, requested_end=end, reason=reason))
                print(f'skipped segment {i}: {reason} ({start:.3f}s - {end:.3f}s)')
                continue
            start = selected[0]['pts']
        out = os.path.join(outdir, f"{base}_{i:0{digits}d}_{s['label'].replace('>', '-')}_{start:.1f}s.mp4")
        cmd = ['ffmpeg', '-v', 'error', '-y', '-ss', f'{start:.6f}', '-i', mp4, '-map', '0:v:0', '-an']
        if mode == 'copy':
            cmd += ['-c', 'copy', '-frames:v', str(len(selected)), '-avoid_negative_ts', 'make_zero']
        else:
            cmd += ['-t', f'{end - start:.6f}'] + encoder_args(mode, bitrate)
            if mode == 'prores': out = out[:-4] + '.mov'
        cmd += ['-movflags', '+faststart', out]
        subprocess.run(cmd, check=True)
        actual = video_packets(out, fps)
        verified = _verify_copy(selected, actual) if selected is not None else bool(actual)
        if not verified:
            raise RuntimeError(f'exported packet timestamps do not match the safe cut plan: {out}; use encode mode')
        source_end = max(p['pts'] + p['duration'] for p in selected) if selected else end
        duration = max(p['pts'] + p['duration'] for p in actual) - min(p['pts'] for p in actual)
        clips.append(dict(path=os.path.basename(out), requested_start=s['start'], requested_end=s['end'],
                          source_start=start, source_end=source_end, duration=duration,
                          frames=len(actual), verified=verified,
                          **playback_speed_metadata(s, start, source_end)))
        paths.append(out)
        print(f"wrote {out}  ({duration:.2f}s; source {start:.3f}s - {source_end:.3f}s)")
    manifest = dict(video=mp4, mode=mode, clips=clips, skipped=skipped,
                    total_duration=sum(c['duration'] for c in clips))
    with open(os.path.join(outdir, base + '_exports.json'), 'w') as fh:
        json.dump(manifest, fh, indent=1)
    return paths

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('mp4'); ap.add_argument('segments_json'); ap.add_argument('outdir')
    ap.add_argument('--mode', choices=['copy', 'encode', 'prores'], default='copy')
    ap.add_argument('--bitrate', default='80M')
    a = ap.parse_args()
    d = json.load(open(a.segments_json)); segs = d['segments']; fps = d['fps']
    os.makedirs(a.outdir, exist_ok=True)
    write_edl(segs, a.mp4, os.path.join(a.outdir, os.path.splitext(os.path.basename(a.mp4))[0] + '.edl'), fps)
    cut(a.mp4, segs, a.outdir, a.mode, fps, bitrate=a.bitrate)

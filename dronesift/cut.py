"""Write clips / EDL from a segments JSON."""
import json, os, platform, subprocess, sys

def keyframes(mp4):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                          '-show_entries', 'packet=pts_time,flags', '-of', 'csv=p=0', mp4], capture_output=True, text=True).stdout
    return sorted(float(l.split(',')[0]) for l in out.splitlines() if l and 'K' in l.split(',')[1])

def tc(t, fps):
    fr = int(round(t * fps)); ff = int(round(fps))
    return f"{fr // (3600 * ff):02d}:{fr // (60 * ff) % 60:02d}:{fr // ff % 60:02d}:{fr % ff:02d}"

def write_edl(segs, mp4, path, fps):
    name = os.path.basename(mp4); rec = 0.0
    with open(path, 'w') as fh:
        fh.write(f"TITLE: {os.path.splitext(name)[0]}_cuts\nFCM: NON-DROP FRAME\n\n")
        for i, s in enumerate(segs, 1):
            fh.write(f"{i:03d}  AX       V     C        {tc(s['start'], fps)} {tc(s['end'], fps)} {tc(rec, fps)} {tc(rec + s['duration'], fps)}\n")
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

def cut(mp4, segs, outdir, mode='copy', fps=59.94, bitrate='80M'):
    os.makedirs(outdir, exist_ok=True); base = os.path.splitext(os.path.basename(mp4))[0]
    kf = keyframes(mp4) if mode == 'copy' else None
    paths = []
    for i, s in enumerate(segs, 1):
        start, end = s['start'], s['end']
        if mode == 'copy':   # snap inward to keyframes so the clip starts cleanly with no re-encode
            start = min([k for k in kf if k >= start - 1e-3] or [start])
            if start >= end: continue
        out = os.path.join(outdir, f"{base}_{i:02d}_{s['label'].replace('>', '-')}_{start:.1f}s.mp4")
        cmd = ['ffmpeg', '-v', 'error', '-y', '-ss', f'{start:.6f}', '-to', f'{end:.6f}', '-i', mp4, '-map', '0:v:0', '-an']
        if mode == 'copy':
            cmd += ['-c', 'copy', '-avoid_negative_ts', 'make_zero']
        else:
            cmd += encoder_args(mode, bitrate)
            if mode == 'prores': out = out[:-4] + '.mov'
        cmd += ['-movflags', '+faststart', out]
        subprocess.run(cmd, check=True); paths.append(out)
        print(f"wrote {out}  ({end - start:.2f}s)")
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

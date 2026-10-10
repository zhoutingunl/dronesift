"""Per-frame global camera motion from optical flow on a downscaled decode.

Outputs CSV with, per frame: n_tracks, dx, dy (px at analysis res), da (rad), ds (log scale),
inlier_ratio, residual (median px error of inliers under the fitted similarity), lap_var (sharpness),
mean_luma, mdx, mdy (median feature displacement: robust to parallax, used for jitter).
"""
import csv, platform, subprocess, sys
import numpy as np, cv2

def hwaccel_default():
    """Out-of-the-box hardware decoder by platform (None = software decode)."""
    return 'videotoolbox' if platform.system() == 'Darwin' else None

def _probe(mp4, hwaccel):
    return subprocess.run(['ffmpeg', '-v', 'error', '-hwaccel', hwaccel, '-i', mp4, '-frames:v', '1', '-f', 'null', '-'],
                          capture_output=True).returncode == 0

def decode_frames(mp4, w=640, h=360, hwaccel='auto'):
    if hwaccel == 'auto': hwaccel = hwaccel_default()
    elif hwaccel == 'none': hwaccel = None
    if hwaccel and not _probe(mp4, hwaccel): hwaccel = None   # e.g. videotoolbox unsupported for this file
    cmd = ['ffmpeg', '-v', 'error']
    if hwaccel: cmd += ['-hwaccel', hwaccel]
    cmd += ['-i', mp4, '-vf', f'scale={w}:{h}', '-pix_fmt', 'gray', '-f', 'rawvideo', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w*h*8)
    n = w * h
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n: break
        yield np.frombuffer(buf, np.uint8).reshape(h, w)
    p.wait()

def analyze(mp4, out_csv, w=640, h=360, hwaccel='auto'):
    prev = None; prev_pts = None
    fields = ['frame', 'n_tracks', 'dx', 'dy', 'da', 'ds', 'inlier_ratio', 'residual', 'lap_var', 'mean_luma', 'mdx', 'mdy']
    with open(out_csv, 'w', newline='') as fh:
        wr = csv.writer(fh); wr.writerow(fields)
        for i, g in enumerate(decode_frames(mp4, w, h, hwaccel)):
            lap = cv2.Laplacian(g, cv2.CV_32F).var(); luma = float(g.mean())
            row = [i, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, lap, luma, 0.0, 0.0]
            if prev is not None:
                pts = cv2.goodFeaturesToTrack(prev, maxCorners=600, qualityLevel=0.01, minDistance=8, blockSize=7)
                if pts is not None and len(pts) >= 20:
                    nxt, st, err = cv2.calcOpticalFlowPyrLK(prev, g, pts, None, winSize=(21, 21), maxLevel=4,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
                    ok = st.ravel() == 1
                    p0, p1 = pts[ok].reshape(-1, 2), nxt[ok].reshape(-1, 2)
                    if len(p0) >= 20:
                        mdxy = np.median(p1 - p0, axis=0)
                        M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=500, confidence=0.99)
                        if M is not None and inl is not None and inl.sum() >= 10:
                            inl = inl.ravel().astype(bool)
                            a, b = M[0, 0], M[0, 1]
                            s = float(np.hypot(a, b)); da = float(np.arctan2(M[1, 0], M[0, 0]))
                            # translation measured at image centre
                            c = np.array([w / 2, h / 2])
                            cc = M[:, :2] @ c + M[:, 2]
                            dxy = cc - c
                            proj = (M[:, :2] @ p0[inl].T).T + M[:, 2]
                            res = float(np.median(np.linalg.norm(proj - p1[inl], axis=1)))
                            row = [i, int(len(p0)), float(dxy[0]), float(dxy[1]), da, float(np.log(s)), float(inl.mean()), res, lap, luma, float(mdxy[0]), float(mdxy[1])]
            wr.writerow(row); prev = g
            if i % 500 == 0: print(i, file=sys.stderr, flush=True)

if __name__ == '__main__':
    analyze(sys.argv[1], sys.argv[2])

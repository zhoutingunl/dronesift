"""Fuse telemetry (djmd) and optical-flow motion into labelled stable/continuous segments."""
import csv, json, math
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d, maximum_filter1d, binary_dilation

FPS = 59.94

def _col(rows, k, default=0.0):
    return np.array([float(r[k]) if r.get(k) not in (None, '') else default for r in rows])

def _unwrap_deg(a):
    return np.degrees(np.unwrap(np.radians(a)))

def _wrap_deg(a):
    return (a + 180) % 360 - 180

def _deriv(x, fps):
    return np.gradient(x) * fps

TEL_KEYS = ('vx', 'vy', 'vz', 'ac_yaw', 'ac_pitch', 'ac_roll', 'g_yaw', 'g_pitch', 'g_roll',
            'rel_alt', 'abs_alt', 'lat', 'lon', 'iso', 'ev', 'exposure')

def aligned_tel(tel, n, fps):
    """Telemetry columns resampled onto decoded-frame timestamps t = i / fps.

    Telemetry rows carry their packet time in `pts`; zipping the two streams by index
    silently misaligns them whenever frames are dropped or the track starts at an offset.
    Returns (cols, covered): cols maps every TEL_KEYS column to a length-n array,
    covered marks frames that actually have telemetry.
    """
    cols = {k: _col(tel, k) for k in TEL_KEYS}
    t = np.arange(n) / fps
    pts = _col(tel, 'pts', float('nan'))
    if np.isfinite(pts).all() and len(pts) >= 2:
        ok = np.concatenate(([True], np.diff(pts) > 0))       # drop duplicate / stale timestamps
        if ok.sum() >= 2:
            pk = pts[ok]
            return {k: np.interp(t, pk, v[ok]) for k, v in cols.items()}, (t >= pk[0]) & (t <= pk[-1] + 0.5 / fps)
    m = min(len(tel), n)
    cols = {k: (np.pad(v[:m], (0, n - m), mode='edge') if m else np.zeros(n)) for k, v in cols.items()}
    return cols, np.arange(n) < m   # no usable pts: old index alignment, last value extended

def features(tel, flow, fps=FPS, tel_sigma_s=0.5, vis_sigma_s=1.0, hf_sigma_s=0.15, rev_amp=25.0):
    """Per-frame measurements, telemetry coverage, and the source FPS."""
    n = len(flow)
    s_t = tel_sigma_s * fps; s_v = vis_sigma_s * fps; s_h = hf_sigma_s * fps
    f = {}
    A, f['tel_ok'] = aligned_tel(tel, n, fps)
    # ---- telemetry: velocity (NED, m/s) & acceleration
    vx, vy, vz = (gaussian_filter1d(A[k], s_t) for k in ('vx', 'vy', 'vz'))
    f['vx'], f['vy'], f['vz'] = vx, vy, vz
    f['speed_h'] = np.hypot(vx, vy); f['speed3'] = np.sqrt(vx**2 + vy**2 + vz**2)
    ax, ay, az = (_deriv(v, fps) for v in (vx, vy, vz))
    f['acc_h'] = np.hypot(ax, ay); f['acc_z'] = np.abs(az)
    f['acc_tan'] = np.abs(_deriv(f['speed_h'], fps))      # along-track (stick) acceleration; excludes centripetal
    f['track'] = np.degrees(np.arctan2(vy, vx))            # heading of horizontal velocity (deg, N=0, E=90)
    # ---- gimbal attitude (camera pointing), deg
    gyaw = gaussian_filter1d(_unwrap_deg(A['g_yaw']), s_t)
    gpitch = gaussian_filter1d(A['g_pitch'], s_t)
    f['g_yaw'] = gyaw; f['g_pitch'] = gpitch
    # Filter angular velocity rather than differentiating reflected angle ends:
    # a constant pan must remain constant right up to both source boundaries.
    angles = {axis: median_filter(_unwrap_deg(A['g_' + axis]), size=3, mode='nearest')
              for axis in ('yaw', 'pitch', 'roll')}
    f['yaw_rate'] = gaussian_filter1d(_deriv(angles['yaw'], fps), s_t, mode='nearest')
    f['pitch_rate'] = gaussian_filter1d(_deriv(angles['pitch'], fps), s_t, mode='nearest')
    f['yaw_acc'] = np.abs(_deriv(f['yaw_rate'], fps)); f['pitch_acc'] = np.abs(_deriv(f['pitch_rate'], fps))
    f['ac_yaw_rate'] = _deriv(gaussian_filter1d(_unwrap_deg(A['ac_yaw']), s_t), fps)
    # A stick correction need not reverse direction. Compare a responsive angular
    # velocity with its local trajectory: a uniform ramp has virtually no residual.
    # Smooth first, then differentiate, to avoid amplifying quantised telemetry.
    for axis in ('yaw', 'pitch', 'roll'):
        rate = gaussian_filter1d(_deriv(angles[axis], fps), 0.12 * fps, mode='nearest')
        f[axis + '_rate_fast'] = rate
        f[axis + '_correction'] = np.abs(rate - gaussian_filter1d(rate, 0.8 * fps))
    f['fps'] = fps
    # velocity in camera frame: forward (along gimbal yaw) and lateral (right positive)
    cam_yaw = gaussian_filter1d(_unwrap_deg(A['ac_yaw']), s_t)   # gimbal yaw has a bias vs compass; pan range is small
    rel = np.radians(_wrap_deg(f['track'] - _wrap_deg(cam_yaw)))
    f['v_fwd'] = f['speed_h'] * np.cos(rel); f['v_lat'] = f['speed_h'] * np.sin(rel)
    f['rel_alt'] = A['rel_alt']
    # exposure changes
    f['iso'] = A['iso']; f['ev'] = A['ev']
    expo = A['exposure']; iso = np.maximum(f['iso'], 1.0)
    lv = np.log2(np.where(expo > 0, expo, 1.0) * iso)                      # exposure value equivalent (stops)
    h = max(1, int(0.25 * fps)); lv_pad = np.pad(lv, h, mode='edge')
    f['expo_jump'] = np.abs(lv_pad[2 * h:] - lv_pad[:-2 * h])[:len(lv)]   # change over +-0.25 s
    # ---- vision: raw per-frame motion vs low-pass motion -> jitter.  All in px/s at 640px width (fps-independent).
    has_med = 'mdx' in flow[0]
    dx, dy, da, ds = (_col(flow, k) * fps for k in ('dx', 'dy', 'da', 'ds'))
    jdx, jdy = ((_col(flow, 'mdx') * fps, _col(flow, 'mdy') * fps) if has_med else (dx, dy))   # median flow: robust to parallax
    f['horizontal_motion_correction'] = np.abs(
        gaussian_filter1d(jdx, 0.12 * fps) - gaussian_filter1d(jdx, 0.8 * fps))
    f['motion_correction'] = np.hypot(
        gaussian_filter1d(jdx, 0.12 * fps) - gaussian_filter1d(jdx, 0.8 * fps),
        gaussian_filter1d(jdy, 0.12 * fps) - gaussian_filter1d(jdy, 0.8 * fps))
    ok = _col(flow, 'n_tracks') >= 20
    # Frame zero has no predecessor, hence no optical flow. It is not a bad
    # image and must not force a whole GOP off the beginning of a smooth video.
    if n > 1 and ok[1]: ok[0] = True
    w = 640
    for name, raw in (('dx', dx), ('dy', dy), ('da', da), ('ds', ds)):
        f[name + '_raw'] = raw; f[name + '_sm'] = gaussian_filter1d(raw, s_v)
    f['rotation_correction'] = np.abs(gaussian_filter1d(da, 0.12 * fps)
                                     - gaussian_filter1d(da, 0.8 * fps)) * w / 2
    jit = np.sqrt((jdx - gaussian_filter1d(jdx, s_h))**2 + (jdy - gaussian_filter1d(jdy, s_h))**2
                  + ((da - gaussian_filter1d(da, s_h)) * w / 2)**2 * (0 if has_med else 1))
    jit[~ok] = np.nan
    win = max(3, int(0.5 * fps) | 1)
    valid = np.isfinite(jit)
    jit2 = np.interp(np.arange(n), np.flatnonzero(valid), jit[valid]**2) if valid.any() else np.zeros(n)
    f['jitter'] = jit; f['jitter_rms'] = np.sqrt(median_filter(jit2, size=win))
    # short-window rms: a 0.1-0.3 s shake burst (manual dial snap) is diluted below jitter_rms
    # by the 0.5 s median filter, but clearly visible; single-frame flow glitches are not (yet)
    f['jitter_burst'] = np.sqrt(np.maximum(0, uniform_filter1d(jit2, max(3, int(0.15 * fps) | 1))))
    msm = gaussian_filter1d(np.hypot(jdx, jdy), s_v)
    f['motion_px'] = msm + np.abs(f['da_sm']) * w / 2 + np.abs(f['ds_sm']) * w / 2     # px/s
    f['motion_acc'] = np.abs(_deriv(f['motion_px'], fps))                               # px/s^2
    # slow component reversal: a translation/zoom component that changes SIGN inside the shot
    # (pan right... then left, push decaying into a drift back) reads as contradictory motion
    # even at low frequency, where hf_flip cannot see it. Rotation is exempt: orbit direction
    # changes are choreographed moves. Frames near such a crossing are vetoed.
    rev = np.zeros(n, bool)
    rwin = int(2.5 * fps) | 1
    for c in (f['dx_sm'], f['dy_sm'], f['ds_sm'] * w / 2):
        pos_env = maximum_filter1d(np.maximum(c, 0), rwin)
        neg_env = maximum_filter1d(np.maximum(-c, 0), rwin)
        rev |= (pos_env > rev_amp) & (neg_env > rev_amp)
    f['motion_reversal'] = rev
    # perceptual shake = image motion that REVERSES (oscillates around its path: deviates and comes
    # back). A single fast excursion (whip tilt/pan, MasterShots moves) never reverses and reads as
    # an intentional move, however violent. Metric: fraction of frames whose hf flow flips sign in 0.1 s.
    hx, hy = jdx - gaussian_filter1d(jdx, s_h), jdy - gaussian_filter1d(jdy, s_h)
    lag = max(1, int(0.1 * fps))
    en = np.hypot(hx, hy)
    live = (en > 3.0).astype(float)
    live_lag = np.concatenate((np.zeros(lag), live[:-lag]))
    flip = ((hx[lag:] * hx[:-lag] + hy[lag:] * hy[:-lag]) < 0).astype(float)
    flip = np.concatenate((np.zeros(lag), flip)) * np.minimum(live, live_lag)
    f['hf_flip'] = uniform_filter1d(flip, max(3, int(0.3 * fps) | 1))
    # directional consistency of the image motion over a 2 s window: |sum v| / sum |v|  (1 = steady move, ~0 = wobble)
    vx, vy = gaussian_filter1d(jdx, 0.3 * fps), gaussian_filter1d(jdy, 0.3 * fps); mag = np.hypot(vx, vy)
    win2 = int(2.0 * fps) | 1
    f['vmag'] = uniform_filter1d(mag, win2)
    f['consist'] = np.hypot(uniform_filter1d(vx, win2), uniform_filter1d(vy, win2)) / np.maximum(f['vmag'], 1e-6)
    # implied orbit radius (m) from lateral speed and camera yaw rate
    f['radius'] = np.abs(f['v_lat']) / np.maximum(np.radians(np.abs(f['yaw_rate'])), 1e-3)
    lap = _col(flow, 'lap_var'); lap_med = median_filter(lap, size=int(2 * fps) | 1)
    f['blur_drop'] = lap / np.maximum(lap_med, 1e-6)
    f['residual'] = _col(flow, 'residual'); f['n_tracks'] = _col(flow, 'n_tracks')
    f['flow_ok'] = ok
    return f, n

DEFAULT_THR = dict(
    jitter_px=30.0,       # rms image motion above ~1 Hz, px/s at 640px width (≈0.5 px/frame @60fps)
    hf_flip=0.25,         # direction-flip gate for image-amplitude tests; 0 explicitly disables it
    motion_rev_amp=0.0,   # optional legacy component-sign veto; smooth reversals are kept by default
    jitter_spike_px=45.0, # 0.15 s rms shake burst (manual dial snap, ≈0.75 px/frame); lasts >= spike_min_frames
    spike_min_frames=3, spike_pad_s=0.35,  # minimum burst length, then a settling margin on each side
    acc_h=1.2, acc_z=1.0, # m/s^2, along-track / vertical acceleration (stick inputs); centripetal accel is not counted
    accel_veto_rel=0.5,   # ...and only vetoes when image motion accel is above this fraction of its threshold
    yaw_acc=8.0, pitch_acc=8.0,  # deg/s^2, gimbal angular acceleration (for orbits, relaxed to acc_h / radius)
    yaw_correction_dps=6.0,  # fast yaw rate minus its 0.8 s trajectory, catches one-way stick steps
    yaw_nudge_dps=2.0, yaw_nudge_rel=0.5, yaw_nudge_motion_px=10.0,  # small isolated yaw step, with horizontal image evidence
    yaw_nudge_episode_s=5.0, yaw_nudge_gap_s=0.25, yaw_nudge_quiet_dps=0.5,  # grow confirmed short gestures through their settling tails
    pitch_return_dps=1.5, pitch_return_rel=0.35,  # brief tilt-and-return, relative to the local gesture
    roll_correction_dps=1.5,  # roll telemetry must agree with optical rotation, not just parallax
    correction_motion_px=5.0,  # corroborating median image-motion residual, px/s @640
    brake_yaw_acc=12.0, brake_motion_acc=40.0,  # sudden camera slowdown corroborated by image motion
    disturbance_min_s=0.10, disturbance_pad_s=0.35,  # sustained evidence + settling margin; never merged over
    motion_acc=60.0,      # px/s^2, change of low-pass image motion speed (start/stop of moves)
    motion_acc_rel=0.5,   # ... or 50 %/s of current image speed, whichever is larger
    expo_jump=0.3,        # stops of exposure change within 0.5 s (auto-exposure steps are ~0.15)
    consist=0.6, consist_floor=8.0,  # low direction consistency must agree with short-timescale oscillation
    blur_drop=0.45,       # sharpness relative to local median
    min_len_s=3.0, merge_gap_s=0.5, trim_s=0.3,
    label_min_s=3.0, label_filter_s=2.0, climb_rate=1.0,  # sub-moves shorter than label_min_s are not named; labels mode-filtered over label_filter_s
    split_labels=0,       # 1 = also cut segments where the move type changes (old behaviour)
    max_shot_s=0.0,       # shot-ification: split segments into edit-ready shots <= max_shot_s (0 = off)
    target_shot_s=6.0, min_shot_s=4.0,   # preferred shot length; remainder shorter than min is merged back
    hover_speed=0.4, pan_rate=1.5,
    orbit_rate=0.8, orbit_r_min=1.0, orbit_r_max=1000.0,  # orbit: yaw rate >= orbit_rate deg/s and implied radius v_lat/yaw_rate in range
)

def base_labels(f, thr, fps=FPS):
    """Per-frame motion labels; mode-filtered over label_filter_s to suppress flicker."""
    n = len(f['speed3']); labels = []
    for i in range(n):
        sp, sh, vf, vl, vz = f['speed3'][i], f['speed_h'][i], f['v_fwd'][i], f['v_lat'][i], f['vz'][i]
        yr, pr = f['yaw_rate'][i], f['pitch_rate'][i]
        if sp < thr['hover_speed']:
            if abs(yr) >= thr['pan_rate']: lab = 'pan'
            elif abs(pr) >= thr['pan_rate']: lab = 'tilt'
            else: lab = 'static'
        elif abs(vl) > 0.5 * max(sh, 1e-6) and vl * yr < 0 and abs(yr) >= thr['orbit_rate'] \
                and thr['orbit_r_min'] < abs(vl) / math.radians(abs(yr)) < thr['orbit_r_max']:
            # lateral motion with the camera turning the opposite way at a rate consistent with a sane radius
            lab = 'orbit'
        elif abs(vz) > 0.7 * sp: lab = 'vertical'
        elif abs(vf) >= 0.7 * sh: lab = 'push' if vf > 0 else 'pull'
        elif abs(vl) >= 0.7 * sh: lab = 'truck'
        else: lab = 'diagonal'
        if lab not in ('static', 'pan', 'tilt', 'vertical') and abs(vz) > thr['climb_rate']:
            lab += '+climb' if vz < 0 else '+descend'
        labels.append(lab)
    labels = np.array(labels); win = int(thr['label_filter_s'] * fps) | 1; half = win // 2; out = labels.copy()
    for i in range(n):
        u, c = np.unique(labels[max(0, i - half): i + half + 1], return_counts=True); out[i] = u[np.argmax(c)]
    return out

def spike_mask(f, thr, fps=FPS):
    """Frames hit by a visible shake burst: 0.15 s rms above jitter_spike_px for >= spike_min_frames
    AND reversing (hf_flip), padded by spike_pad_s so the hole outlasts merge_gap_s and splits the segment."""
    from scipy.ndimage import binary_dilation, binary_erosion
    shake = (f['hf_flip'] > thr['hf_flip']) if thr['hf_flip'] > 0 else np.ones(len(f['hf_flip']), bool)
    hot = (f['jitter_burst'] > thr['jitter_spike_px']) & shake
    hot = binary_erosion(hot, structure=np.ones(max(1, int(thr['spike_min_frames']))), border_value=1)
    pad = int(thr['spike_pad_s'] * fps)
    return binary_dilation(hot, structure=np.ones(2 * pad + 1)) if pad else hot

def _sustained(mask, seconds, fps):
    """Keep entire qualifying events, not just an eroded centre (which loses boundaries)."""
    out = np.zeros(len(mask), bool)
    minimum = max(1, int(math.ceil(seconds * fps)))
    for a, b, value in _runs(mask):
        if value and b - a + 1 >= minimum: out[a:b + 1] = True
    return out

def _yaw_nudge_episodes(seed, rate, peak, thr, fps):
    """Cover a confirmed short correction from onset to rest, not just its peaks.

    Join brief pauses inside a left-right gesture. Grow only compact gestures:
    an active pan lasting longer than the limit remains a continuous trajectory,
    so a local correction must not erase that entire move.
    """
    if not seed.any(): return seed
    active = np.abs(rate) > np.maximum(thr['yaw_nudge_quiet_dps'], 0.1 * peak)
    episodes = []
    for a, b, value in _runs(active):
        if not value: continue
        if episodes and (a - episodes[-1][1] - 1) / fps <= thr['yaw_nudge_gap_s']:
            episodes[-1][1] = b
        else: episodes.append([a, b])
    out = seed.copy()
    for a, b in episodes:
        if (b - a + 1) / fps <= thr['yaw_nudge_episode_s'] and seed[a:b + 1].any():
            out[a:b + 1] = True
    return out

def disturbance_masks(f, thr, fps=FPS):
    """Independent, explainable camera disturbances; all times use the source FPS."""
    n = len(f['speed3']); zero = np.zeros(n)
    correction = (f.get('yaw_correction', zero) > thr['yaw_correction_dps']) if thr['yaw_correction_dps'] > 0 else zero.astype(bool)
    # A subtle lateral reframe can be clearly visible in a steady push despite
    # falling below the severe yaw threshold. Require BOTH a departure from the
    # local angular trajectory and horizontal image motion. A fast vertical tilt
    # can perturb Euler yaw, but its large vertical flow is not lateral evidence.
    win = max(1, int(4.0 * fps) | 1)
    yaw_peak = maximum_filter1d(np.abs(f.get('yaw_rate_fast', zero)), win)
    yaw_residual = f.get('yaw_correction', zero)
    yaw_nudge = (yaw_residual > thr['yaw_nudge_dps']) \
        & (yaw_residual > thr['yaw_nudge_rel'] * yaw_peak) \
        & (f.get('horizontal_motion_correction', zero) > thr['yaw_nudge_motion_px'])
    # A fast orbit can be smooth. Reject a brake only when both the camera's
    # angular deceleration and the image's slowdown substantiate the event.
    braking = (f['yaw_rate'] * _deriv(f['yaw_rate'], fps) < 0) \
        & (f['yaw_acc'] > thr['brake_yaw_acc']) & (f['motion_acc'] > thr['brake_motion_acc']) \
        & (_deriv(f['motion_px'], fps) < 0)
    pitch = f.get('pitch_rate_fast', zero)
    pos = maximum_filter1d(np.maximum(pitch, 0), win)
    neg = maximum_filter1d(np.maximum(-pitch, 0), win)
    residual = f.get('pitch_correction', zero)
    pitch_return = (residual > thr['pitch_return_dps']) & (pos > thr['pitch_return_dps']) \
        & (neg > thr['pitch_return_dps']) & (residual > thr['pitch_return_rel'] * np.maximum(pos, neg)) \
        & (f.get('motion_correction', zero) > thr['correction_motion_px'])
    roll = (f.get('roll_correction', zero) > thr['roll_correction_dps']) \
        & (f.get('rotation_correction', zero) > thr['correction_motion_px'])
    masks = {'yaw_correction': correction, 'yaw_nudge': yaw_nudge, 'pitch_return': pitch_return,
             'roll_correction': roll, 'camera_brake': braking}
    pad = int(math.ceil(thr['disturbance_pad_s'] * fps))
    for name, mask in masks.items():
        mask = _sustained(mask, thr['disturbance_min_s'], fps)
        if name == 'yaw_nudge':
            mask = _yaw_nudge_episodes(mask, f.get('yaw_rate_fast', zero), yaw_peak, thr, fps)
        masks[name] = binary_dilation(mask, structure=np.ones(2 * pad + 1)) if pad else mask
    masks['shake_burst'] = spike_mask(f, thr, fps)
    return masks

def keep_mask(f, thr, labels=None, fps=None):
    fps = f.get('fps', FPS) if fps is None else fps
    n = len(f['speed3'])
    orbit = np.zeros(n, bool) if labels is None else np.char.startswith(labels.astype(str), 'orbit')
    # in an orbit yaw rate = v / r, so a legal along-track acceleration implies yaw accel of acc_h / r
    yaw_thr = np.where(orbit, np.maximum(thr['yaw_acc'], np.degrees(thr['acc_h'] / np.maximum(f['radius'], 1.0))), thr['yaw_acc'])
    m_thr = np.maximum(thr['motion_acc'], thr['motion_acc_rel'] * f['motion_px'])
    # This gate applies to the older image-amplitude tests. Angular trajectory
    # corrections have their own evidence and need not reverse image direction.
    shake = (f['hf_flip'] > thr['hf_flip']) if thr['hf_flip'] > 0 else np.ones(n, bool)
    # A smooth U-turn also has low direction consistency. Low consistency alone
    # cannot establish shake; require short-timescale image irregularity.
    wobble = (f['consist'] < thr['consist']) & (f['vmag'] > thr['consist_floor']) & shake
    image_ok = ((f['jitter_rms'] < thr['jitter_px']) & (f['motion_acc'] < m_thr) | ~shake) \
        & (f['blur_drop'] > thr['blur_drop']) & f['flow_ok'] & ~wobble
    gimbal_ok = ((f['yaw_acc'] < yaw_thr) & (f['pitch_acc'] < thr['pitch_acc'])) | ~shake
    # ground-frame acceleration is ambiguous when tracking a moving subject (circling a car), so it only
    # vetoes frames whose image motion both changes noticeably AND reverses
    accel_veto = ((f['acc_tan'] > thr['acc_h']) | (f['acc_z'] > thr['acc_z'])) \
        & (f['motion_acc'] > thr['accel_veto_rel'] * m_thr) & shake
    keep = image_ok & gimbal_ok & ~accel_veto
    keep &= (f['expo_jump'] < thr['expo_jump']) & (np.abs(np.gradient(f['ev'])) == 0)   # exposure jumps / EV dial changes
    keep &= f['tel_ok']                                                                 # frames without telemetry cannot be judged
    disturbances = disturbance_masks(f, thr, fps)
    # Moderate oscillation below the burst threshold is still an observed event,
    # not a weak measurement gap. Preserve it even with a large merge-gap setting.
    oscillation = shake & ((f['jitter_rms'] >= thr['jitter_px']) | (f['motion_acc'] >= m_thr)
                           | ~gimbal_ok | wobble)
    oscillation = _sustained(oscillation, thr['disturbance_min_s'], fps)
    pad = int(math.ceil(thr['disturbance_pad_s'] * fps))
    disturbances['image_oscillation'] = binary_dilation(oscillation, structure=np.ones(2 * pad + 1)) if pad else oscillation
    hard = ~f['tel_ok']
    for mask in disturbances.values(): hard = hard | mask
    # Store exclusions separately: segment assembly may bridge weak isolated
    # measurements, but it must never resurrect an observed camera disturbance.
    f['hard_reject'] = hard
    f['disturbances'] = disturbances
    keep &= ~hard
    if thr.get('motion_rev_amp', 0) > 0:                                                # slow contradictory component changes
        keep &= ~(f['motion_reversal'] if 'motion_reversal' in f
                  else np.zeros(len(f['speed3']), bool))
    return keep

def classify_frames(f, thr, fps=FPS):
    labels = base_labels(f, thr, fps)
    return keep_mask(f, thr, labels, fps), labels

def describe(f, sl, base, thr):
    """Final label for a run from its mean motion. base may carry a '+climb' / '+descend' suffix."""
    core, _, vmod = base.partition('+')
    yr, pr, vz = f['yaw_rate'][sl].mean(), f['pitch_rate'][sl].mean(), f['vz'][sl].mean()
    vf, vl = f['v_fwd'][sl].mean(), f['v_lat'][sl].mean()
    if core == 'pan': lab = 'pan_L' if yr < 0 else 'pan_R'
    elif core == 'tilt': lab = 'tilt_up' if pr > 0 else 'tilt_down'
    elif core == 'orbit':
        lab = 'orbit_CW' if vl > 0 else 'orbit_CCW'
        if abs(vf) > abs(vl): lab = 'spiral_in' if vf > 0 else 'spiral_out'
    elif core == 'vertical': lab = 'ascend' if vz < 0 else 'descend'
    elif core == 'push': lab = 'push_in'
    elif core == 'pull': lab = 'dronie' if vmod == 'climb' else 'pull_back'
    elif core == 'truck': lab = 'truck_R' if vl > 0 else 'truck_L'
    else: lab = core
    if core not in ('pan', 'orbit', 'static') and abs(yr) >= thr['pan_rate']: lab += '+pan'
    if vmod and lab != 'dronie': lab += '+' + vmod
    return lab

def _runs(mask_or_labels):
    """Yield (start, end_inclusive, value) runs of equal consecutive values."""
    a = np.asarray(mask_or_labels); n = len(a); i = 0
    while i < n:
        j = i
        while j + 1 < n and a[j + 1] == a[i]: j += 1
        yield i, j, a[i]; i = j + 1

def _label_sequence(f, labels, a, b, thr, fps):
    """Describe a kept run as a sequence of moves; sub-moves shorter than label_min_s are absorbed."""
    seq = []
    for i, j, lab in _runs(labels[a:b + 1]):
        if (j - i + 1) / fps < thr['label_min_s'] and seq: continue
        d = describe(f, slice(a + i, a + j + 1), str(lab), thr)
        if seq and (d == seq[-1][0] or (j - i + 1) / fps < thr['label_min_s']): continue
        seq.append((d, a + i, a + j))
    # a leading short run is absorbed by what follows
    if len(seq) > 1 and (seq[0][2] - seq[0][1] + 1) / fps < thr['label_min_s']: seq = seq[1:]
    return [d for d, _, _ in seq]

def segments(f, keep, labels, thr, fps=FPS):
    """Continuous kept runs -> merged across short gaps, trimmed, min-length filtered, labelled by move sequence."""
    runs = [[i, j] for i, j, v in _runs(keep) if v]
    if thr.get('split_labels'):
        runs = [[i, j] for i, j, v in _runs(np.where(keep, labels, '')) if v != '']
    merged = []
    hard = f.get('hard_reject', np.zeros(len(keep), bool))
    for a, b in runs:
        gap_start = merged[-1][1] + 1 if merged else 0
        if merged and (a - gap_start) / fps <= thr['merge_gap_s'] and not hard[gap_start:a].any() and (not thr.get('split_labels') or labels[a] == labels[merged[-1][0]]):
            merged[-1][1] = b
        else: merged.append([a, b])
    trim = int(thr['trim_s'] * fps); out = []
    for a, b in merged:
        # Safe source endpoints need no settling trim. Interior boundaries still
        # get handles so a retained clip starts after, and ends before, a correction.
        a2 = a + trim if a > 0 else a
        b2 = b - trim if b < len(keep) - 1 else b
        if (b2 - a2 + 1) / fps < thr['min_len_s']: continue
        sl = slice(a2, b2 + 1)
        seq = _label_sequence(f, labels, a2, b2, thr, fps)
        u, c = np.unique(labels[sl], return_counts=True); base = str(u[np.argmax(c)]).partition('+')[0]
        out.append(dict(start_frame=int(a2), end_frame=int(b2), start=a2 / fps, end=(b2 + 1) / fps,
                        duration=(b2 - a2 + 1) / fps, label='>'.join(seq), moves=seq, base=base,
                        jitter_px=float(np.nanmean(f['jitter_rms'][sl])), speed_mps=float(f['speed3'][sl].mean()),
                        yaw_rate_dps=float(f['yaw_rate'][sl].mean()), pitch_rate_dps=float(f['pitch_rate'][sl].mean()),
                        climb_mps=float(-f['vz'][sl].mean()), rel_alt_m=float(f['rel_alt'][sl].mean()),
                        motion_px=float(f['motion_px'][sl].mean()),
                        radius_m=(float(np.abs(f['v_lat'][sl]).mean() / max(math.radians(abs(f['yaw_rate'][sl].mean())), 1e-6))
                                  if base == 'orbit' else None)))
    return out

def run(tel_csv, flow_csv, out_json, thr=None, fps=FPS):
    thr = {**DEFAULT_THR, **(thr or {})}
    tel = list(csv.DictReader(open(tel_csv))); flow = list(csv.DictReader(open(flow_csv)))
    f, n = features(tel, flow, fps, rev_amp=thr['motion_rev_amp'])
    keep, labels = classify_frames(f, thr, fps)
    segs = segments(f, keep, labels, thr, fps)
    with open(out_json, 'w') as fh: json.dump(dict(fps=fps, frames=n, thresholds=thr, segments=segs), fh, indent=1)
    return f, keep, labels, segs

def hms(t):
    return f"{int(t // 60):02d}:{t % 60:06.3f}"

def _shot(f, labels, a, b, fps, seg_i):
    sl = slice(a, b + 1)
    u, c = np.unique(labels[sl], return_counts=True)
    lab = str(u[np.argmax(c)])
    return dict(start_frame=int(a), end_frame=int(b), start=a / fps, end=(b + 1) / fps,
                duration=(b - a + 1) / fps, label=lab, base=lab.partition('+')[0], segment=seg_i,
                jitter_px=float(np.nanmean(f['jitter_burst'][sl])), speed_mps=float(f['speed3'][sl].mean()),
                yaw_rate_dps=float(f['yaw_rate'][sl].mean()))

def shots(segs, f, labels, thr, fps=FPS):
    """Split segments into edit-ready shots <= max_shot_s (edited drone cuts average 4-10 s per shot).

    Cut points prefer sub-move label boundaries inside the window [min_shot_s, max_shot_s] ahead of the
    cut, choosing the one closest to target_shot_s; without a boundary, the calmest moment (low
    jitter_burst + motion_acc) near the target is used. A remainder shorter than min_shot_s is merged
    back into the previous shot. max_shot_s = 0 disables and returns [].
    """
    tmax, ttar, tmin = thr.get('max_shot_s', 0.0), thr.get('target_shot_s', 6.0), thr.get('min_shot_s', 4.0)
    if not tmax:
        return []
    lab = labels.astype(str)
    bounds = np.nonzero(lab[1:] != lab[:-1])[0] + 1
    score = np.asarray(f['jitter_burst']) + np.asarray(f['motion_acc']) / 100.0
    hot = np.asarray(f['jitter_burst']) > thr['jitter_px']               # shake bursts: keep cut points away
    pad = int(0.5 * fps)
    if pad and hot.any():
        hot = np.convolve(hot.astype(float), np.ones(2 * pad + 1), mode='same') > 0
    out = []
    for si, s in enumerate(segs):
        a, b = s['start_frame'], s['end_frame']
        if (b - a + 1) / fps <= tmax:
            out.append(_shot(f, labels, a, b, fps, si)); continue
        cur = a
        while (b - cur + 1) / fps > tmax:
            lo, hi = cur + int(tmin * fps), min(cur + int(tmax * fps), b - int(tmin * fps))
            if hi < lo: hi = cur + int(tmax * fps)            # tiny remainder: allow full window, merge below
            tgt = cur + int(ttar * fps)
            cand = bounds[(bounds >= lo) & (bounds <= hi)]
            pool = cand[(b - cand) / fps >= tmin] if len(cand) else cand
            if len(pool):
                c = int(pool[np.argmin(np.abs(pool - tgt))])   # boundary closest to the target length
            else:
                l2, h2 = max(lo, tgt - int(1.5 * fps)), min(hi, tgt + int(1.5 * fps))
                quiet = np.nonzero(~hot[l2:h2 + 1])[0] if h2 > l2 else np.arange(0)
                if h2 > l2 and quiet.size:
                    wq = np.nan_to_num(score[l2:h2 + 1][quiet], nan=np.inf)
                    calm = quiet[wq <= wq.min() + max(1e-9, 0.05 * wq.min())]
                    c = l2 + int(calm[np.argmin(np.abs(l2 + calm - tgt))])   # calmest, then closest to target
                else:
                    c = cur + int(tmax * fps)
            out.append(_shot(f, labels, cur, c - 1, fps, si))  # cut c is the next shot's first frame
            cur = c
        if (b - cur + 1) / fps < tmin and out:                 # merge a stub back into the previous shot
            p = out[-1]; out[-1] = _shot(f, labels, p['start_frame'], b, fps, si)
        else:
            out.append(_shot(f, labels, cur, b, fps, si))
    for i, s in enumerate(out): s['shot'] = i
    return out

if __name__ == '__main__':
    import sys
    f, keep, labels, segs = run(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f"kept frames: {keep.mean() * 100:.1f}%")
    tot = 0
    for s in segs:
        tot += s['duration']
        print(f"{hms(s['start'])} - {hms(s['end'])}  {s['duration']:5.1f}s  {s['label']:14s} jit={s['jitter_px']:.1f}px/s v={s['speed_mps']:.1f}m/s yaw={s['yaw_rate_dps']:+.1f}°/s climb={s['climb_mps']:+.1f}")
    print(f"total kept {tot:.1f}s in {len(segs)} segments")

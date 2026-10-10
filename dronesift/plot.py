import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

def timeline(f, keep, segs, thr, path, fps, title=''):
    n = len(keep); t = np.arange(n) / fps
    fig, axes = plt.subplots(5, 1, figsize=(16, 10), sharex=True)
    def shade(ax):
        for s in segs: ax.axvspan(s['start'], s['end'], color='tab:green', alpha=0.18)
        ax.fill_between(t, 0, 1, where=~keep, transform=ax.get_xaxis_transform(), color='tab:red', alpha=0.08, step='mid')
    ax = axes[0]; ax.plot(t, f['jitter_rms'], lw=0.8, label='image jitter rms (px/s @640)'); ax.axhline(thr['jitter_px'], c='r', ls='--', lw=0.7); ax.set_ylim(0, 150); ax.legend(loc='upper right'); shade(ax)
    ax = axes[1]; ax.plot(t, f['motion_px'], lw=0.8, label='image motion (px/s)'); ax.plot(t, f['motion_acc'], lw=0.8, label='motion accel (px/s²)'); ax.axhline(thr['motion_acc'], c='r', ls='--', lw=0.7); ax.set_ylim(0, 600); ax.legend(loc='upper right'); shade(ax)
    ax = axes[2]; ax.plot(t, f['speed_h'], lw=0.8, label='ground speed m/s'); ax.plot(t, -f['vz'], lw=0.8, label='climb m/s'); ax.plot(t, f['acc_tan'], lw=0.8, label='along-track accel m/s²'); ax.axhline(thr['acc_h'], c='r', ls='--', lw=0.7); ax.legend(loc='upper right'); shade(ax)
    ax = axes[3]; ax.plot(t, f['yaw_rate'], lw=0.8, label='gimbal yaw rate °/s'); ax.plot(t, f['pitch_rate'], lw=0.8, label='gimbal pitch rate °/s'); ax.plot(t, f['yaw_acc'], lw=0.6, alpha=0.6, label='yaw accel'); ax.axhline(thr['yaw_acc'], c='r', ls='--', lw=0.7); ax.set_ylim(-40, 60); ax.legend(loc='upper right'); shade(ax)
    ax = axes[4]; ax.plot(t, f['blur_drop'], lw=0.8, label='sharpness / local median'); ax.plot(t, f['consist'], lw=0.8, label='motion direction consistency'); ax.axhline(thr['consist'], c='r', ls='--', lw=0.7); ax.set_ylim(0, 1.5); shade(ax)
    for s in segs: ax.text((s['start'] + s['end']) / 2, 0.2, f"{s['label']}\n{s['duration']:.1f}s", ha='center', fontsize=8)
    ax.legend(loc='upper right'); ax.set_xlabel('time (s)'); axes[0].set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)

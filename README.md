# dronesift

**Sift an hour of drone footage down to the good takes — automatically.**

dronesift watches your DJI footage the way an editor would: it finds the steady, continuous
camera moves (push-ins, orbits, dronies, trucking shots), labels them, and hands you
ready-to-use clips plus an [EDL](https://en.wikipedia.org/wiki/Edit_decision_list) you can drop
straight into DaVinci Resolve, Premiere Pro or Final Cut — original files untouched.

```
kept frames 69.7%  ->  2 segments, 40.3s
  00:12.246 - 00:46.697  34.5s  push_in>push_in+climb  jitter=2.7px/s v=11.5m/s
  01:08.435 - 01:14.308   5.9s  push_in                jitter=7.3px/s v=9.3m/s
```

Every run also produces a self-contained interactive HTML page: 3D flight trajectory coloured
by move type, camera-orientation ticks, an aligned stack of speed / gimbal / image-motion /
keep timelines, and a thumbnail strip you can scrub.

## Why

Reviewing flight footage is a chore: for every usable 20-second orbit there are minutes of
takeoff, hovering, stick twitches and aborted moves. dronesift fuses two signals to judge every
frame — **what the camera did** (per-frame gimbal attitude, velocity and acceleration parsed
from the MP4 itself) and **what the image did** (optical-flow motion, jitter, sharpness,
exposure steps) — then keeps only the frames where both say "this is a clean move".

## Install

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and ffmpeg/ffprobe with HEVC support
(on macOS: `brew install ffmpeg uv`).

```bash
git clone https://github.com/zhoutingunl/dronesift && cd dronesift
uv sync
uv run python -m dronesift /path/to/DJI_0001.D.MP4
```

Or after `uv sync`, the console script works too: `uv run dronesift /path/to/DJI_0001.D.MP4`.

## Usage

```bash
uv run dronesift DJI_xxx.MP4                  # default: lossless copy-mode clips into <video>_cuts/
uv run dronesift DJI_xxx.MP4 --mode encode   # frame-exact HEVC 10-bit (Videotoolbox / libx265)
uv run dronesift DJI_xxx.MP4 --mode prores   # ProRes 422 HQ for grading
uv run dronesift DJI_xxx.MP4 --mode none     # EDL / JSON / report only, no files cut
uv run dronesift DJI_xxx.MP4 --max-shot-s 8  # also split into edit-ready shots <= 8 s (shots/ folder)
uv run dronesift DJI_xxx.MP4 --jitter-px 40 --min-len-s 2   # relax thresholds
```

Outputs, written next to the video in `<video>_cuts/`:

| file | contents |
|---|---|
| `<name>_segments.json` | per-segment start/end (time + frame), move-label sequence, jitter, speed, gimbal rates, orbit radius |
| `<name>.edl` | CMX3600 EDL referencing the **original** file — import into Resolve/Premiere/FCP with no re-encode |
| `<name>.fcpxml` | FCPXML 1.9 spine (same edit) — imports into Final Cut Pro and Resolve |
| `<name>_NN_<label>_<t>s.mp4` | the cut clips (copy mode snaps to keyframes and remuxes losslessly) |
| `<name>_shots.json`, `<name>_shots.edl`, `shots/` | with `--max-shot-s`: segments split into edit-ready shots (edited drone cuts average 4–10 s per shot) |
| `<name>_viz.html` | interactive trajectory + timeline report |
| `<name>_timeline.png` | static diagnostic chart (jitter, motion, telemetry vs thresholds) |
| `<name>_telemetry.csv`, `<name>_flow.csv` | cached per-frame telemetry / optical flow — re-running with new thresholds takes seconds |

## Supported footage

- **Newer DJI models that embed telemetry in the MP4** (`djmd` protobuf timed-metadata track:
  Air 3S, Air 3, Mini 4 Pro, Mavic 3 generation — firmware without `.SRT` sidecars). Field
  mapping was reverse-engineered from an Air 3S; per-frame gimbal attitude, NED velocity,
  GPS and camera exposure are used. Schemas may drift with firmware — if parsing fails on
  your model, `python -m dronesift.pbraw VIDEO.MP4` dumps the raw tree so a mapping can be
  re-inferred (PRs welcome).
- **Older models with `.SRT` sidecars** as a fallback (GPS-differentiated velocity; no gimbal
  angles, so pan/orbit detection is disabled).

macOS gets Videotoolbox hardware decode and encoders automatically; Linux/Windows fall back
to software decode and `libx265`/`prores_ks` (`--hwaccel vaapi`/`cuda` available for the
analysis pass).

## How it works (short version)

1. **Telemetry** — the `djmd` track is decoded per frame: aircraft velocity/attitude/GPS,
   gimbal pitch/roll/yaw + quaternion, ISO/shutter/EV.
2. **Image motion** — ffmpeg decodes to 640×360 grayscale; Shi-Tomasi corners + LK optical
   flow + a RANSAC similarity give per-frame translation/rotation/zoom; the **median** feature
   displacement (robust to parallax) is the jitter measure.
3. **Fusion** — a frame is kept only if the image is free of *perceptual shake* and telemetry
   shows no stick jerks. Perceptual shake is motion that **reverses** (deviates from its path and
   comes back): the fraction of frames whose high-frequency flow flips direction within 0.1 s gates
   every image veto. A violent excursion that never reverses — a whip tilt, a MasterShots move —
   reads as an intentional move and is kept, however fast; motion that oscillates is dropped even
   at small amplitude. Telemetry (aircraft/gimbal acceleration) only vetoes frames whose image is
   actually oscillating. Short shake bursts (a manual dial snap) get a dedicated 0.15 s RMS
   detector — the sustained-shake median filter would dilute them below threshold — and break the
   segment instead of being merged across.
4. **Labelling** — kept runs are merged across sub-second gaps, trimmed, and described as a
   move sequence: `push_in>orbit_CW`, `pull_back+pan>orbit_CCW`, `dronie`, `spiral_out`,
   `truck_L`, ... with the implied orbit radius from v_lat / yaw rate.
5. **Cutting** — keyframe-snapped `-c copy` (DJI writes an I-frame every 0.5 s, so at most
   half a second is lost, zero quality loss) or frame-exact HEVC/ProRes; plus the EDL.

All thresholds are CLI flags (`--jitter-px`, `--yaw-acc`, `--min-len-s`, ...); see
`uv run dronesift --help`. Cached analysis makes threshold tuning a few seconds per run.

Full engineering notes (in Chinese, with worked failure cases) are in
[README.zh-CN.md](README.zh-CN.md).

## Batch runs

`./batch.sh DIR...` walks directories of MP4s two at a time and `python -m dronesift.summary OUT DIR...`
builds a browsable index of every flight with its timeline chart and labels.

## GUI (experimental)

A minimal PySide6 app for the drop-folder → review → export loop:

```bash
uv sync --extra gui
uv run dronesift-gui
```

Add a folder of flights, let the queue run (2 videos at a time), then double-click a finished
row: the interactive report opens in-app (or your browser) with a shot checklist on the side —
untick what you don't want and re-export the selection losslessly into `<out>/export/`.

## Status & caveats

- Thresholds were tuned on a handful of Air 3S flights (4K30–60); treat them as a starting
  point and tune per shoot. Feedback from other models highly welcome.
- This is not an official DJI product and is not affiliated with DJI. The `djmd` format is
  reverse-engineered; names and field numbers belong to whatever DJI's firmware writes.

Licensed under the Apache License 2.0 — see [LICENSE](LICENSE). 中文说明见 [README.zh-CN.md](README.zh-CN.md).

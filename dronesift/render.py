"""Frame-accurate, full-music montage rendering from original stable windows.

The timeline is expressed in absolute output frames.  Reverse playback requires
an explicit opt-in; music always plays forward from zero without time offsets.
"""
from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from fractions import Fraction
from numbers import Real
from pathlib import Path


def _run(args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed ({result.returncode}): {result.stderr[-4000:]}")
    return result


def probe_media(path, *, count_frames=False):
    args = ["ffprobe", "-v", "error", "-threads", "4"]
    if count_frames:
        args.append("-count_frames")
    args += ["-show_streams", "-show_format", "-of", "json", str(path)]
    return json.loads(_run(args).stdout)


def music_info(path, sample_rate=48000):
    """Count decoded samples, including the intro/outro, rather than MP3 padding."""
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1",
         "-ar", str(sample_rate), "-f", "s16le", "-"], capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    samples = len(result.stdout) // 2
    if not samples:
        raise ValueError(f"Music has no decoded audio samples: {path}")
    return dict(path=str(Path(path).resolve()), sample_rate=sample_rate,
                samples=samples, duration=samples / sample_rate, audio_start=0.0)


def frame_timeline(rows, music_duration, fps=30, *, allow_reverse=False):
    """Validate full coverage and safe sources; quantise cumulative boundaries once."""
    if not rows or not math.isfinite(music_duration) or music_duration <= 0:
        raise ValueError("A nonempty timeline and a positive music duration are required")
    if int(fps) != fps or fps <= 0:
        raise ValueError("Output FPS must be a positive integer")
    total_frames = math.ceil(music_duration * fps - 1e-7)
    explicit = all("start_frame" in row and "end_frame" in row for row in rows)
    if any("start_frame" in row or "end_frame" in row for row in rows) and not explicit:
        raise ValueError("Every row must specify both absolute frame boundaries")
    if not explicit:
        duration = sum(float(row["duration"]) for row in rows)
        if abs(duration - music_duration) > 1 / fps + 1e-6:
            raise ValueError("Timeline must include the full music intro and outro")
    cursor, elapsed, result = 0, 0.0, []
    for i, row in enumerate(rows):
        reverse = bool(row.get("reversed"))
        if reverse and not allow_reverse:
            raise ValueError("Reverse playback requires allow_reverse=True")
        speed = row.get("playback_speed", 1.0)
        if isinstance(speed, bool) or not isinstance(speed, Real) or not math.isfinite(speed) or speed <= 0:
            raise ValueError(f"Shot {i + 1} playback_speed must be a finite positive number")
        speed = float(speed)
        start = float(row.get("source_start", row.get("start", math.nan)))
        end = float(row.get("source_end", row.get("end", math.nan)))
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError(f"Invalid source bounds in shot {i + 1}")
        if explicit:
            a, b = row["start_frame"], row["end_frame"]
            if int(a) != a or int(b) != b:
                raise ValueError("Frame boundaries must be integers")
            a, b = int(a), int(b)
        else:
            elapsed += float(row["duration"])
            a, b = cursor, round(elapsed * fps)
            if i == len(rows) - 1:
                b = total_frames
        if a != cursor or b <= a:
            raise ValueError(f"Timeline gap, overlap or empty shot at row {i + 1}")
        duration = (b - a) / fps
        # The planner reserves one output-frame margin for the last slot.
        # A short source is never made longer by freezing its last frame.
        consumed = duration * speed
        if not math.isfinite(consumed) or end - start + 1e-6 < consumed:
            raise ValueError(f"Shot {i + 1} is too short: source {end - start:.6f}s, "
                             f"output {duration:.6f}s at {speed:g}x requires {consumed:.6f}s")
        video = Path(row["video"]).resolve()
        if not video.is_file():
            raise FileNotFoundError(video)
        out = dict(row, video=str(video), source_start=start, source_end=end,
                   start_frame=a, end_frame=b, frames=b - a, duration=duration,
                   output_start=a / fps, output_end=b / fps, reversed=reverse,
                   playback_speed=speed, source_consumed_duration=consumed,
                   source_consumed_start=start, source_consumed_end=start + consumed)
        result.append(out)
        cursor = b
    if cursor != total_frames:
        raise ValueError(f"Timeline has {cursor} frames; full music requires {total_frames}")
    return result


def _encoder_args(name, fps):
    common = ["-c:v", name, "-profile:v", "high", "-pix_fmt", "yuv420p",
              "-bf", "0", "-g", str(fps * 2)]
    if name == "h264_videotoolbox":
        return common + ["-allow_sw", "0", "-b:v", "12M"]
    return common + ["-preset", "veryfast", "-crf", "19", "-threads", "4"]


def probe_acceleration(video, workdir, *, width=1920, height=1080, fps=30, start=0):
    """Probe real sessions; listing a VideoToolbox codec does not establish support."""
    result = dict(encoder="libx264", decoder=None, encoder_reason="non-macOS host",
                  decoder_reason="non-macOS host")
    if platform.system() != "Darwin":
        return result
    encoder_probe = Path(workdir) / "hardware_probe.mp4"
    enc = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         f"testsrc2=size={width}x{height}:rate={fps}", "-frames:v", "2"]
        + _encoder_args("h264_videotoolbox", fps) + [str(encoder_probe)],
        capture_output=True, text=True)
    if enc.returncode == 0:
        result.update(encoder="h264_videotoolbox", encoder_reason="real hardware encode passed")
    else:
        result["encoder_reason"] = enc.stderr.strip()[:1800]
    if encoder_probe.exists():
        encoder_probe.unlink()
    dec = subprocess.run(
        ["ffmpeg", "-v", "error", "-hwaccel", "videotoolbox", "-ss", f"{start:.9f}",
         "-i", str(video), "-frames:v", "2", "-vf", "scale=160:90", "-an", "-f", "null", "-"],
        capture_output=True, text=True)
    # ffmpeg can return success after silently falling back to its software decoder.
    failed = any(t in dec.stderr.lower() for t in
                 ("failed setup", "initialisation returned error", "cannot", "failed", "error"))
    if dec.returncode == 0 and not failed:
        result.update(decoder="videotoolbox", decoder_reason="real hardware decode passed")
    else:
        result["decoder_reason"] = dec.stderr.strip()[:1800]
    return result


def _video_stream(media):
    return next(s for s in media["streams"] if s["codec_type"] == "video")


def _color_args(stream):
    # Preserve source transfer/primaries metadata.  Scaling converts levels to
    # ordinary limited-range H.264; no contrast/saturation/grading is invented.
    args = ["-color_range", "tv"]
    for source, option in (("color_primaries", "-color_primaries"),
                           ("color_transfer", "-color_trc"), ("color_space", "-colorspace")):
        value = stream.get(source)
        if value and value not in ("unknown", "unspecified", "reserved"):
            args += [option, value]
    return args


def _concat_quote(path):
    return str(path).replace("'", "'\\''")


def validate_montage(output, rows, music, fps=30, width=1920, height=1080):
    """Read actual decoded frame counts and stream timestamps, not planned lengths."""
    media = probe_media(output, count_frames=True)
    video = _video_stream(media)
    audio = next((s for s in media["streams"] if s["codec_type"] == "audio"), None)
    expected = rows[-1]["end_frame"]
    actual = int(video.get("nb_read_frames", video.get("nb_frames", 0)))
    if actual != expected:
        raise RuntimeError(f"Final video has {actual} decoded frames; expected {expected}")
    if not audio:
        raise RuntimeError("Final video is missing the music stream")
    audio_duration = float(audio["duration"])
    video_duration = float(video["duration"])
    audio_samples = int(Fraction(int(audio["duration_ts"])) * Fraction(audio["time_base"])
                        * music["sample_rate"])
    if abs(float(video.get("start_time", 0))) > 1e-4 or abs(float(audio.get("start_time", 0))) > 1e-4:
        raise RuntimeError("Video/music does not begin at timeline zero")
    if audio_samples != music["samples"]:
        raise RuntimeError("The music intro or outro was truncated")
    if abs(video_duration - expected / fps) > 1e-4:
        raise RuntimeError("Video duration drifted away from its absolute frame timeline")
    if (video["width"], video["height"]) != (width, height):
        raise RuntimeError("Unexpected output geometry")
    if video["avg_frame_rate"] != f"{fps}/1":
        raise RuntimeError("Unexpected output frame rate")
    return dict(verified=True, frames=actual, video_duration=video_duration,
                audio_duration=audio_duration, audio_samples=audio_samples,
                container_duration=float(media["format"]["duration"]),
                fps=fps, width=width, height=height, audio_codec=audio["codec_name"],
                video_codec=video["codec_name"], video_profile=video.get("profile"),
                color_range=video.get("color_range"), color_space=video.get("color_space"),
                color_transfer=video.get("color_transfer"), color_primaries=video.get("color_primaries"),
                cut_frames=[row["start_frame"] for row in rows[1:]],
                music_start=0.0, video_audio_end_difference=video_duration - audio_duration)


def render_montage(rows, music_path, output_path, workdir=None, *, fps=30,
                   width=1920, height=1080, end_fade=0.6, use_hardware=True,
                   allow_reverse=False):
    """Render stable windows, then mux the complete music forward from zero.

    Each row supplies video, source_start/source_end (or start/end), duration,
    and optionally contiguous start_frame/end_frame and playback_speed (default
    1). Speed changes consume proportionally more source while preserving music
    and cut frames. Explicit reversed rows require allow_reverse=True.
    Returns a validated report.
    """
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(workdir or output.parent / (output.stem + "_render_work")).resolve()
    work.mkdir(parents=True, exist_ok=True)
    music = music_info(music_path)
    timeline = frame_timeline(rows, music["duration"], fps, allow_reverse=allow_reverse)
    acceleration = (probe_acceleration(timeline[0]["video"], work, width=width, height=height,
                                       fps=fps, start=timeline[0]["source_start"])
                    if use_hardware else dict(encoder="libx264", decoder=None,
                                              encoder_reason="hardware disabled", decoder_reason="hardware disabled"))
    sources, parts, part_reports = {}, [], []
    for i, row in enumerate(timeline):
        video = row["video"]
        if video not in sources:
            sources[video] = _video_stream(probe_media(video))
        stream = sources[video]
        source_duration = float(stream.get("duration", math.inf))
        if row["source_end"] > source_duration + 1e-4:
            raise ValueError(f"Shot {i + 1} extends beyond the source video")
        part = work / f"part_{i:03d}.mp4"
        source_range = "full" if stream.get("color_range") == "pc" else "limited"
        vf = (f"trim=end={min(row['source_end'] - row['source_start'], row['source_consumed_duration']):.9f},"
              f"setpts=(PTS-STARTPTS)/{row['playback_speed']:.12g},"
              f"fps={fps}:start_time=0:round=near,"
              f"scale={width}:{height}:force_original_aspect_ratio=decrease:"
              f"force_divisible_by=2:in_range={source_range}:out_range=limited,"
              f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p,"
              f"trim=end_frame={row['frames']}")
        if row["reversed"]:
            # Buffer only the selected, scaled 8-bit output frames, rather
            # than a whole original 4K/60 fps window.  Fade is applied after
            # reversal so it remains at the final playback/music endpoint.
            vf += ",reverse,setpts=PTS-STARTPTS"
        if i == len(timeline) - 1 and end_fade > 0:
            fade = min(float(end_fade), row["duration"], music["duration"])
            # Fade reaches the same musical endpoint as the attached audio.
            fade_start = max(0.0, music["duration"] - row["output_start"] - fade)
            vf += f",fade=t=out:st={fade_start:.9f}:d={fade:.9f}"
        decoder = acceleration["decoder"]
        encoder = acceleration["encoder"]
        cmd = ["ffmpeg", "-v", "error", "-y", "-threads", "4", "-filter_threads", "2"]
        if decoder:
            cmd += ["-hwaccel", decoder]
        cmd += ["-ss", f"{row['source_start']:.9f}", "-i", video,
                "-map", "0:v:0", "-an", "-vf", vf, "-frames:v", str(row["frames"])]
        cmd += _encoder_args(encoder, fps) + _color_args(stream)
        cmd += ["-video_track_timescale", "90000", "-movflags", "+faststart", str(part)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode:
            if not decoder and encoder == "libx264":
                raise RuntimeError(f"Shot {i + 1} render failed: {result.stderr[-4000:]}")
            # Re-render every part with one encoder.  Mixing hardware/software
            # SPS/profile parameters in a stream-copy concat is unsafe.
            print(f"Hardware failed at shot {i + 1}; restarting with software encoding", file=sys.stderr)
            fallback = render_montage(rows, music_path, output, work, fps=fps, width=width,
                                      height=height, end_fade=end_fade, use_hardware=False,
                                      allow_reverse=allow_reverse)
            fallback["acceleration"]["attempted_acceleration"] = acceleration
            fallback["acceleration"]["runtime_fallback"] = result.stderr.strip()[:1800]
            Path(fallback["report_path"]).write_text(json.dumps(fallback, indent=2, ensure_ascii=False), encoding="utf-8")
            return fallback
        actual = _video_stream(probe_media(part, count_frames=True))
        count = int(actual.get("nb_read_frames", actual.get("nb_frames", 0)))
        if count != row["frames"]:
            raise RuntimeError(f"Shot {i + 1} decoded to {count} frames; expected {row['frames']}; refusing freeze/padding")
        part_reports.append(dict(part=str(part), frames=count,
                                 source_start=row["source_start"], source_end=row["source_end"],
                                 playback_speed=row["playback_speed"],
                                 source_consumed_start=row["source_consumed_start"],
                                 source_consumed_end=row["source_consumed_end"],
                                 source_consumed_duration=row["source_consumed_duration"],
                                 start_frame=row["start_frame"], end_frame=row["end_frame"],
                                 reversed=row["reversed"]))
        parts.append(part)
        print(f"render [{i + 1}/{len(timeline)}] {Path(video).name} "
              f"{row['source_start']:.3f}–{row['source_end']:.3f}s -> "
              f"frames {row['start_frame']}–{row['end_frame']}", file=sys.stderr, flush=True)
    concat = work / "concat.ffconcat"
    concat.write_text("ffconcat version 1.0\n" + "".join(f"file '{_concat_quote(p)}'\n" for p in parts), encoding="utf-8")
    af = f"atrim=end_sample={music['samples']},asetpts=PTS-STARTPTS"
    if end_fade > 0:
        fade = min(float(end_fade), music["duration"])
        af += f",afade=t=out:st={music['duration'] - fade:.9f}:d={fade:.9f}"
    _run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
          "-i", str(music_path), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
          "-af", f"aresample={music['sample_rate']},{af}", "-c:a", "aac", "-b:a", "256k",
          "-ar", str(music["sample_rate"]), "-video_track_timescale", "90000",
          # MOV's default 1 kHz movie/edit-list clock can round off real audio
          # samples.  Use the audio sample clock so its whole-song endpoint is exact.
          "-movie_timescale", str(music["sample_rate"]),
          "-movflags", "+faststart", str(output)])
    validation = validate_montage(output, timeline, music, fps, width, height)
    report = dict(output=str(output), music=music, timeline=timeline, acceleration=acceleration,
                  parts=part_reports, validation=validation, end_fade=end_fade,
                  allow_reverse=bool(allow_reverse),
                  reversed_shots=sum(row["reversed"] for row in timeline),
                  accelerated_shots=sum(row["playback_speed"] > 1 for row in timeline),
                  policy=("explicitly approved reverse stable windows; forward full audio from zero; absolute frame cuts"
                          if any(row["reversed"] for row in timeline) else
                          "forward original stable windows; full audio from zero; absolute frame cuts")
                          + "; per-shot playback_speed preserves music and cut frames")
    report_path = output.with_name(output.stem + "_render.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report

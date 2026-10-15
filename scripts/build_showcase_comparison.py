#!/usr/bin/env python3
"""Build the README's real interval-selection example from the original footage.

Usage: python scripts/build_showcase_comparison.py --source path/to/DJI_20261001123356_0013_D.MP4
Requires the project's existing Pillow dependency, ffmpeg/ffprobe, and gifsicle.
The preview cuts out unwanted source time; it does not stabilize any frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SOURCE_NAME = "DJI_20261001123356_0013_D.MP4"
WINDOWS = ((6.373033333333334, 10.477133333333333), (12.746066666666668, 16.449766666666665))
DETECTED_NUDGE = (10.76075, 12.46245)
FPS = 24
WIDTH, HEIGHT = 960, 330
PANEL_WIDTH, PANEL_HEIGHT = 464, 260
BACKGROUND = "#0c1420"
FOREGROUND = "#f2f5fa"
MUTED = "#aab8ca"
GREEN = "#6ae0b3"
RED = "#ff786e"


def font(size: int) -> ImageFont.FreeTypeFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "Arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def decode(source: Path, start: float, end: float, hwaccel: str | None) -> list[Image.Image]:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if hwaccel:
        command += ["-hwaccel", hwaccel]
    command += [
        "-ss", f"{start:.12f}", "-i", str(source), "-t", f"{end - start:.12f}",
        "-map", "0:v:0", "-an", "-sn", "-dn", "-vf",
        f"fps={FPS},scale={PANEL_WIDTH}:{PANEL_HEIGHT}:flags=lanczos,format=rgb24",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    frame_bytes = PANEL_WIDTH * PANEL_HEIGHT * 3
    if len(result.stdout) % frame_bytes:
        raise RuntimeError("Decoder returned an incomplete RGB frame")
    frames = [
        Image.frombytes("RGB", (PANEL_WIDTH, PANEL_HEIGHT), result.stdout[offset:offset + frame_bytes])
        for offset in range(0, len(result.stdout), frame_bytes)
    ]
    if not frames:
        raise RuntimeError("Decoder returned no frames")
    return frames


def center(draw: ImageDraw.ImageDraw, text: str, y: int, face: ImageFont.FreeTypeFont, fill: str) -> None:
    x = 484 + (PANEL_WIDTH - draw.textlength(text, font=face)) / 2
    draw.text((x, y), text, font=face, fill=fill)


def probe(path: Path) -> dict:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    metadata = json.loads(result.stdout)
    stream = metadata["streams"][0]
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "width": stream["width"],
        "height": stream["height"],
        "frames": int(stream["nb_read_frames"]),
        "duration_s": float(metadata["format"]["duration"]),
        "average_frame_rate": stream["avg_frame_rate"],
        "codec": stream["codec_name"],
        "pixel_format": stream["pix_fmt"],
        "audio_streams": sum(item["codec_type"] == "audio" for item in metadata["streams"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/media"))
    parser.add_argument("--hwaccel", help="Optional ffmpeg decoder acceleration, e.g. videotoolbox")
    parser.add_argument("--gif-fps", type=int, default=8)
    parser.add_argument("--gif-lossy", type=int, default=220, help="gifsicle's GIF compression setting")
    args = parser.parse_args()
    if args.source.name != SOURCE_NAME:
        parser.error(f"This documented example requires {SOURCE_NAME}")
    if not shutil.which("gifsicle"):
        parser.error("gifsicle must be available on PATH to build the optimized GIF")
    if not 1 <= args.gif_fps <= FPS or args.gif_lossy < 0:
        parser.error("Use a GIF frame rate between 1 and 24 and a nonnegative compression setting")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    original = decode(args.source, WINDOWS[0][0], WINDOWS[-1][1], args.hwaccel)
    selected_parts = [decode(args.source, start, end, args.hwaccel) for start, end in WINDOWS]
    selected_count = sum(map(len, selected_parts))
    kept_duration = sum(end - start for start, end in WINDOWS)
    if selected_count >= len(original):
        raise RuntimeError("The selected preview must end before the continuous original")

    title_font, clock_font, detail_font = font(18), font(15), font(13)
    end_font = font(23)
    target = args.output_dir / "cut-comparison.mp4"
    encoder = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS), "-i", "pipe:0", "-an", "-map_metadata", "-1", "-map_chapters", "-1",
        "-c:v", "libx264", "-preset", "medium", "-crf", "25", "-maxrate", "1000k",
        "-bufsize", "2000k", "-pix_fmt", "yuv420p", "-color_range", "tv",
        "-colorspace", "bt709", "-color_trc", "bt709", "-color_primaries", "bt709",
        "-movflags", "+faststart", str(target),
    ], stdin=subprocess.PIPE)
    try:
        for index, frame in enumerate(original):
            canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
            draw = ImageDraw.Draw(canvas)
            canvas.paste(frame, (12, 40))
            draw.text((12, 10), "ORIGINAL  /  10.08 s", font=title_font, fill=FOREGROUND)
            draw.text((484, 10), "SELECTED  /  7.81 s kept", font=title_font, fill=GREEN)
            original_source_time = WINDOWS[0][0] + index / FPS
            draw.rectangle((12, 40, 475, 66), fill=BACKGROUND)
            draw.text((22, 45), f"SOURCE {original_source_time:06.3f} s  /  1x", font=clock_font, fill=FOREGROUND)
            removed = WINDOWS[0][1] <= original_source_time < WINDOWS[1][0]
            draw.rectangle((12, 274, 475, 299), fill=BACKGROUND)
            draw.text((22, 279), "CORRECTION + MARGIN  /  REMOVED" if removed else "STABLE SOURCE", font=detail_font, fill=RED if removed else MUTED)
            if removed:
                draw.rectangle((12, 40, 475, 299), outline=RED, width=3)

            if index < selected_count:
                part = 0 if index < len(selected_parts[0]) else 1
                part_index = index if part == 0 else index - len(selected_parts[0])
                canvas.paste(selected_parts[part][part_index], (484, 40))
                selected_source_time = min(WINDOWS[part][0] + part_index / FPS, WINDOWS[part][1])
                draw.rectangle((484, 40, 947, 66), fill=BACKGROUND)
                draw.text((494, 45), f"SOURCE {selected_source_time:06.3f} s  /  1x", font=clock_font, fill=FOREGROUND)
                draw.rectangle((484, 274, 947, 299), fill=BACKGROUND)
                draw.text((494, 279), f"STABLE CUT {part + 1} / 2" + ("  /  SOURCE TIME JUMPS" if part == 1 else ""), font=detail_font, fill=GREEN)
            else:
                center(draw, "SELECTION COMPLETE", 128, end_font, GREEN)
                center(draw, f"{kept_duration:.2f} s kept from 10.08 s", 168, title_font, FOREGROUND)
                center(draw, "2 stable cuts  /  1x playback", 200, clock_font, MUTED)

            draw.text((12, 309), "SOURCE: 6.373 - 16.450 s", font=detail_font, fill=MUTED)
            draw.text((484, 309), "AUTOMATIC CUTS  /  NO DIGITAL STABILIZATION", font=detail_font, fill=MUTED)
            assert encoder.stdin is not None
            encoder.stdin.write(canvas.tobytes())
        assert encoder.stdin is not None
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Preview encoding failed")
    except BaseException:
        encoder.kill()
        encoder.wait()
        raise

    gif = args.output_dir / "cut-comparison.gif"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(target),
        "-filter_complex",
        f"fps={args.gif_fps},scale=640:-2:flags=lanczos,split[a][b];"
        "[a]palettegen=max_colors=64:stats_mode=diff[p];"
        "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
        "-an", "-map_metadata", "-1", "-loop", "0", str(gif),
    ], check=True)
    with tempfile.TemporaryDirectory(prefix="dronesift-showcase-") as temporary:
        optimized = Path(temporary) / gif.name
        subprocess.run([
            "gifsicle", "--optimize=3", f"--lossy={args.gif_lossy}", str(gif), "-o", str(optimized),
        ], check=True)
        shutil.copyfile(optimized, gif)

    manifest = {
        "schema_version": 1,
        "generator": "scripts/build_showcase_comparison.py",
        "source": SOURCE_NAME,
        "source_metadata": SOURCE_NAME.removesuffix(".MP4") + "_cuts_v6/" + SOURCE_NAME.removesuffix(".MP4") + "_segments.json",
        "source_fps": "60000/1001",
        "original_window_s": list((WINDOWS[0][0], WINDOWS[-1][1])),
        "selected_windows_s": [list(window) for window in WINDOWS],
        "selected_source_frames_inclusive": [[382, 627], [764, 985]],
        "discarded_window_including_margins_s": [WINDOWS[0][1], WINDOWS[1][0]],
        "detected_yaw_nudge_s": list(DETECTED_NUDGE),
        "original_duration_s": WINDOWS[-1][1] - WINDOWS[0][0],
        "selected_duration_s": kept_duration,
        "playback_speed": 1,
        "reversed": False,
        "digital_stabilization": False,
        "audio": False,
        "render": {
            "fps": FPS,
            "original_frames": len(original),
            "original_rendered_duration_s": len(original) / FPS,
            "selected_part_frames": list(map(len, selected_parts)),
            "selected_rendered_duration_s": selected_count / FPS,
            "selected_complete_at_frame": selected_count,
            "selected_complete_at_s": selected_count / FPS,
            "selected_end_card_frames": len(original) - selected_count,
            "gif": {
                "width": 640, "fps": args.gif_fps, "palette_colors": 64,
                "dither": "bayer", "bayer_scale": 5, "gifsicle_lossy": args.gif_lossy,
                "timing_note": "GIF delays are quantized to 10 ms, so its measured duration can differ slightly from the MP4. Use the target fps and measured duration; ffprobe may report the reciprocal of the first frame delay as avg_frame_rate.",
            },
            "note": "Each trimmed interval is independently rounded by ffmpeg's fps filter. The right panel jumps source time at the cut and shows an explicit completion card after the selection ends. The raw source continues to the end. Both panels play forward at 1x.",
        },
        "assets": [probe(target), probe(gif)],
    }
    if manifest["assets"][0]["bytes"] >= 1_700_000:
        raise RuntimeError("MP4 exceeds the 1.7 MB showcase budget")
    if manifest["assets"][1]["bytes"] >= 2_000_000:
        raise RuntimeError("GIF exceeds the 2 MB showcase budget")
    (args.output_dir / "comparison-sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_frames": selected_count, "assets": manifest["assets"]}, indent=2))


if __name__ == "__main__":
    main()

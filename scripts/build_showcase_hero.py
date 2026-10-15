"""Build the small, silent README preview from reviewed original footage.

Requires FFmpeg and FFprobe on PATH; does not need Python packages. Optional
gifsicle on PATH reduces the animated GIF size. Example:
    python scripts/build_showcase_hero.py --source-root /path/to/20261001 \
        --out docs/media

The public manifest contains source basenames and edit settings only. Original
GPS, creation-time, audio and other camera metadata are not copied to previews.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


SHOTS = (
    ("DJI_20261001124446_0017_D.MP4", 4.6044, "MOUNTAIN REVEAL"),
    ("DJI_20261001153456_0031_D.MP4", 38.304733333, "VALLEY PUSH-IN"),
    ("DJI_20261001153939_0040_D.MP4", 6.2054, "HIKER TRACK"),
)
DURATION = 4.0
FPS = 24


def run(command):
    subprocess.run(command, check=True)


def probe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/media"))
    args = parser.parse_args()
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            parser.error(f"{tool} must be installed and on PATH")
    for basename, _, _ in SHOTS:
        if not (args.source_root / basename).is_file():
            parser.error(f"source video is missing: {basename}")
    args.out.mkdir(parents=True, exist_ok=True)
    video = args.out / "steady-moves.mp4"
    gif = args.out / "steady-moves.gif"
    poster = args.out / "steady-moves.jpg"
    common = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    strip_metadata = ["-map_metadata", "-1", "-map_chapters", "-1"]

    with tempfile.TemporaryDirectory(prefix="dronesift-hero-") as directory:
        work = Path(directory)
        for index, (basename, start, _) in enumerate(SHOTS):
            filters = (
                "setpts=PTS-STARTPTS,fps=24,"
                "scale=960:540:flags=lanczos:out_color_matrix=bt709:out_range=tv,"
                "format=yuv420p,setsar=1"
            )
            run(common + [
                "-threads", "2", "-ss", f"{start:.9f}",
                "-i", str(args.source_root / basename), "-map", "0:v:0",
                "-vf", filters, "-frames:v", str(int(DURATION * FPS)),
                "-an", "-sn", "-dn", *strip_metadata,
                "-c:v", "libx264", "-preset", "medium", "-crf", "28",
                "-threads", "2", "-pix_fmt", "yuv420p", "-color_range", "tv",
                "-colorspace", "bt709", "-color_trc", "bt709",
                "-color_primaries", "bt709", "-movflags", "+faststart",
                str(work / f"part_{index}.mp4"),
            ])
        concat = work / "concat.ffconcat"
        concat.write_text(
            "ffconcat version 1.0\n" + "".join(
                f"file part_{index}.mp4\n" for index in range(len(SHOTS))
            ), encoding="utf-8",
        )
        run(common + [
            "-f", "concat", "-safe", "0", "-i", str(concat),
            "-map", "0:v:0", "-c:v", "copy", "-an", *strip_metadata,
            "-movflags", "+faststart", str(video),
        ])
        palette_filter = (
            "fps=8,scale=480:-2:flags=lanczos,split[a][b];"
            "[a]palettegen=max_colors=64:reserve_transparent=0:stats_mode=diff[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=5"
        )
        run(common + [
            "-i", str(video), "-filter_complex", palette_filter,
            "-an", *strip_metadata, "-loop", "0", "-final_delay", "12",
            str(gif),
        ])
        gif_optimizer = None
        if shutil.which("gifsicle"):
            optimized = work / "optimized.gif"
            run(["gifsicle", "-O3", "--lossy=200", str(gif),
                 "-o", str(optimized)])
            if optimized.stat().st_size < gif.stat().st_size:
                shutil.copyfile(optimized, gif)
                gif_optimizer = {"tool": "gifsicle", "optimization": 3,
                                 "lossy": 200}
        run(common + [
            "-ss", "2", "-i", str(video), "-frames:v", "1",
            "-q:v", "3", *strip_metadata, str(poster),
        ])

    media = probe(video)
    streams = media["streams"]
    visual = [stream for stream in streams if stream["codec_type"] == "video"]
    if len(streams) != 1 or len(visual) != 1:
        raise RuntimeError("preview must contain only one video stream")
    stream = visual[0]
    expected = {
        "codec_name": "h264", "width": 960, "height": 540,
        "pix_fmt": "yuv420p", "r_frame_rate": "24/1", "color_range": "tv",
        "color_space": "bt709", "color_transfer": "bt709",
        "color_primaries": "bt709", "nb_frames": "288",
    }
    for field, value in expected.items():
        if stream.get(field) != value:
            raise RuntimeError(f"unexpected {field}: {stream.get(field)!r}")
    duration = float(media["format"]["duration"])
    if abs(duration - DURATION * len(SHOTS)) > 1 / FPS:
        raise RuntimeError(f"unexpected video duration: {duration}")
    if video.stat().st_size >= 2_500_000:
        raise RuntimeError("MP4 exceeds the README media size budget")
    if gif.stat().st_size >= 4_000_000:
        print("GIF exceeds the 4 MB target; install gifsicle for a smaller "
              "derivative or link the much smaller MP4.", file=sys.stderr)
    gif_media = probe(gif)
    gif_stream = gif_media["streams"][0]
    if (gif_stream["width"], gif_stream["height"], gif_stream["nb_frames"]) != (
            480, 270, "96"):
        raise RuntimeError("unexpected animated GIF dimensions or frame count")
    if abs(float(gif_media["format"]["duration"]) - duration) > 0.02:
        raise RuntimeError("GIF duration differs from the natural-speed MP4")
    forbidden_tags = ("location", "gps", "creation_time", "make", "model")
    tags = [media["format"].get("tags", {}), stream.get("tags", {})]
    if any(any(term in key.lower() for term in forbidden_tags)
           for group in tags for key in group):
        raise RuntimeError("preview contains unexpected source metadata")
    manifest = {
        "description": "Real stable v6 source windows, silent and at natural speed.",
        "duration_s": duration,
        "audio": False,
        "digital_stabilization": False,
        "metadata": "Source metadata and chapters removed; no GPS included.",
        "shots": [
            {"source": basename, "source_start_s": start,
             "source_end_s": round(start + DURATION, 9),
             "duration_s": DURATION, "playback_speed": 1.0,
             "reversed": False, "label": label}
            for basename, start, label in SHOTS
        ],
        "derivatives": {
            "steady-moves.mp4": {
                "width": 960, "height": 540, "fps": FPS, "frames": 288,
                "codec": "H.264", "pixel_format": "yuv420p",
                "color": "BT.709 limited range", "crf": 28,
                "bytes": video.stat().st_size,
            },
            "steady-moves.gif": {
                "width": 480, "height": 270, "fps": 8, "frames": 96,
                "duration_s": float(gif_media["format"]["duration"]),
                "palette_colors": 64,
                "dither": "bayer", "bayer_scale": 5, "loop": True,
                "final_delay_centiseconds": 12,
                "optional_optimizer": gif_optimizer,
                "bytes": gif.stat().st_size,
            },
            "steady-moves.jpg": {
                "poster_time_s": 2.0, "bytes": poster.stat().st_size,
            },
        },
    }
    (args.out / "hero-sources.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

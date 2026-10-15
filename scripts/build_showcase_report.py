#!/usr/bin/env python3
"""Rebuild the public v6 flight report and its short, silent selected clip.

Uses the current telemetry/flow/segments and cached thumbnail sprite; no original
CSV, absolute path or absolute GPS coordinates are published. The standard viz
report is generated unchanged before this showcase adds an explanatory video.

Example:
    uv run python scripts/build_showcase_report.py --cuts-dir /path/to/0011_cuts_v6
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import quote


SOURCE_NAME = "DJI_20261004123758_0011_D.MP4"
START = 24.457766666666668
DURATION = 8.0
FPS = 24
FRAMES = int(DURATION * FPS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def probe(path: Path) -> dict:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_streams",
         "-show_format", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(completed.stdout)


def relative(path: Path, directory: Path) -> str:
    return Path(os.path.relpath(path.resolve(), directory.resolve())).as_posix()


def embedded_json(document: str, variable: str) -> dict:
    match = re.search(r"\bconst " + re.escape(variable) + r"\s*=\s*", document)
    if not match:
        raise RuntimeError(f"Generated report has no {variable} data")
    value, _ = json.JSONDecoder().raw_decode(document[match.end():])
    return value


def showcase_wrapper(document: str, video: Path, poster: Path, report: Path,
                     selection: dict) -> str:
    # Plotly remains the report's only external dependency. System fonts work
    # without requesting Google Fonts when a reader opens the public example.
    document = re.sub(
        r'<link\b[^>]*href="https://fonts\.googleapis\.com/[^\"]*"[^>]*>\s*',
        "", document,
    )
    document = document.replace('"IBM Plex Sans",system-ui,sans-serif', "system-ui,sans-serif")
    document = document.replace('"IBM Plex Mono",ui-monospace,Menlo,monospace',
                                "ui-monospace,Menlo,monospace")
    document = document.replace('保留帧 <b id="s-kept">', '帧门禁通过 <b id="s-kept">')
    document = document.replace("(M.kept*100).toFixed(0)+'%'", "(M.kept*100).toFixed(2)+'%'")
    movie_uri = html.escape(quote(relative(video, report.parent), safe="/"), quote=True)
    poster_uri = html.escape(quote(relative(poster, report.parent), safe="/"), quote=True)
    raw_fraction = selection["kept_frame_fraction"] * 100
    retained_fraction = selection["retention"] * 100
    # Appended after the original report's .wrap and scripts: the product report
    # itself does not automatically embed or export a selected-footage player.
    return document + f'''
<style>
.showcase-example{{max-width:1400px;margin:0 auto;padding:0 16px 32px;box-sizing:border-box}}
.showcase-example h2{{font-size:20px;margin:0 0 8px}}
.showcase-example p{{max-width:1000px;color:var(--fg2)}}
.showcase-example video{{display:block;width:100%;max-width:960px;height:auto;border-radius:6px;background:#111}}
.showcase-example code{{font-family:var(--mono)}}
</style>
<section class="showcase-example" aria-labelledby="selected-example-title">
 <h2 id="selected-example-title">稳定运镜示例 / Selected footage</h2>
 <p>本开源展示页在标准 v6 分析报告后附加了一段示例视频；播放器由展示脚本添加。<br>
 This showcase appends a selected-footage player to the standard v6 analysis report.</p>
 <p><code>{SOURCE_NAME}</code> · 源区间 / source {START:.6f}–{START + DURATION:.6f} s ·
 8.00 s · 1× 正向原速 / forward · 静音 / silent · 无数字防抖 / no digital stabilization.</p>
 <video controls playsinline preload="metadata" width="960" height="540" poster="{poster_uri}">
  <source src="{movie_uri}" type="video/mp4">
  <a href="{movie_uri}">播放或下载示例 / Play or download the clip</a>
 </video>
 <p>帧门禁通过 {raw_fraction:.2f}%；再经过连续片段、最短时长和边缘裁切，最终选段
 {selection['retained_duration']:.2f} / {selection['source_duration']:.2f} s（{retained_fraction:.2f}%）。
 这里的 8 秒视频只展示第二段的一部分，两种保留率并不相同。<br>
 {raw_fraction:.2f}% passes the frame gate; final continuous selections retain {retained_fraction:.2f}%
 after minimum-length and edge trimming. This 8-second preview is part of selection 2.</p>
 <p>轨迹采用以均值为原点的相对东、北坐标与相对高度；不发布绝对 GPS 或本地文件路径。
 Trajectory coordinates are relative; absolute GPS and local file paths are omitted.</p>
</section>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuts-dir", type=Path, required=True,
                        help="Current v6 directory with segments, telemetry, flow and cached thumbs")
    parser.add_argument("--video", type=Path, help="Override the original source path in segments JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/media"))
    parser.add_argument("--report", type=Path, default=Path("docs/examples/flight-report.html"))
    args = parser.parse_args()
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            parser.error(f"{tool} must be available on PATH")
    segments = args.cuts_dir / (Path(SOURCE_NAME).stem + "_segments.json")
    if not segments.is_file():
        parser.error(f"This reviewed example requires {segments.name}")
    metadata = json.loads(segments.read_text(encoding="utf-8"))
    video = args.video or Path(metadata["video"])
    if not video.is_file() or video.name != SOURCE_NAME:
        parser.error(f"Provide the original {SOURCE_NAME} using --video if it has moved")
    telemetry = args.cuts_dir / (video.stem + "_telemetry.csv")
    flow = args.cuts_dir / (video.stem + "_flow.csv")
    sprite = args.cuts_dir / (video.stem + "_thumbs.jpg")
    for path in (telemetry, flow, sprite):
        if not path.is_file():
            parser.error(f"Required cached analysis asset is missing: {path.name}")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dronesift import segment, shot_quality, viz

    thresholds = {**segment.DEFAULT_THR, **metadata["thresholds"]}
    with telemetry.open(encoding="utf-8", newline="") as stream:
        telemetry_rows = list(csv.DictReader(stream))
    with flow.open(encoding="utf-8", newline="") as stream:
        flow_rows = list(csv.DictReader(stream))
    features, frame_count = segment.features(
        telemetry_rows, flow_rows, metadata["fps"], rev_amp=thresholds["motion_rev_amp"],
    )
    keep, labels = segment.classify_frames(features, thresholds, metadata["fps"])
    recomputed_segments = segment.segments(features, keep, labels, thresholds, metadata["fps"])
    identifying_fields = ("start_frame", "end_frame", "label", "moves", "base")
    identities = lambda segments_: [{key: item[key] for key in identifying_fields} for item in segments_]
    if frame_count != metadata["frames"] or identities(recomputed_segments) != identities(metadata["segments"]):
        raise RuntimeError("Current source code no longer reproduces the saved v6 segment frames and labels")
    recomputed_selection = {
        "source_duration": frame_count / metadata["fps"],
        "retained_duration": sum(item["duration"] for item in recomputed_segments),
        "retention": sum(item["duration"] for item in recomputed_segments) / (frame_count / metadata["fps"]),
        "kept_frame_fraction": float(keep.mean()),
    }
    if any(not math.isclose(value, metadata["selection"][key], abs_tol=1e-9)
           for key, value in recomputed_selection.items()):
        raise RuntimeError("Current source code no longer reproduces the saved v6 selection totals")
    selected = next((segment for segment in metadata["segments"]
                     if segment["start"] <= START and segment["end"] >= START + DURATION), None)
    if selected is None:
        parser.error("The reviewed 8-second window is no longer contained in a current stable segment")
    selection = metadata["selection"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    preview = args.output_dir / "analysis-selection.mp4"
    poster = args.output_dir / "analysis-selection.jpg"
    strip_metadata = ["-map_metadata", "-1", "-map_chapters", "-1"]
    common = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    run(common + [
        "-threads", "2", "-ss", f"{START:.12f}", "-i", str(video),
        "-map", "0:v:0", "-vf",
        "setpts=PTS-STARTPTS,fps=24,"
        "scale=960:540:flags=lanczos:out_color_matrix=bt709:out_range=tv,"
        "format=yuv420p,setsar=1",
        "-frames:v", str(FRAMES), "-an", "-sn", "-dn", *strip_metadata,
        "-c:v", "libx264", "-preset", "medium", "-crf", "27", "-threads", "2",
        "-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709",
        "-color_trc", "bt709", "-color_primaries", "bt709",
        "-movflags", "+faststart", str(preview),
    ])
    run(common + [
        "-ss", "3", "-i", str(preview), "-map", "0:v:0", "-frames:v", "1",
        "-q:v", "3", *strip_metadata, str(poster),
    ])
    # viz.build reuses this cached sprite and recomputes only the existing CSV
    # features. Its temporary files never become public repository assets.
    with tempfile.TemporaryDirectory(prefix="dronesift-public-report-") as temporary:
        work = Path(temporary)
        shutil.copyfile(sprite, work / sprite.name)
        generated = work / "report.html"
        viz.build(str(video), str(telemetry), str(flow), str(segments),
                  str(generated), hwaccel="none")
        document = generated.read_text(encoding="utf-8")
    data, report_metadata = embedded_json(document, "D"), embedded_json(document, "M")
    expected_data = {"t", "e", "n", "u", "sp", "cl", "yr", "pr", "jit", "mpx",
                     "macc", "cons", "keep", "lab", "cy", "gp"}
    if set(data) != expected_data:
        raise RuntimeError("Review newly introduced report fields before publishing")
    if not math.isclose(report_metadata["kept"], selection["kept_frame_fraction"], abs_tol=1e-9):
        raise RuntimeError("Current frame gate no longer matches this saved v6 result")
    document = showcase_wrapper(document, preview, poster, args.report, selection)
    if any(marker in document for marker in (str(video.parent), str(args.cuts_dir.resolve()),
                                             "file://", "fonts.googleapis.com")):
        raise RuntimeError("Public report contains a local path or removed font dependency")
    args.report.write_text(document, encoding="utf-8")

    measured = probe(preview)
    streams = measured["streams"]
    if len(streams) != 1 or streams[0]["codec_type"] != "video":
        raise RuntimeError("Public preview must contain one video stream and no audio or data")
    stream = streams[0]
    expected = {"codec_name": "h264", "width": 960, "height": 540,
                "pix_fmt": "yuv420p", "avg_frame_rate": "24/1", "nb_read_frames": str(FRAMES),
                "color_space": "bt709", "color_transfer": "bt709", "color_primaries": "bt709"}
    if any(stream.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Preview stream failed size, frame count or format validation")
    if not math.isclose(float(measured["format"]["duration"]), DURATION, abs_tol=1e-6):
        raise RuntimeError("Preview duration is not exactly 8 seconds")
    manifest = {
        "schema_version": 1,
        "generator": "scripts/build_showcase_report.py",
        "source": {"basename": video.name, "sha256": sha256(video),
                   "fps": metadata["fps"], "frames": metadata["frames"]},
        "analysis": {"version": "v6", "segments_basename": segments.name,
                     "segments_sha256": sha256(segments),
                     "segment_source_sha256": sha256(Path(segment.__file__)),
                     "playback_speed_source_sha256": sha256(Path(shot_quality.__file__)),
                     "viz_source_sha256": sha256(Path(viz.__file__)),
                     "recomputed_with_current_source": True,
                     "segment_frames_and_labels_match_saved_v6": True,
                     "thresholds": thresholds,
                     "telemetry_sha256": sha256(telemetry), "flow_sha256": sha256(flow),
                     "thumbnail_sprite_sha256": sha256(sprite),
                     "segment_windows_s": [[item["start"], item["end"]] for item in metadata["segments"]],
                     "frame_gate_pass_fraction": selection["kept_frame_fraction"],
                     "final_selected_duration_s": selection["retained_duration"],
                     "source_duration_s": selection["source_duration"],
                     "final_selected_fraction": selection["retention"],
                     "retention_note": "Frame gate pass rate precedes continuous-segment, minimum-length and edge trimming; it is not final selected duration / source duration."},
        "preview": {"source_window_s": [START, START + DURATION], "stable_segment_index": 2,
                    "playback_speed": 1, "reversed": False, "digital_stabilization": False,
                    "audio": False, "fps": FPS, "frames": FRAMES, "duration_s": DURATION,
                    "width": 960, "height": 540, "codec": "h264", "pixel_format": "yuv420p",
                    "colorspace": "bt709", "original_metadata_copied": False,
                    "timing_note": "Source seek begins at the stated interval; output is sampled to 24 fps without changing playback speed."},
        "report": {"generator": "dronesift.viz.build", "showcase_appends_video_player": True,
                   "absolute_paths": False, "absolute_gps": False,
                   "trajectory_coordinates": "Mean-centered east/north in meters; relative altitude",
                   "external_dependencies": ["https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.0/plotly.min.js"]},
        "assets": [{"path": relative(path, args.output_dir), "bytes": path.stat().st_size,
                    "sha256": sha256(path)} for path in (preview, poster, args.report)],
        "validation": {"decoded_frames": int(stream["nb_read_frames"]),
                       "video_duration_s": float(measured["format"]["duration"]),
                       "audio_streams": 0, "absolute_gps_fields": 0,
                       "report_matches_saved_frame_gate": True},
    }
    target = args.output_dir / "analysis-sources.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.report), "manifest": str(target),
                      "assets": manifest["assets"], "validation": manifest["validation"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

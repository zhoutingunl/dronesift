"""Full-song frame coverage, safe windows and actual mux validation."""
import math
import shutil
import subprocess
import wave

import numpy as np
import pytest

from dronesift.render import frame_timeline, render_montage


def _rows(source):
    # Individual rounding yields 9 frames; full music coverage needs 11.
    return [dict(video=str(source), start=0, end=1, duration=0.115),
            dict(video=str(source), start=1, end=2, duration=0.115),
            dict(video=str(source), start=2, end=3, duration=0.115)]


def test_round_absolute_boundaries_without_per_shot_drift(tmp_path):
    source = tmp_path / "input.mp4"
    source.touch()
    rows = frame_timeline(_rows(source), 0.345, 30)
    assert [r["frames"] for r in rows] == [3, 4, 4]
    assert rows[-1]["end_frame"] == math.ceil(0.345 * 30)
    assert all(a["end_frame"] == b["start_frame"] for a, b in zip(rows, rows[1:]))


def test_reject_missing_music_intro_and_reverse(tmp_path):
    source = tmp_path / "input.mp4"
    source.touch()
    with pytest.raises(ValueError, match="gap"):
        frame_timeline([dict(video=str(source), start=0, end=1, start_frame=1, end_frame=30)], 1)
    with pytest.raises(ValueError, match="Reverse"):
        frame_timeline([dict(video=str(source), start=0, end=1, duration=1, reversed=True)], 1)
    with pytest.raises(ValueError, match="intro and outro"):
        frame_timeline([dict(video=str(source), start=0, end=1, duration=0.5)], 1)


def test_short_source_is_rejected_instead_of_freezing(tmp_path):
    source = tmp_path / "input.mp4"
    source.touch()
    with pytest.raises(ValueError, match="too short"):
        frame_timeline([dict(video=str(source), start=0, end=0.9,
                             start_frame=0, end_frame=30)], 1)


@pytest.mark.parametrize("speed", [True, False, "2", None, 0, -1, math.inf, math.nan])
def test_reject_invalid_playback_speed(tmp_path, speed):
    source = tmp_path / "input.mp4"
    source.touch()
    with pytest.raises(ValueError, match="playback_speed"):
        frame_timeline([dict(video=str(source), start=0, end=2, duration=1,
                             playback_speed=speed)], 1)


def test_speed_uses_more_source_without_changing_absolute_cut_frames(tmp_path):
    source = tmp_path / "input.mp4"
    source.touch()
    rows = [dict(video=str(source), start=0, end=1.2, start_frame=0, end_frame=30),
            dict(video=str(source), start=2, end=3.7, start_frame=30, end_frame=60,
                 playback_speed=1.5),
            dict(video=str(source), start=4, end=6.2, start_frame=60, end_frame=90,
                 playback_speed=2)]
    timeline = frame_timeline(rows, 3)
    assert [row["playback_speed"] for row in timeline] == [1, 1.5, 2]
    assert [row["source_consumed_duration"] for row in timeline] == [1, 1.5, 2]
    assert [row["source_consumed_end"] for row in timeline] == [1, 3.5, 6]
    assert [row["frames"] for row in timeline] == [30, 30, 30]
    assert [row["start_frame"] for row in timeline] == [0, 30, 60]
    rows[1]["end"] = 3.49
    with pytest.raises(ValueError, match="too short.*1.5x"):
        frame_timeline(rows, 3)


def test_reverse_permission_preserves_only_explicit_reversed_rows(tmp_path):
    source = tmp_path / "input.mp4"
    source.touch()
    rows = [dict(video=str(source), start=0, end=1, start_frame=0, end_frame=30, reversed=True),
            dict(video=str(source), start=1, end=2, start_frame=30, end_frame=60)]
    result = frame_timeline(rows, 2, allow_reverse=True)
    assert [row["reversed"] for row in result] == [True, False]
    assert [(row["source_start"], row["source_end"]) for row in result] == [(0, 1), (1, 2)]


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="requires ffmpeg")
@pytest.mark.parametrize("source_rate", ["60000/1001", "30000/1001"])
@pytest.mark.parametrize("reverse", [False, True])
def test_render_exact_nonintegral_full_music_and_frame_cuts(tmp_path, source_rate, reverse):
    source = tmp_path / "moving.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc2=size=160x90:rate={source_rate}", "-t", "4", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709",
                    "-color_primaries", "bt709", "-color_trc", "bt709", str(source)], check=True)
    # 2.317333333 seconds: the default 1 kHz MOV edit-list timebase loses
    # 16 samples here despite those samples existing in the final AAC packet.
    sr, samples = 48000, 111232
    t = np.arange(samples) / sr
    # Distinct, changing phase allows an actual check that audio was not offset.
    waveform = (0.35 * np.sin(2 * np.pi * (340 * t + 50 * t ** 2))).astype(np.float64)
    music = tmp_path / "music.wav"
    with wave.open(str(music), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(np.round(waveform * 32767).astype("<i2").tobytes())
    frames = math.ceil(samples / sr * 30)
    rows = [dict(video=str(source), source_start=0.7, source_end=0.7 + 31 / 30,
                 start_frame=0, end_frame=31, reversed=reverse),
            dict(video=str(source), source_start=2.0, source_end=2.0 + (frames - 31) / 30,
                 start_frame=31, end_frame=frames, reversed=reverse)]
    output = tmp_path / "output.mp4"
    result = render_montage(rows, music, output, tmp_path / "work", width=160, height=90,
                           use_hardware=False, allow_reverse=reverse)
    validation = result["validation"]
    assert validation["verified"]
    assert validation["frames"] == frames
    assert validation["cut_frames"] == [31]
    assert validation["audio_duration"] == pytest.approx(samples / sr, abs=1e-4)
    assert validation["audio_samples"] == samples
    assert 0 <= validation["video_audio_end_difference"] < 1 / 30
    assert validation["color_range"] == "tv"
    assert validation["color_space"] == "bt709"
    raw_audio = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(output), "-map", "0:a:0",
                                         "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"])
    decoded = np.frombuffer(raw_audio, dtype="<f4")
    assert np.corrcoef(waveform[:sr], decoded[:sr])[0, 1] > 0.99
    # Read packet timestamps to establish the frame grid at every cut, not just a total.
    from dronesift.cut import video_packets
    packets = video_packets(str(output), 30)
    assert len(packets) == frames
    assert packets[31]["pts"] == pytest.approx(31 / 30, abs=1e-6)
    assert all(p["pts"] == pytest.approx(i / 30, abs=1e-6) for i, p in enumerate(packets))


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="requires ffmpeg")
def test_reverse_exact_source_color_sequence_keeps_audio_forward_and_fade_at_end(tmp_path):
    # Yellow and white sentinels surround the exact safe source interval.
    # The selected 60 frames are 20 red, 20 green, 20 blue; all must appear
    # in the opposite order, without either out-of-window sentinel.
    width, height = 96, 64
    colors = np.empty((120, height, width, 3), dtype=np.uint8)
    colors[:30] = (220, 220, 0)
    colors[30:50] = (220, 0, 0)
    colors[50:70] = (0, 220, 0)
    colors[70:90] = (0, 0, 220)
    colors[90:] = (220, 220, 220)
    source = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{width}x{height}", "-r", "30", "-i", "-", "-an", "-c:v", "libx264",
                    "-crf", "0", "-pix_fmt", "yuv444p", str(source)], input=colors.tobytes(), check=True)
    sr, samples = 48000, 96000
    t = np.arange(samples) / sr
    waveform = 0.35 * np.sin(2 * np.pi * (340 * t + 50 * t ** 2))
    music = tmp_path / "music.wav"
    with wave.open(str(music), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(np.round(waveform * 32767).astype("<i2").tobytes())
    output = tmp_path / "reverse.mp4"
    rows = [dict(video=str(source), source_start=1, source_end=3, reversed=True,
                 start_frame=0, end_frame=60)]
    with pytest.raises(ValueError, match="allow_reverse=True"):
        render_montage(rows, music, output, tmp_path / "work", width=width, height=height,
                       use_hardware=False)
    assert not output.exists()
    result = render_montage(rows, music, output, tmp_path / "work", width=width, height=height,
                           use_hardware=False, allow_reverse=True, end_fade=0.3)
    assert result["timeline"][0]["reversed"] is True
    assert result["parts"][0]["reversed"] is True
    assert result["reversed_shots"] == 1
    assert result["validation"]["frames"] == 60
    assert result["validation"]["audio_samples"] == samples
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(output), "-map", "0:v:0",
                                   "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
    means = np.frombuffer(raw, np.uint8).reshape(60, height, width, 3).mean(axis=(1, 2))
    expected_channel = np.repeat([2, 1, 0], 20)
    assert np.array_equal(means.argmax(axis=1), expected_channel)
    assert np.all(means[np.arange(60), expected_channel] > 10)
    assert np.all(np.sort(means, axis=1)[:, :2] < 12)  # No yellow/white outside the stable window.
    assert means[0, 2] > 190  # Playback begins unfaded at the source window's end.
    assert means[-1, 0] < means[45, 0] * 0.2  # Fade acts at the reversed playback end.
    audio = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(output), "-map", "0:a:0",
                                     "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"])
    decoded = np.frombuffer(audio, dtype="<f4")
    assert np.corrcoef(waveform[:sr], decoded[:sr])[0, 1] > 0.99
    assert abs(np.corrcoef(waveform[-sr:][::-1], decoded[:sr])[0, 1]) < 0.1


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="requires ffmpeg")
def test_actual_mixed_speeds_preserve_cuts_music_and_end_fade(tmp_path):
    # Each two source frames increase brightness by one.  Decoded output can
    # therefore establish which real source times 1x, 1.5x and 2x sampled.
    width, height, source_fps = 96, 64, 60
    source_gray = 16 + np.arange(420, dtype=np.uint16) // 2
    colors = np.broadcast_to(source_gray[:, None, None, None], (420, height, width, 3)).astype(np.uint8)
    source = tmp_path / "clock.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{width}x{height}", "-r", str(source_fps), "-i", "-", "-an",
                    "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p", str(source)],
                   input=colors.tobytes(), check=True)
    sr, samples = 48000, 144000
    t = np.arange(samples) / sr
    waveform = 0.35 * np.sin(2 * np.pi * (340 * t + 50 * t ** 2))
    music = tmp_path / "music.wav"
    with wave.open(str(music), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(np.round(waveform * 32767).astype("<i2").tobytes())
    # Safe bounds include extra source.  The renderer must consume only the
    # required accelerated duration, leaving cut positions and the song intact.
    rows = [dict(video=str(source), source_start=0.5, source_end=1.9,
                 start_frame=0, end_frame=30),
            dict(video=str(source), source_start=2, source_end=3.9, playback_speed=1.5,
                 start_frame=30, end_frame=60),
            dict(video=str(source), source_start=4, source_end=6.4, playback_speed=2,
                 start_frame=60, end_frame=90)]
    output = tmp_path / "mixed.mp4"
    report = render_montage(rows, music, output, tmp_path / "work", width=width, height=height,
                            use_hardware=False, end_fade=0.2)
    assert report["validation"]["frames"] == 90
    assert report["validation"]["cut_frames"] == [30, 60]
    assert report["validation"]["audio_samples"] == samples
    assert report["accelerated_shots"] == 2
    assert [p["playback_speed"] for p in report["parts"]] == [1, 1.5, 2]
    assert [p["source_consumed_end"] for p in report["parts"]] == [1.5, 3.5, 6]
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(output), "-map", "0:v:0",
                                   "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
    means = np.frombuffer(raw, np.uint8).reshape(90, height, width, 3).mean(axis=(1, 2, 3))
    for start_frame, source_start, speed in [(0, 0.5, 1), (30, 2, 1.5), (60, 4, 2)]:
        for frame in [0, 7, 15, 23]:
            expected = 16 + (source_start + frame / 30 * speed) * 30
            assert means[start_frame + frame] == pytest.approx(expected, abs=2.5)
    assert means[29] > means[23]  # No fade at the first cut.
    assert means[59] > means[53]  # No fade at the second cut.
    assert means[-1] < means[83] * 0.25  # Fade remains at the playback/song end.
    audio = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(output), "-map", "0:a:0",
                                     "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"])
    decoded = np.frombuffer(audio, dtype="<f4")
    assert np.corrcoef(waveform[:int(2.7 * sr)], decoded[:int(2.7 * sr)])[0, 1] > 0.99

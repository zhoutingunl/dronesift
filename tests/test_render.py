"""Full-song frame coverage, safe windows and actual mux validation."""
import math
import shutil
import subprocess
import wave
from fractions import Fraction

import numpy as np
import pytest

from dronesift.render import frame_timeline, render_montage
from dronesift import render


def _rows(source):
    # Individual rounding yields 9 frames; full music coverage needs 11.
    return [dict(video=str(source), start=0, end=1, duration=0.115),
            dict(video=str(source), start=1, end=2, duration=0.115),
            dict(video=str(source), start=2, end=3, duration=0.115)]


def test_renderer_rejects_unassessed_or_over_limit_speed_before_encoding(tmp_path, monkeypatch,
                                                                        playback_profile):
    source = tmp_path / 'source.mp4'
    source.touch()
    output = tmp_path / 'blocked.mp4'
    monkeypatch.setattr(render, 'music_info', lambda _: dict(duration=2))
    def must_not_encode(*args, **kwargs):
        raise AssertionError('unsupported acceleration reached encoder setup')
    monkeypatch.setattr(render, 'probe_acceleration', must_not_encode)
    row = dict(video=str(source), source_start=0, source_end=3.5,
               start_frame=0, end_frame=60, playback_speed=1.5)
    with pytest.raises(ValueError, match='requires a playback_speed_profile'):
        render_montage([row], 'unused.wav', output)
    row.update(max_playback_speed=1.25, playback_speed_profile=playback_profile(0, 3, maximum=1.25))
    with pytest.raises(ValueError, match='exceeds'):
        render_montage([row], 'unused.wav', output)
    row.update(max_playback_speed=2, playback_speed_profile=playback_profile(0, 3.5))
    with pytest.raises(ValueError, match='exact consumed source window'):
        render_montage([row], 'unused.wav', output)
    assert not output.exists()


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


def test_photo_timeline_has_no_fictional_video_consumption(tmp_path):
    image = tmp_path / '秋林.JPG'
    image.touch()
    row = dict(media_type='photo', image=str(image), start_frame=0, end_frame=90)
    result = frame_timeline([row], 3)[0]
    assert result['media_type'] == 'photo'
    assert result['frames'] == 90 and result['duration'] == 3
    assert result['photo_motion'] == dict(zoom_start=1, zoom_end=1.03, center_x=.5, center_y=.5)
    assert result['image'] == str(image.resolve())
    assert not any(k.startswith('source_consumed') for k in result)
    assert 'video' not in result and 'playback_speed_profile' not in result


@pytest.mark.parametrize('change', [
    dict(reversed=True), dict(playback_speed=1.5), dict(source_start=0),
    dict(playback_speed_profile={'max_playback_speed': 2}), dict(video='not-a-photo.mp4'),
    dict(photo_motion='zoom'), dict(photo_motion={'zoom_start': .99}),
    dict(photo_motion={'zoom_end': 1.16}), dict(photo_motion={'zoom_end': math.nan}),
    dict(photo_motion={'center_x': True}), dict(photo_motion={'center_y': 1.1}),
    dict(photo_motion={'unknown': 1}), dict(media_type='gif'), dict(image=None),
])
def test_invalid_photo_metadata_cannot_reach_encoder(tmp_path, change):
    image = tmp_path / 'photo.jpg'; image.touch()
    row = dict(media_type='photo', image=str(image), start_frame=0, end_frame=30)
    row.update(change)
    with pytest.raises(ValueError):
        frame_timeline([row], 1, allow_reverse=True)


def test_photo_only_hardware_probe_tests_encoder_without_decoding_jpeg(tmp_path, monkeypatch):
    from types import SimpleNamespace
    commands = []
    monkeypatch.setattr(render.platform, 'system', lambda: 'Darwin')
    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stderr='')
    monkeypatch.setattr(render.subprocess, 'run', run)
    result = render.probe_acceleration(None, tmp_path, width=160, height=90)
    assert result['encoder'] == 'h264_videotoolbox' and result['decoder'] is None
    assert len(commands) == 1 and 'lavfi' in commands[0]
    assert 'photo-only' in result['decoder_reason']


def _photo_and_audio(tmp_path, samples):
    from PIL import Image, ImageDraw
    image = tmp_path / '秋林 #100%.JPG'
    canvas = Image.new('RGB', (800, 600), (40, 40, 40))
    draw = ImageDraw.Draw(canvas)
    # A feature away from the centre establishes actual zoom, rather than
    # relying on encoder noise to distinguish two decoded still frames.
    draw.rectangle((250, 0, 350, 599), fill=(230, 10, 10))
    draw.rectangle((0, 0, 799, 65), fill=(10, 230, 10))
    draw.rectangle((0, 535, 799, 599), fill=(10, 230, 10))
    canvas.save(image, quality=95)
    sr = 48000
    t = np.arange(samples) / sr
    waveform = 0.3 * np.sin(2 * np.pi * (320*t + 65*t*t))
    music = tmp_path / 'music.wav'
    with wave.open(str(music), 'wb') as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(sr)
        stream.writeframes(np.round(waveform*32767).astype('<i2').tobytes())
    return image, music, waveform


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='requires ffmpeg')
def test_actual_mixed_photo_video_keeps_cuts_audio_and_eased_motion(tmp_path, monkeypatch):
    image, music, waveform = _photo_and_audio(tmp_path, 144000)
    source = tmp_path / 'blue.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    'color=blue:size=160x90:rate=30', '-t', '1.2', '-vf',
                    'setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709',
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', '-color_range', 'tv', '-colorspace', 'bt709',
                    '-color_primaries', 'bt709', '-color_trc', 'bt709', str(source)], check=True)
    probed = []
    def probe(video, work, **kwargs):
        probed.append(video)
        return dict(encoder='libx264', decoder=None, encoder_reason='test', decoder_reason='test')
    monkeypatch.setattr(render, 'probe_acceleration', probe)
    rows = [dict(media_type='photo', image=str(image), start_frame=0, end_frame=30,
                 photo_motion=dict(zoom_start=1, zoom_end=1.10)),
            dict(video=str(source), source_start=0, source_end=1, start_frame=30, end_frame=60),
            dict(media_type='photo', image=str(image), start_frame=60, end_frame=90,
                 photo_motion=dict(zoom_start=1.10, zoom_end=1))]
    output = tmp_path / 'mixed.mp4'
    report = render_montage(rows, music, output, tmp_path / 'work', width=160, height=90,
                            end_fade=0, use_hardware=True)
    assert probed == [str(source.resolve())]  # Never try video acceleration on the first JPG.
    assert report['validation']['frames'] == 90
    assert report['validation']['cut_frames'] == [30, 60]
    assert report['validation']['audio_samples'] == 144000
    assert report['photo_shots'] == 2 and report['video_shots'] == 1
    assert [p['media_type'] for p in report['parts']] == ['photo', 'video', 'photo']
    assert report['playback_speed_checks'][0]['status'] == 'not_applicable'
    assert 'source_consumed_end' not in report['parts'][0]
    assert 'playback_speed_profile' not in report['parts'][0]
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(output), '-map', '0:v:0',
                                   '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'])
    decoded = np.frombuffer(raw, np.uint8).reshape(90, 90, 160, 3)
    centres = []
    for frame in decoded[:30]:
        mask = (frame[:, :, 0] > 150) & (frame[:, :, 1] < 60) & (frame[:, :, 2] < 60)
        centres.append(np.nonzero(mask)[1].mean())
    assert centres[0] - centres[-1] > .8  # Chroma quantisation can move a thresholded edge by half a pixel.
    assert max(abs(np.diff(centres))) < 1  # No one-output-pixel crop stair steps.
    assert decoded[45, :, :, 2].mean() > decoded[45, :, :, 0].mean() + 150
    # The top/bottom green sentinels in the 4:3 photo were cover-cropped out.
    assert decoded[0, :, :, 1].mean() < 65
    for part in report['parts']:
        stream = render._video_stream(render.probe_media(part['part']))
        assert stream['color_range'] == 'tv' and stream['color_space'] == 'bt709'
        assert stream['color_transfer'] == 'bt709' and stream['color_primaries'] == 'bt709'
    audio = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(output), '-map', '0:a:0',
                                     '-ac', '1', '-ar', '48000', '-f', 'f32le', '-'])
    assert np.corrcoef(waveform, np.frombuffer(audio, '<f4')[:len(waveform)])[0, 1] > .99


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='requires ffmpeg')
def test_actual_photo_only_full_music_and_final_fade(tmp_path):
    image, music, _ = _photo_and_audio(tmp_path, 51360)  # 1.07s -> 33 output frames.
    rows = [dict(media_type='photo', image=str(image), start_frame=0, end_frame=33)]
    output = tmp_path / 'photo-only.mp4'
    report = render_montage(rows, music, output, tmp_path / 'work', width=160, height=90,
                            end_fade=.2, use_hardware=False)
    assert report['validation']['frames'] == 33 and report['validation']['audio_samples'] == 51360
    assert report['photo_shots'] == 1 and report['video_shots'] == 0
    assert report['validation']['video_audio_end_difference'] == pytest.approx(.03, abs=1e-5)
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(output), '-map', '0:v:0',
                                   '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'])
    frames = np.frombuffer(raw, np.uint8).reshape(33, 90, 160, 3)
    assert frames[-1].mean() < frames[20].mean()*.2


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
def test_actual_mixed_speeds_preserve_cuts_music_and_end_fade(tmp_path, playback_profile):
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
    for row in rows[1:]:
        # This test verifies actual speed/timing; quality estimation is tested
        # separately. Assess only consumed frames, not the unused source reserve.
        row['playback_speed_profile'] = playback_profile(
            row['source_start'], row['source_start'] + row['playback_speed'])
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


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='requires ffmpeg')
@pytest.mark.parametrize('source_rate', ['30000/1001', '60000/1001'])
def test_fractional_seek_eof_keeps_last_real_frame_inside_exact_146_frame_window(tmp_path, source_rate):
    # Reproduce the production 61.662–66.5286667s seek phase. Every native
    # source frame carries its index as 16 high-contrast binary vertical bars,
    # so decoded output identifies real source frames even after H.264.
    width, height, frames = 128, 64, 146
    source = tmp_path / 'frame-ids.mp4'
    pattern = (f"nullsrc=size={width}x{height}:rate={source_rate},"
               "geq=lum='16+220*mod(floor(N/pow(2,floor(X/8))),2)':cb=128:cr=128")
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', pattern,
                    '-t', '67', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '0',
                    '-bf', '0', '-g', '60', '-pix_fmt', 'yuv420p', str(source)], check=True)
    start, duration = 61.662, frames/30
    sr, samples = 48000, frames*1600
    music = tmp_path / 'music.wav'
    waveform = .25*np.sin(2*np.pi*440*np.arange(samples)/sr)
    with wave.open(str(music), 'wb') as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(sr)
        stream.writeframes(np.round(waveform*32767).astype('<i2').tobytes())
    output = tmp_path / 'exact.mp4'
    row = dict(video=str(source), source_start=start, source_end=start+duration,
               start_frame=0, end_frame=frames)
    report = render_montage([row], music, output, tmp_path / 'work', width=width, height=height,
                            use_hardware=False, end_fade=0)
    assert report['validation']['frames'] == frames
    assert report['validation']['audio_samples'] == samples
    assert report['timeline'][0]['source_consumed_start'] == start
    assert report['timeline'][0]['source_consumed_end'] == start+duration
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(output),
                                   '-vf', 'extractplanes=y', '-pix_fmt', 'gray', '-f', 'rawvideo', '-'])
    pictures = np.frombuffer(raw, np.uint8).reshape(frames, height, width)
    bits = pictures[:, 8:-8].mean(axis=1).reshape(frames, 16, 8).mean(axis=2) > 128
    source_ids = bits.astype(np.int64) @ (2**np.arange(16))
    rate = float(Fraction(source_rate))
    first = math.ceil(start*rate-1e-8)
    last = math.ceil((start+duration)*rate-1e-8)-1
    assert source_ids[0] == first and source_ids[-1] == last
    assert np.all(source_ids/rate >= start) and np.all(source_ids/rate < start+duration)
    # 29.97 has 145 in-range frames for 146 CFR ticks: one normal cadence
    # repetition is required. At 59.94 all 146 selected frames are distinct.
    assert len(np.unique(source_ids)) == min(frames, last-first+1)
    assert np.all(np.diff(source_ids) >= 0)
    assert np.sum(np.diff(source_ids) == 0) <= (1 if rate < 30 else 0)

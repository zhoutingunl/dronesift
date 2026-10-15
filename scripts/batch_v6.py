"""Batch current v6 analysis into a writable stage, with verified checkpoints.

Example:
    .venv/bin/python scripts/batch_v6.py SOURCE_DIR ... --out validation/batch_v6
    .venv/bin/python scripts/batch_v6.py SOURCE_DIR ... --out validation/batch_v6 --publish

Analysis only reads original directories. Publishing requires a separately
authorized invocation and refuses existing destination outputs.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import validate_v5

VARIANT = 'v6'
ALGORITHM_FILES = ('segment.py', 'cut.py', '__main__.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def algorithm_hashes():
    return {name: hashlib.sha256((ROOT / 'dronesift' / name).read_bytes()).hexdigest()
            for name in ALGORITHM_FILES}


def source_stat(video):
    stat = video.stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def load_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def checkpoint_path(stage, source, video):
    return stage / 'checkpoints' / source.name / (video.stem + '.json')


def probe_source(video):
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
         'stream=nb_frames,r_frame_rate,duration:format=duration', '-of', 'json', str(video)],
        check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    stream = data['streams'][0]
    raw_duration = stream.get('duration')
    duration = float(raw_duration if raw_duration not in (None, 'N/A') else data['format']['duration'])
    numerator, denominator = stream['r_frame_rate'].split('/')
    fps = float(numerator) / float(denominator)
    frames = stream.get('nb_frames')
    return dict(source_seconds=duration, fps=fps,
                frames=int(frames) if frames and frames != 'N/A' else None)


def measurement_issues(video, target, probe):
    """Detect interrupted measurement files without reading or changing sources."""
    problems = {}
    flow = target / (video.stem + '_flow.csv')
    if flow.exists():
        try:
            count = 0
            with flow.open(newline='') as handle:
                reader = csv.DictReader(handle)
                required = {'frame', 'n_tracks', 'dx', 'dy', 'da', 'ds'}
                if not required.issubset(reader.fieldnames or []):
                    raise ValueError('missing flow columns')
                for count, row in enumerate(reader, 1):
                    if int(row['frame']) != count - 1:
                        raise ValueError('non-contiguous flow frame index')
                    if any(not math.isfinite(float(row[key])) for key in required):
                        raise ValueError('invalid flow sample')
            expected = probe['frames']
            if expected is not None and count != expected:
                raise ValueError(f'flow has {count} frames, source has {expected}')
            if expected is None and abs(count / probe['fps'] - probe['source_seconds']) > 0.2:
                raise ValueError('flow does not cover full source duration')
            if count < 2:
                raise ValueError('empty flow measurements')
        except (OSError, ValueError, TypeError, KeyError) as exc:
            problems[flow] = str(exc)
    telemetry = target / (video.stem + '_telemetry.csv')
    if telemetry.exists():
        try:
            with telemetry.open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) < 2:
                raise ValueError('empty telemetry measurements')
            points = [float(row['pts']) for row in rows]
            if not all(math.isfinite(point) for point in points):
                raise ValueError('invalid telemetry timestamps')
            if min(points) > 0.2 or max(points) < probe['source_seconds'] - 0.2:
                raise ValueError('telemetry does not cover full source duration')
        except (OSError, ValueError, TypeError, KeyError) as exc:
            problems[telemetry] = str(exc)
    return problems


def verify_result(video, target, row):
    if not row or row.get('verified') is not True or row.get('video') != video.name:
        raise ValueError('invalid verification row')
    expected_folder = video.stem + '_cuts_v6'
    if row.get('v5_folder') != expected_folder or target.name != expected_folder:
        raise ValueError('result directory does not match source')
    metadata = load_json(target / (video.stem + '_segments.json'))
    manifest = load_json(target / (video.stem + '_exports.json'))
    if not metadata or manifest is None:
        raise ValueError('missing segments/export manifest')
    if Path(metadata.get('video', '')).resolve() != video:
        raise ValueError('segments source does not match original')
    if Path(manifest.get('video', '')).resolve() != video:
        raise ValueError('export source does not match original')
    if manifest.get('skipped'):
        raise ValueError('manifest contains skipped selected segments')
    clips = manifest.get('clips')
    if not isinstance(clips, list):
        raise ValueError('missing clip list')
    names = []
    for clip in clips:
        name = clip.get('path', '')
        if not name or Path(name).name != name:
            raise ValueError('clip must be a direct child of result directory')
        if clip.get('verified') is not True:
            raise ValueError(f'unverified clip: {name}')
        if not (target / name).is_file() or (target / name).stat().st_size == 0:
            raise ValueError(f'missing/empty exported clip: {name}')
        names.append(target.name + '/' + name)
    if row.get('v5_clips') != names:
        raise ValueError('checkpoint clips do not match manifest')
    if abs(float(row['v5_exported_seconds']) - float(manifest['total_duration'])) > 1e-6:
        raise ValueError('checkpoint duration does not match manifest')
    if row.get('v5_segments') != len(metadata.get('segments', [])):
        raise ValueError('checkpoint segment count does not match metadata')
    return len(clips)


def resumable(checkpoint, video, target, hashes):
    if not checkpoint or checkpoint.get('status') != 'success':
        return False
    if checkpoint.get('source') != str(video) or checkpoint.get('source_stat') != source_stat(video):
        return False
    if checkpoint.get('algorithm_sha256') != hashes:
        return False
    try:
        verify_result(video, target, checkpoint.get('row'))
        return True
    except (ValueError, TypeError, KeyError, OSError):
        return False


def clean_generated_outputs(video, target):
    """Keep raw caches; remove stale cut files before a changed/failed rerun."""
    if not target.exists():
        return
    for path in target.glob(video.stem + '*.mp4'):
        path.unlink()
    shots = target / 'shots'
    if shots.exists():
        shutil.rmtree(shots)
    for suffix in ('_segments.json', '_exports.json', '_shots.json'):
        (target / (video.stem + suffix)).unlink(missing_ok=True)


def analyze_one(video, output, hashes, stat, hwaccel='auto'):
    checkpoint = dict(source=str(video), source_stat=stat, algorithm_sha256=hashes,
                      status='error', finished_at=now())
    target = output / (video.stem + '_cuts_v6')
    try:
        probe = probe_source(video)
        checkpoint['source_seconds'] = probe['source_seconds']
        for path, reason in measurement_issues(video, target, probe).items():
            print(f'REMOVE incomplete stage cache {path.name}: {reason}', flush=True)
            path.unlink()
        clean_generated_outputs(video, target)
        row = validate_v5.analyze(video, output, VARIANT, hwaccel=hwaccel)
        for suffix in ('_flow.csv', '_telemetry.csv'):
            if not (target / (video.stem + suffix)).is_file():
                raise ValueError('missing measurement cache: ' + suffix)
        problems = measurement_issues(video, target, probe)
        if problems:
            raise ValueError('; '.join(path.name + ': ' + reason for path, reason in problems.items()))
        clips = verify_result(video, target, row)
        if source_stat(video) != stat:
            raise ValueError('original changed while processing')
        if algorithm_hashes() != hashes:
            raise ValueError('algorithm changed while processing')
        checkpoint.update(status='success', row=row, source_seconds=row['source_seconds'], clips=clips)
    except Exception as exc:
        checkpoint.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    checkpoint['finished_at'] = now()
    return checkpoint


def adopt_ready(stage, source, videos, hashes):
    """Explicitly adopt complete reports that predate checkpoint support."""
    output = stage / source.name
    report = load_json(output / 'v6_validation.json')
    if not report:
        return
    missing = [video for video in videos if not checkpoint_path(stage, source, video).exists()]
    if not missing:
        return
    if (report.get('variant') != VARIANT or report.get('algorithm_sha256') != hashes
            or Path(report.get('source', '')).resolve() != source
            or sorted(row.get('video', '') for row in report.get('videos', []))
            != sorted(video.name for video in videos)):
        print(f'ADOPT declined {source.name}: incomplete, stale, or mismatched report', flush=True)
        return
    rows = {row['video']: row for row in report['videos']}
    adopted = []
    try:
        for video in videos:
            row = {('v5' + key[2:] if key.startswith('v6_') else key): value
                   for key, value in rows[video.name].items()}
            target = output / (video.stem + '_cuts_v6')
            clips = verify_result(video, target, row)
            probe = probe_source(video)
            for suffix in ('_flow.csv', '_telemetry.csv'):
                if not (target / (video.stem + suffix)).is_file():
                    raise ValueError('missing measurement cache for ' + video.name)
            if measurement_issues(video, target, probe):
                raise ValueError('incomplete measurement cache for ' + video.name)
            adopted.append((video, dict(source=str(video), source_stat=source_stat(video),
                                       algorithm_sha256=hashes, status='success', row=row,
                                       source_seconds=row['source_seconds'], clips=clips,
                                       finished_at=now(), adopted_from_report=True)))
    except (ValueError, TypeError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        print(f'ADOPT declined {source.name}: {exc}', flush=True)
        return
    for video, checkpoint in adopted:
        path = checkpoint_path(stage, source, video)
        # Adoption is deliberately initial-only: an existing stale checkpoint
        # must rerun rather than gaining a new fingerprint for old exports.
        if not path.exists():
            atomic_json(path, checkpoint)
    print(f'ADOPT {source.name}: {len(missing)} verified originals', flush=True)


def discover_sources(values, stage):
    sources = [path.resolve() for path in values]
    if len(set(sources)) != len(sources) or len({path.name for path in sources}) != len(sources):
        raise ValueError('source directories and their date names must be unique')
    result = {}
    for source in sources:
        if not source.is_dir():
            raise FileNotFoundError(source)
        if stage == source or source in stage.parents or stage in source.parents:
            raise ValueError('stage must be separate from original source directories')
        videos = sorted(path.resolve() for path in source.iterdir()
                        if path.is_file() and path.suffix.lower() == '.mp4')
        if not videos:
            raise ValueError(f'no source MP4 files: {source}')
        if len({video.stem for video in videos}) != len(videos):
            raise ValueError('source stems must be unique: ' + str(source))
        result[source] = videos
    return result


def summary_document(stage, sources, hashes, checkpoints, published=()):
    dates = []
    errors = []
    for source, videos in sources.items():
        items = [checkpoints[str(video)] for video in videos]
        successes = [item for item in items if item['status'] == 'success']
        failures = [item for item in items if item['status'] == 'error']
        for item in failures:
            errors.append(dict(source=item['source'], error=item.get('error', 'unknown failure')))
        rows = [item['row'] for item in successes]
        done = source.name in published
        report = source / 'v6_validation.html' if done else stage / source.name / 'v6_validation.html'
        dates.append(dict(date=source.name, source=str(source), stage=str(stage / source.name),
                          videos=len(videos), success=len(successes), failed=len(failures),
                          pending=len(items) - len(successes) - len(failures),
                          source_seconds=sum(item.get('source_seconds', 0) for item in items),
                          v6_exported_seconds=sum(row['v5_exported_seconds'] for row in rows),
                          clips=sum(item.get('clips', 0) for item in successes),
                          published=done, report=str(report) if rows else None))
    return dict(variant=VARIANT, generated_at=now(), algorithm_sha256=hashes,
                sources=[str(source) for source in sources], dates=dates, errors=errors,
                totals={key: sum(date[key] for date in dates)
                        for key in ('videos', 'success', 'failed', 'pending', 'source_seconds',
                                    'v6_exported_seconds', 'clips')})


def save_summary(stage, sources, hashes, checkpoints, published=()):
    summary = summary_document(stage, sources, hashes, checkpoints, published)
    atomic_json(stage / 'batch_v6.json', summary)
    lines = []
    for date in summary['dates']:
        label = html.escape(date['date'])
        if date['report'] and Path(date['report']).exists():
            href = Path(date['report']).as_uri() if date['published'] else date['date'] + '/v6_validation.html'
            label = f'<a href="{html.escape(href, quote=True)}">{label}</a>'
        lines.append(f'<tr><td>{label}</td><td>{date["videos"]}</td>'
                     f'<td>{date["source_seconds"]:.2f}</td><td>{date["v6_exported_seconds"]:.2f}</td>'
                     f'<td>{date["clips"]}</td><td>{date["success"]}</td>'
                     f'<td>{date["failed"]}</td><td>{date["pending"]}</td>'
                     f'<td>{"已发布" if date["published"] else "工作区"}</td></tr>')
    errors = ''.join('<li>' + html.escape(error['source'] + ': ' + error['error']) + '</li>'
                     for error in summary['errors'])
    totals = summary['totals']
    page = '<!doctype html><meta charset="utf-8"><title>DJI v6 批处理</title>' + '''
<style>body{font:15px system-ui;margin:32px auto;max-width:1200px;padding:0 20px;color:#182333;background:#f8fafc}
table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border-bottom:1px solid #e0e5eb;text-align:left}p{line-height:1.6}</style>
<h1>DJI v6 批处理</h1>''' + (
        f'<p>原片 {totals["videos"]} 段；成功 {totals["success"]}，失败 {totals["failed"]}，'
        f'待处理 {totals["pending"]}。稳定切片 {totals["clips"]} 条，实际导出 '
        f'{totals["v6_exported_seconds"]:.2f} 秒。点击日期查看切片与扰动时间。</p>'
        '<p>当前算法与源文件状态一致且导出已验证的结果可续跑复用；原始视频没有改动。'
        '失败片段的完整日志位于对应切片目录的 analysis.log。</p>'
        '<p><a href="batch_v6.json">完整批处理数据</a></p>'
        '<table><thead><tr><th>日期</th><th>原片数</th><th>原片秒数</th><th>保留秒数</th>'
        '<th>切片数</th><th>成功</th><th>失败</th><th>待处理</th><th>位置</th></tr></thead><tbody>'
        + ''.join(lines) + '</tbody></table>' + ('<h2>失败列表</h2><ul>' + errors + '</ul>' if errors else ''))
    temporary = stage / 'index.html.tmp'
    temporary.write_text(page)
    temporary.replace(stage / 'index.html')
    return summary


def publish_preflight(stage, sources, hashes):
    """Check every date and target before the first external filesystem write."""
    batch = load_json(stage / 'batch_v6.json')
    if not batch or batch.get('sources') != [str(source) for source in sources]:
        raise ValueError('publication sources must exactly match the analyzed batch')
    if batch.get('algorithm_sha256') != hashes:
        raise ValueError('batch was produced by a different algorithm')
    if batch.get('errors') or any(batch['totals'].get(key) for key in ('failed', 'pending')):
        raise ValueError('cannot publish a batch with failed or pending originals')
    if batch['totals'].get('success') != sum(len(videos) for videos in sources.values()):
        raise ValueError('batch does not include all current originals')
    checkpoints = {}
    for source, videos in sources.items():
        output = stage / source.name
        report = load_json(output / 'v6_validation.json')
        if not report or Path(report.get('source', '')).resolve() != source:
            raise ValueError('report destination does not match original directory: ' + source.name)
        if report.get('variant') != VARIANT or report.get('algorithm_sha256') != hashes:
            raise ValueError('report algorithm/variant mismatch: ' + source.name)
        if sorted(row['video'] for row in report['videos']) != sorted(video.name for video in videos):
            raise ValueError('report originals do not match source directory: ' + source.name)
        expected_rows = []
        names = ['v6_validation.json', 'v6_validation.csv', 'v6_validation.html']
        for video in videos:
            checkpoint = load_json(checkpoint_path(stage, source, video))
            target = output / (video.stem + '_cuts_v6')
            if not resumable(checkpoint, video, target, hashes):
                raise ValueError('unverified/stale checkpoint: ' + str(video))
            checkpoints[str(video)] = checkpoint
            names.append(target.name)
            expected_rows.append({('v6' + key[2:] if key.startswith('v5_') else key): value
                                  for key, value in checkpoint['row'].items()})
        if report['videos'] != expected_rows:
            raise ValueError('report rows do not match verified checkpoints: ' + source.name)
        for name in names:
            if not (output / name).exists():
                raise FileNotFoundError(output / name)
            if (source / name).exists():
                raise FileExistsError(source / name)
    return checkpoints


def publish(stage, sources, hashes):
    checkpoints = publish_preflight(stage, sources, hashes)
    print(f'PUBLISH preflight passed: {len(sources)} dates, {len(checkpoints)} originals', flush=True)
    published = []
    try:
        for source in sources:
            subprocess.run([sys.executable, str(ROOT / 'scripts' / 'publish_v5.py'),
                            str(stage / source.name), str(source), '--variant', VARIANT],
                           cwd=ROOT, check=True)
            published.append(source.name)
            save_summary(stage, sources, hashes, checkpoints, published)
    finally:
        save_summary(stage, sources, hashes, checkpoints, published)


def run(stage, sources, hashes, workers, hwaccel='auto', adopt=False, min_free_gib=10):
    stage.mkdir(parents=True, exist_ok=True)
    checkpoints = {}
    work = []
    total = sum(len(videos) for videos in sources.values())
    for source, videos in sources.items():
        output = stage / source.name
        output.mkdir(parents=True, exist_ok=True)
        if adopt:
            adopt_ready(stage, source, videos, hashes)
        for video in videos:
            path = checkpoint_path(stage, source, video)
            checkpoint = load_json(path)
            if resumable(checkpoint, video, output / (video.stem + '_cuts_v6'), hashes):
                print(f'RESUME {source.name}/{video.name}: {checkpoint["clips"]} verified clips', flush=True)
            else:
                checkpoint = dict(source=str(video), source_stat=source_stat(video),
                                  algorithm_sha256=hashes, status='pending', queued_at=now(),
                                  source_seconds=(checkpoint or {}).get('source_seconds', 0))
                atomic_json(path, checkpoint)
                work.append((source, video, output, checkpoint['source_stat']))
            checkpoints[str(video)] = checkpoint
    save_summary(stage, sources, hashes, checkpoints)
    completed = total - len(work)
    def complete(source, video, checkpoint):
        nonlocal completed
        checkpoints[str(video)] = checkpoint
        atomic_json(checkpoint_path(stage, source, video), checkpoint)
        completed += 1
        free_gib = shutil.disk_usage(stage).free / 1024 ** 3
        print(f'DONE {completed}/{total} {source.name}/{video.name} '
              f'{checkpoint["status"]} clips={checkpoint.get("clips", 0)} free={free_gib:.1f} GiB', flush=True)
        if checkpoint['status'] == 'error':
            print('ERROR ' + checkpoint['error'], flush=True)
        save_summary(stage, sources, hashes, checkpoints)

    minimum_free = min_free_gib * 1024 ** 3
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        queue = list(work)
        while queue or futures:
            # Account for all in-flight writes, rather than allowing two jobs to
            # independently spend the same currently free filesystem bytes.
            available = shutil.disk_usage(stage).free - minimum_free - sum(
                reservation for _, _, reservation in futures.values())
            while queue and len(futures) < workers:
                selected = next((index for index, (_, _, _, stat) in enumerate(queue)
                                 if stat['size'] + 1024 ** 3 <= available), None)
                if selected is None:
                    break
                source, video, output, stat = queue.pop(selected)
                reservation = stat['size'] + 1024 ** 3
                checkpoint = checkpoints[str(video)]
                checkpoint.update(status='running', started_at=now())
                atomic_json(checkpoint_path(stage, source, video), checkpoint)
                future = pool.submit(analyze_one, video, output, hashes, stat, hwaccel)
                futures[future] = (source, video, reservation)
                available -= reservation
            if not futures:
                # No pending video fits. Persist every blocked original as an
                # error, so a later rerun can retry after disk space is freed.
                for source, video, _, stat in queue:
                    free = shutil.disk_usage(stage).free / 1024 ** 3
                    required = min_free_gib + stat['size'] / 1024 ** 3 + 1
                    checkpoint = checkpoints[str(video)]
                    if not checkpoint.get('source_seconds'):
                        try:
                            checkpoint['source_seconds'] = probe_source(video)['source_seconds']
                        except Exception:
                            pass
                    checkpoint.update(status='error', finished_at=now(), blocked='disk_space',
                                      error=f'insufficient disk space: free {free:.2f} GiB; '
                                            f'need {required:.2f} GiB including {min_free_gib:g} GiB reserve')
                    complete(source, video, checkpoint)
                queue.clear()
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                source, video, _ = futures.pop(future)
                checkpoint = future.result()
                complete(source, video, checkpoint)
    for source, videos in sources.items():
        rows = [checkpoints[str(video)]['row'] for video in videos
                if checkpoints[str(video)]['status'] == 'success']
        if rows:
            validate_v5.reports(rows, stage / source.name, source, VARIANT)
    summary = save_summary(stage, sources, hashes, checkpoints)
    print(json.dumps(summary['totals'], ensure_ascii=False), flush=True)
    for failure in summary['errors']:
        print('FAILED ' + failure['source'] + ': ' + failure['error'], flush=True)
    if algorithm_hashes() != hashes:
        print('FAILED algorithm changed during this batch; rerun before publishing', flush=True)
        return 1
    return 1 if summary['errors'] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='+', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--hwaccel', choices=['auto', 'none', 'videotoolbox', 'vaapi', 'cuda'], default='auto')
    parser.add_argument('--adopt-ready', action='store_true', help='adopt existing complete current-algorithm reports')
    parser.add_argument('--min-free-gib', type=float, default=10, help='keep this much space free, in addition to in-flight reservations')
    parser.add_argument('--publish', action='store_true', help='publish an entirely successful verified batch')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    if args.min_free_gib < 0:
        parser.error('--min-free-gib must be nonnegative')
    stage = args.out.resolve()
    try:
        sources = discover_sources(args.sources, stage)
        hashes = algorithm_hashes()
        if args.publish:
            publish(stage, sources, hashes)
            return 0
        return run(stage, sources, hashes, args.jobs, args.hwaccel, args.adopt_ready, args.min_free_gib)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'ERROR {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

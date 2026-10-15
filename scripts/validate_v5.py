"""Generate v5 cuts and a playable comparison against v4, without modifying v4.

Usage: .venv/bin/python scripts/validate_v5.py SOURCE_DIR --out OUTPUT_ROOT
Use SOURCE_DIR as OUTPUT_ROOT to write sibling <video>_cuts_v5 directories.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dronesift import cut

MASTERS = {'DJI_20261001123644_0014_D', 'DJI_20261001124446_0017_D'}
# Human-reviewed false negatives, in ORIGINAL-video time. These are acceptance
# checks only; selection never reads filenames or these annotations.
REVIEWED_DISTURBANCES = {
    'DJI_20261001112724_0005_D': [(67.95, 71.95)],
    'DJI_20261001123356_0013_D': [(11.3, 11.7)],
    'DJI_20261001123307_0012_D': [(33.7, 34.25)],
}


def duration(path):
    packets = cut.video_packets(str(path))
    return max(p['pts'] + p['duration'] for p in packets) - min(p['pts'] for p in packets)


def analyze(video, output, variant='v5', hwaccel='auto'):
    previous = video.parent / (video.stem + '_cuts_v4')
    # Reuse raw measurements from older exports without changing their clips.
    # The comparison baseline remains v4, regardless of the cache's version.
    cache = next((video.parent / (video.stem + suffix)
                  for suffix in ('_cuts_v4', '_cuts_v5', '_cuts_v3', '_cuts')
                  if all((video.parent / (video.stem + suffix) / (video.stem + ext)).is_file()
                         for ext in ('_telemetry.csv', '_flow.csv'))), None)
    target = output / (video.stem + '_cuts_' + variant)
    target.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MPLCONFIGDIR=str(output / '.mpl-cache'))
    cmd = [sys.executable, '-m', 'dronesift', str(video), '--out', str(target), '--mode', 'copy',
           '--hwaccel', hwaccel]
    if cache is not None: cmd += ['--cache-dir', str(cache)]
    with (target / 'analysis.log').open('w') as log:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    current = json.loads((target / (video.stem + '_segments.json')).read_text())
    manifest = json.loads((target / (video.stem + '_exports.json')).read_text())
    old_json = previous / (video.stem + '_segments.json')
    old = json.loads(old_json.read_text()) if old_json.exists() else {'segments': []}
    original_duration = duration(video)
    prior_clips = sorted(previous.glob('*.mp4')) if previous.exists() else []
    prior_exported = sum(duration(p) for p in prior_clips)
    selected = sum(s['duration'] for s in current['segments'])
    prior_selected = sum(s['duration'] for s in old['segments'])
    failures = []
    for c in manifest['clips']:
        if not c['verified']: failures.append(f"unverified {c['path']}")
        for event in current['disturbances']:
            if c['source_start'] < event['end'] - 1e-6 and c['source_end'] > event['start'] + 1e-6:
                failures.append(f"{c['path']} overlaps {event['reason']}")
        for start, end in REVIEWED_DISTURBANCES.get(video.stem, []):
            if c['source_start'] < end and c['source_end'] > start:
                failures.append(f"{c['path']} retains a human-reviewed disturbance {start:.2f}–{end:.2f}s")
    if manifest['skipped']: failures.append(f"{len(manifest['skipped'])} selected segments not exported")
    retention = manifest['total_duration'] / original_duration
    if video.stem in MASTERS and retention < 0.99:
        failures.append(f'master retention below 99%: {retention:.3%}')
    if failures: raise RuntimeError(video.name + ': ' + '; '.join(failures))
    row = dict(video=video.name, master=video.stem in MASTERS, source_seconds=original_duration,
               v4_selected_seconds=prior_selected, v4_exported_seconds=prior_exported,
               v4_retention=prior_exported / original_duration, v4_segments=len(old['segments']),
               v5_selected_seconds=selected, v5_exported_seconds=manifest['total_duration'],
               v5_retention=retention, v5_segments=len(current['segments']),
               exported_frames=sum(c['frames'] for c in manifest['clips']), verified=True,
               disturbances=current['disturbances'], v5_folder=target.name,
               v4_clips=[p.as_uri() for p in prior_clips],
               v5_clips=[target.name + '/' + c['path'] for c in manifest['clips']])
    print(f"{video.name}: v4 {row['v4_retention']:.1%} -> {variant} {retention:.1%}, "
          f"{row['v5_segments']} segments, verified", flush=True)
    return row


def reports(rows, output, source, variant='v5'):
    if variant != 'v5':
        rows = [{(variant + k[2:] if k.startswith('v5_') else k): v for k, v in row.items()} for row in rows]
    source_seconds = sum(r['source_seconds'] for r in rows)
    totals = dict(videos=len(rows), source_seconds=source_seconds,
                  v4_exported_seconds=sum(r['v4_exported_seconds'] for r in rows),
                  **{variant + '_exported_seconds': sum(r[variant + '_exported_seconds'] for r in rows)})
    files = ['segment.py', 'cut.py', '__main__.py']
    hashes = {name: hashlib.sha256((ROOT / 'dronesift' / name).read_bytes()).hexdigest() for name in files}
    report = dict(source=str(source), variant=variant, algorithm_sha256=hashes, totals=totals, videos=rows)
    (output / (variant + '_validation.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    keys = [k for k in rows[0] if k not in ('disturbances', 'v4_clips', variant + '_clips')]
    with (output / (variant + '_validation.csv')).open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader(); writer.writerows({k: r[k] for k in keys} for r in rows)
    body = []
    labels = dict(yaw_correction='转向速度修正', pitch_return='云台来回修正',
                  roll_correction='横滚扰动', camera_brake='急刹', shake_burst='振荡抖动', image_oscillation='画面振荡',
                  yaw_nudge='小幅横向修正')
    for row in rows:
        e = html.escape
        name = row['video']; folder = row[variant + '_folder']
        players = []
        for version in ('v4', variant):
            clips = row[version + '_clips']
            options = ''.join(f'<option value="{e(p, quote=True)}">{e(Path(p).name)}</option>' for p in clips)
            players.append(f'<div><b>{version}</b><br><select onchange="this.nextElementSibling.src=this.value">'
                           f'{options}</select><video controls muted preload="none" src="{e(clips[0], quote=True) if clips else ""}"></video>'
                           f'{"无满足最短时长的稳定片段" if not clips else ""}</div>')
        events = '; '.join(f"{x['start']:.2f}–{x['end']:.2f}s {labels[x['reason']]}" for x in row['disturbances']) or '未检出拨杆/云台扰动'
        body.append(f'<tr class="{"master" if row["master"] else ""}"><td>{e(name)}'
                    f'{" <b>大师模式对照</b>" if row["master"] else ""}</td><td>{row["source_seconds"]:.2f}s</td>'
                    f'<td>{row["v4_retention"]:.2%}</td><td>{row[variant + "_retention"]:.2%}</td>'
                    f'<td>{row["v4_segments"]} → {row[variant + "_segments"]}</td></tr>'
                    f'<tr><td colspan="5"><details><summary>播放比较 · 扰动时间 · 时间线</summary>'
                    f'<p>{e(events)}</p><div class="players">{"".join(players)}</div>'
                    f'<p><a href="{e(folder)}/{e(Path(name).stem)}_viz.html">交互轨迹报告</a> · '
                    f'<a href="{e(folder)}/{e(Path(name).stem)}_segments.json">精确范围</a> · '
                    f'<a href="{e(folder)}/{e(Path(name).stem)}_exports.json">实际导出校验</a></p>'
                    f'<img loading="lazy" src="{e(folder)}/{e(Path(name).stem)}_timeline.png"></details></td></tr>')
    page = f'<!doctype html><meta charset="utf-8"><title>DJI v4 / {variant} 切片验证</title>' + '''
<style>body{font:15px system-ui;margin:32px auto;max-width:1400px;padding:0 20px;background:#f8fafc;color:#182333}
table{width:100%;border-collapse:collapse;background:white}td,th{padding:10px;border-bottom:1px solid #e0e5eb;text-align:left}
.master{background:#e8f6ed}summary{cursor:pointer}.players{display:flex;gap:20px}.players>div{width:50%}
video,img{width:100%}select{display:block;width:100%;margin:8px 0}p{line-height:1.6}@media(max-width:800px){.players{display:block}.players>div{width:100%}}
</style><h1>DJI v4 / v5 切片验证</h1>''' + (
        f'<p>{len(rows)} 段原片，共 {source_seconds:.2f}s。{variant} 实际导出 {totals[variant + "_exported_seconds"]:.2f}s。'
        '保留率按导出视频实际时长计算；每个切片已校验帧数和时间戳，且没有覆盖已检出的扰动区间。</p>'
        '<p>大师模式是保留率对照。展开行可播放 v4/v5 并查看剔除时间；画面舒适度请结合播放审查。'
        '单向俯仰调整仍可能与自动快速俯仰难以区分，当前俯仰检测针对来回修正。</p>'
        f'<p><a href="{variant}_validation.csv">CSV</a> · <a href="{variant}_validation.json">完整验证数据</a></p>'
        f'<table><thead><tr><th>视频</th><th>原始时长</th><th>v4 实际保留率</th><th>{variant} 实际保留率</th><th>片段数</th></tr></thead>'
        '<tbody>' + ''.join(body) + '</tbody></table>')
    page = page.replace('<h1>DJI v4 / v5 切片验证</h1>', f'<h1>DJI v4 / {variant} 切片验证</h1>')
    (output / (variant + '_validation.html')).write_text(page)
    print(json.dumps(totals, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path); ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--variant', choices=['v5', 'v6'], default='v5')
    ap.add_argument('--hwaccel', choices=['auto', 'none', 'videotoolbox', 'vaapi', 'cuda'], default='auto')
    args = ap.parse_args(); source = args.source.resolve(); output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    videos = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() == '.mp4')
    if not videos: ap.error('no source MP4 files')
    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        jobs = {pool.submit(analyze, v, output, args.variant, args.hwaccel): v for v in videos}
        for job in as_completed(jobs): rows.append(job.result())
    reports(sorted(rows, key=lambda r: r['video']), output, source, args.variant)


if __name__ == '__main__': main()

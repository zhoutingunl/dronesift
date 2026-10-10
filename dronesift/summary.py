"""Summarise all <video>_cuts/<video>_segments.json under given directories: CSV + index.html.

usage: python -m dronesift.summary OUT_DIR DIR [DIR...]
"""
import csv, glob, html, json, os, sys

def collect(dirs):
    rows = []
    for d in dirs:
        for js in sorted(glob.glob(os.path.join(d, '*_cuts', '*_segments.json'))):
            try: data = json.load(open(js))
            except Exception as e: print('skip', js, e, file=sys.stderr); continue
            base = os.path.basename(js).replace('_segments.json', ''); cuts = os.path.dirname(js)
            segs = data['segments']; dur = data['frames'] / data['fps']; kept = sum(s['duration'] for s in segs)
            clips = sorted(glob.glob(os.path.join(cuts, base + '_[0-9][0-9]_*.mp4')))
            rows.append(dict(dir=os.path.basename(d), video=base, duration=dur, kept=kept, kept_pct=kept / dur if dur else 0,
                             n=len(segs), segs=segs, cuts=cuts, viz=os.path.join(cuts, base + '_viz.html'),
                             timeline=os.path.join(cuts, base + '_timeline.png'), clips=clips,
                             labels=' | '.join(f"{s['label']} {s['duration']:.0f}s" for s in segs)))
    return rows

def write_csv(rows, path):
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['dir', 'video', 'seg', 'start', 'end', 'duration', 'label', 'speed_mps', 'yaw_rate_dps', 'climb_mps', 'jitter_px_s', 'radius_m', 'clip'])
        for r in rows:
            for i, s in enumerate(r['segs'], 1):
                clip = next((c for c in r['clips'] if f"_{i:02d}_" in os.path.basename(c)), '')
                w.writerow([r['dir'], r['video'], i, f"{s['start']:.2f}", f"{s['end']:.2f}", f"{s['duration']:.2f}", s['label'],
                            f"{s['speed_mps']:.1f}", f"{s['yaw_rate_dps']:.1f}", f"{s['climb_mps']:.1f}", f"{s['jitter_px']:.1f}",
                            f"{s['radius_m']:.0f}" if s.get('radius_m') else '', os.path.basename(clip)])

def hms(t): return f"{int(t // 60):02d}:{t % 60:04.1f}"

def write_index(rows, path):
    out_dir = os.path.dirname(os.path.abspath(path)); rel = lambda p: os.path.relpath(p, out_dir)
    tot = sum(r['duration'] for r in rows); kept = sum(r['kept'] for r in rows)
    parts = [f"""<!doctype html><meta charset="utf-8"><title>dronesift 批处理汇总</title>
<style>body{{font:14px/1.5 -apple-system,"PingFang SC",sans-serif;margin:20px;color:#1c1b18;background:#f6f5f1}}
table{{border-collapse:collapse;width:100%;background:#fff}}th,td{{padding:6px 8px;border-bottom:1px solid #ddd;vertical-align:top;text-align:left}}
th{{background:#eef2f4;font-weight:500;position:sticky;top:0}}td.n{{font-family:Menlo,monospace;white-space:nowrap;text-align:right}}
.bar{{display:inline-block;height:8px;background:#1f6f8b;vertical-align:middle;border-radius:2px}}.lab{{color:#555;font-size:12px}}
.dir{{background:#e8e6df;font-weight:600}}a{{color:#1f6f8b}}img{{width:220px;display:block}}</style>
<h1>dronesift 批处理汇总</h1><p>{len(rows)} 个视频，共 {tot/60:.0f} 分钟，保留 {kept/60:.0f} 分钟（{kept/tot*100:.0f}%）。点视频名打开交互页，点"时间线"看静态图。</p>
<table><tr><th>视频</th><th>时长</th><th>保留</th><th>段数</th><th>运镜序列</th><th>时间线</th></tr>"""]
    cur = None
    for r in rows:
        if r['dir'] != cur:
            cur = r['dir']; parts.append(f'<tr><td class="dir" colspan="6">{cur}</td></tr>')
        parts.append(f"""<tr><td><a href="{html.escape(rel(r['viz']))}">{r['video']}</a><br><a class="lab" href="{html.escape(rel(r['cuts']))}">片段目录</a></td>
<td class="n">{hms(r['duration'])}</td><td class="n"><span class="bar" style="width:{r['kept_pct']*80:.0f}px"></span> {r['kept_pct']*100:.0f}%</td>
<td class="n">{r['n']}</td><td class="lab">{html.escape(r['labels'])}</td><td><a href="{html.escape(rel(r['timeline']))}"><img src="{html.escape(rel(r['timeline']))}" loading="lazy" alt=""></a></td></tr>""")
    parts.append('</table>'); open(path, 'w').write('\n'.join(parts))

if __name__ == '__main__':
    out, dirs = sys.argv[1], sys.argv[2:]
    rows = collect(dirs); os.makedirs(out, exist_ok=True)
    write_csv(rows, os.path.join(out, 'segments.csv')); write_index(rows, os.path.join(out, 'index.html'))
    print(f"{len(rows)} videos, {sum(r['n'] for r in rows)} segments -> {out}/index.html, segments.csv")

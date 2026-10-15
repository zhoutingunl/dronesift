"""Publish validation outputs; --replace archives the preceding v5 for comparison."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', type=Path); ap.add_argument('destination', type=Path)
    ap.add_argument('--replace', action='store_true', help='archive current v5 directories/report before replacement')
    ap.add_argument('--variant', choices=['v5', 'v6'], default='v5')
    args = ap.parse_args(); stage = args.stage.resolve(); destination = args.destination.resolve()
    variant = args.variant
    report = json.loads((stage / (variant + '_validation.json')).read_text())
    if Path(report['source']).resolve() != destination:
        raise ValueError('destination must be the validated original-video directory')
    names = [r[variant + '_folder'] for r in report['videos']] + [
        variant + '_validation.json', variant + '_validation.csv', variant + '_validation.html']
    for name in names:
        if Path(name).name != name: raise ValueError('output names must be direct children of destination')
    for name in names:
        if not (stage / name).exists(): raise FileNotFoundError(stage / name)
        if (destination / name).exists() and not args.replace: raise FileExistsError(destination / name)
    for row in report['videos']:
        manifest = json.loads((stage / row[variant + '_folder'] / (Path(row['video']).stem + '_exports.json')).read_text())
        if not row['verified'] or not all(c['verified'] for c in manifest['clips']):
            raise ValueError('cannot publish unverified clips')
        if not all((stage / row[variant + '_folder'] / c['path']).is_file() for c in manifest['clips']):
            raise FileNotFoundError('missing exported clip')
    existing = [name for name in names if (destination / name).exists()]
    archive = None
    if existing:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        archive = destination / ('_' + variant + '_history') / stamp
        archive.mkdir(parents=True, exist_ok=False)
    for name in names:
        target = destination / name
        if name in existing: shutil.move(str(target), str(archive / name))
        try:
            shutil.move(str(stage / name), str(target))
        except Exception:
            if name in existing and not target.exists(): shutil.move(str(archive / name), str(target))
            raise
        print('published', target, flush=True)
    if archive: print('previous v5 archived in', archive, flush=True)


if __name__ == '__main__': main()

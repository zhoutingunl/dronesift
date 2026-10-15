"""Add advisory playback-speed metadata to existing cuts without recutting.

Usage: python -m dronesift.assess_speed CUTS_DIR... --output-fps 30

Reads each original's cached telemetry/flow once. MP4/MOV/EDL files are never
opened or changed. A visualization is refreshed only when its thumbnail sprite
already exists, so migration never decodes video to regenerate thumbnails.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

from . import cut, segment, viz


def _read_json(path):
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def _rows(path):
    if not path.is_file():
        raise ValueError(f'missing cached measurements: {path.name}')
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'cached measurements are empty: {path.name}')
    return rows


def _catalog_segment(clip, segments):
    if 'requested_start' in clip and 'requested_end' in clip:
        matches = [s for s in segments
                   if abs(s['start'] - clip['requested_start']) <= 1e-6
                   and abs(s['end'] - clip['requested_end']) <= 1e-6]
    else:
        matches = [s for s in segments
                   if s['start'] <= clip['source_start'] + 1e-6
                   and s['end'] >= clip['source_end'] - 1e-6]
    if len(matches) != 1:
        raise ValueError(f'export has no unique catalog segment: {clip.get("path", "unnamed clip")}')
    s = matches[0]
    if clip['source_start'] < s['start'] - 1e-6 or clip['source_end'] > s['end'] + 1e-6:
        raise ValueError(f'export exceeds its catalog segment: {clip.get("path", "unnamed clip")}')
    return s


def _write_json(path, value):
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=1, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def assess_directory(cuts_dir, output_fps=30):
    """Prepare all metadata first, then atomically replace each metadata file."""
    cuts_dir = Path(cuts_dir)
    if not math.isfinite(output_fps) or output_fps <= 0:
        raise ValueError('output_fps must be finite and positive')
    if not cuts_dir.is_dir():
        raise ValueError(f'cuts directory does not exist: {cuts_dir}')
    catalogs = sorted(cuts_dir.glob('*_segments.json'))
    if len(catalogs) != 1:
        raise ValueError(f'expected exactly one *_segments.json, found {len(catalogs)} in {cuts_dir}')
    catalog = catalogs[0]
    base = catalog.name.removesuffix('_segments.json')
    document = _read_json(catalog)
    fps = float(document['fps'])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('catalog source fps must be finite and positive')
    segments = document['segments']
    telemetry_path = cuts_dir / (base + '_telemetry.csv')
    flow_path = cuts_dir / (base + '_flow.csv')
    telemetry, flow = _rows(telemetry_path), _rows(flow_path)
    thresholds = {**segment.DEFAULT_THR, **document.get('thresholds', {})}
    features, count = segment.features(telemetry, flow, fps, rev_amp=thresholds['motion_rev_amp'])
    if 'frames' in document and count != document['frames']:
        raise ValueError(f'flow frame count {count} differs from catalog {document["frames"]}')
    segment.annotate_playback_speeds(segments, features, fps, output_fps=output_fps, overwrite=True)

    export_path = cuts_dir / (base + '_exports.json')
    exports = _read_json(export_path) if export_path.exists() else None
    if exports is not None:
        for clip in exports['clips']:
            source = _catalog_segment(clip, segments)
            clip.update(cut.playback_speed_metadata(source, clip['source_start'], clip['source_end']))

    # Build inside the same filesystem before replacing anything. Copying the
    # cached sprite also guarantees viz.thumb_sprite takes its no-decode path.
    with tempfile.TemporaryDirectory(prefix='.assess-speed-', dir=cuts_dir) as temporary:
        work = Path(temporary)
        new_catalog = work / catalog.name
        _write_json(new_catalog, document)
        prepared = [(new_catalog, catalog)]
        if exports is not None:
            new_exports = work / export_path.name
            _write_json(new_exports, exports)
            prepared.append((new_exports, export_path))
        sprite = cuts_dir / (base + '_thumbs.jpg')
        if sprite.is_file():
            shutil.copyfile(sprite, work / sprite.name)
            keep, labels = segment.classify_frames(features, thresholds, fps)
            new_report = work / (base + '_viz.html')
            # Only the basename is used; source media need not be available.
            viz.build(base + '.MP4', str(telemetry_path), str(flow_path), str(new_catalog),
                      str(new_report), hwaccel='none', analysis=(features, keep, labels))
            prepared.append((new_report, cuts_dir / new_report.name))
        updated = [target.name for _, target in prepared]
        for ready, target in prepared:
            os.replace(ready, target)
    return dict(cuts_dir=str(cuts_dir), segments=len(segments), source_fps=fps,
                output_fps=output_fps, updated=updated,
                visualization_refreshed=sprite.is_file(), media_modified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cuts_dirs', nargs='+', type=Path)
    parser.add_argument('--output-fps', type=float, default=30)
    args = parser.parse_args(argv)
    if not math.isfinite(args.output_fps) or args.output_fps <= 0:
        parser.error('--output-fps must be finite and positive')
    failed = False
    for directory in args.cuts_dirs:
        try:
            result = assess_directory(directory, args.output_fps)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            print(f'{directory}: {error}', file=sys.stderr)
            failed = True
        else:
            print(json.dumps(result, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())

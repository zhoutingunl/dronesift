"""Exercise time-zone loading without the Unix system database Windows lacks."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_capture_timezone_without_system_iana_database():
    code = """
import json
import zoneinfo
from dronesift.montage import capture_datetime
zoneinfo.reset_tzpath([])
zoneinfo.ZoneInfo.clear_cache()
assert zoneinfo.TZPATH == ()
samples = [
    ('DJI_20261001091506_001_D.MP4', 'Asia/Shanghai'),
    ('DJI_20260701091506_001_D.MP4', 'America/New_York'),
    ('DJI_20260101091506_001_D.MP4', 'America/New_York'),
]
print(json.dumps([capture_datetime(name, timezone).utcoffset().total_seconds()
                  for name, timezone in samples]))
"""
    result = subprocess.run(
        [sys.executable, '-c', code], check=True, capture_output=True, text=True,
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ, PYTHONTZPATH=''),
    )
    assert json.loads(result.stdout) == [8 * 3600, -4 * 3600, -5 * 3600]

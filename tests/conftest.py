"""Explicit speed metadata for tests concerned with timing or visual reviews.

The signal-based estimator has its own synthetic-motion regressions. These
fixtures isolate the downstream contracts without asserting footage quality.
"""
import pytest


@pytest.fixture
def playback_profile():
    def create(start, end, maximum=2, output_fps=30):
        return dict(schema_version=1, method='motion_speed_v1',
                    source_start=start, source_end=end, output_fps=output_fps,
                    max_playback_speed=maximum, status='assessed',
                    assessed_speeds=[1, 1.25, 1.5, 1.75, 2],
                    assessments=[], limiting_reasons=[], capped_at_assessment_limit=False)
    return create

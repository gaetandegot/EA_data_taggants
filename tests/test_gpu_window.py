from datetime import datetime

import pytest

from scripts.gpu_window import is_open, status


@pytest.mark.parametrize("when, expected", [
    ("2026-10-08 14:34", False),  # Thursday afternoon
    ("2026-10-08 07:59", True),   # Thursday morning, still in last night's window
    ("2026-10-08 08:00", False),
    ("2026-10-08 17:59", False),
    ("2026-10-08 18:00", True),
    ("2026-10-09 07:59", True),   # Friday morning
    ("2026-10-09 12:00", False),  # Friday noon
    ("2026-10-09 18:00", True),   # weekend starts
    ("2026-10-10 12:00", True),   # Saturday noon
    ("2026-10-11 12:00", True),   # Sunday noon
    ("2026-10-12 07:59", True),   # Monday morning, weekend tail
    ("2026-10-12 08:00", False),
])
def test_is_open(when, expected):
    assert is_open(datetime.fromisoformat(when)) is expected


def test_status_open_until_end_of_night():
    state, deadline, left = status(datetime(2026, 10, 8, 20, 0))
    assert (state, deadline) == ('open', datetime(2026, 10, 9, 8, 0))
    assert left == 12 * 3600


def test_status_weekend_runs_to_monday_morning():
    state, deadline, _ = status(datetime(2026, 10, 9, 18, 30))
    assert (state, deadline) == ('open', datetime(2026, 10, 12, 8, 0))


def test_status_margin_and_closed():
    state, deadline, _ = status(datetime(2026, 10, 8, 20, 0), margin_min=10)
    assert deadline == datetime(2026, 10, 9, 7, 50)
    # inside the margin: no new work may start
    assert status(datetime(2026, 10, 9, 7, 55), margin_min=10)[0] == 'closed'
    state, nxt, left = status(datetime(2026, 10, 8, 14, 34))
    assert (state, nxt) == ('closed', datetime(2026, 10, 8, 18, 0))
    assert left == 3 * 3600 + 26 * 60

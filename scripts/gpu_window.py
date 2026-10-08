#!/usr/bin/env python3
"""GPU usage window of the CS department (local time).

GPUs may be used every day from 18:00 to 08:00, and all day on Saturday and Sunday
(i.e. continuously from Friday 18:00 to Monday 08:00).

    gpu_window.py status [--margin MIN]   -> "open <deadline_ts> <seconds_left>"
                                             or "closed <next_open_ts> <seconds_until>"
    gpu_window.py deadline [--margin MIN] -> unix time of the end of the current window
                                             minus the margin (exit 1 if closed)

Stdlib only so it is cheap to call from cron.
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

START_HOUR = 18  # first allowed hour on weekdays
END_HOUR = 8     # weekday window ends here (next morning)
STEP = timedelta(minutes=1)


def is_open(t):
    """Whether GPU use is allowed at local datetime `t`."""
    return t.weekday() >= 5 or t.hour >= START_HOUR or t.hour < END_HOUR


def _boundary(t, want_open):
    """First minute-aligned instant after `t` where is_open() == want_open."""
    cur = t.replace(second=0, microsecond=0) + STEP
    for _ in range(8 * 24 * 60):  # the pattern repeats weekly
        if is_open(cur) == want_open:
            return cur
        cur += STEP
    raise RuntimeError("no window boundary found")


def status(now=None, margin_min=0):
    """('open', deadline, seconds_left) or ('closed', next_open, seconds_until).

    While open, `deadline` is the end of the window minus the margin; when that
    margin has already been entered the window counts as closed for new work."""
    # GPU_WINDOW_NOW (ISO local time) fakes the clock; for testing only
    now = now or (datetime.fromisoformat(os.environ['GPU_WINDOW_NOW'])
                  if os.environ.get('GPU_WINDOW_NOW') else datetime.now())
    if is_open(now):
        deadline = _boundary(now, False) - timedelta(minutes=margin_min)
        if now < deadline:
            return 'open', deadline, (deadline - now).total_seconds()
        return 'closed', _boundary(_boundary(now, False), True), \
            (_boundary(_boundary(now, False), True) - now).total_seconds()
    nxt = _boundary(now, True)
    return 'closed', nxt, (nxt - now).total_seconds()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['status', 'deadline'])
    parser.add_argument('--margin', type=float, default=0.0,
                        help='minutes to stop before the real end of the window')
    args = parser.parse_args()
    state, when, seconds = status(margin_min=args.margin)
    if args.command == 'deadline':
        if state != 'open':
            sys.exit(1)
        print(int(when.timestamp()))
    else:
        print(state, int(when.timestamp()), int(seconds))


if __name__ == '__main__':
    main()

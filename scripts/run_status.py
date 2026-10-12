#!/usr/bin/env python3
"""Machine-readable state of the run driven by a supervisor.env (printed as JSON on stdout).

    run_status.py <supervisor.env> [--rearmed]
    run_status.py --none            -> {"active": false, ...} (no active run on this host)

Called by remote_tick.sh, whose stdout is consumed by the laptop dashboard, so the schema
(`schema` field) must only change in backward-compatible ways. Stdlib only (system python3).
"""
import csv
import glob
import json
import os
import re
import shlex
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCHEMA = 1
TAG = '# data-taggants-supervisor'
STAGES = ['pretrain', 'sign', 'validate', 'detect']
sys.path.insert(0, str(Path(__file__).resolve().parent))


def iso(ts):
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec='seconds') if ts else None


def mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def read(path, default=None):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def parse_env(path):
    """KEY=value / export KEY=value lines written by launch_supervised.sh (printf %q quoting)."""
    env = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r'^(?:export\s+)?([A-Z_][A-Z0-9_]*)=(.*)$', line.strip())
        if m:
            vals = shlex.split(m.group(2)) if m.group(2) else ['']
            env[m.group(1)] = ' '.join(vals)
    return env


def last_progress_line(log):
    """Last tqdm-like line of a log (tqdm rewrites lines with \\r)."""
    try:
        with open(log, 'rb') as f:
            f.seek(max(0, os.path.getsize(log) - 20000))
            tail = f.read().decode('utf-8', 'replace')
    except OSError:
        return None
    lines = [l.strip() for l in re.split(r'[\r\n]', tail) if re.search(r'\d+%\||\d+/\d+ \[', l)]
    return lines[-1][:200] if lines else None


def read_csv(path):
    try:
        with open(path, newline='') as f:
            rows = list(csv.reader(f, delimiter='\t'))
    except OSError:
        return None
    if not rows:
        return None

    def num(x):
        try:
            return float(x)
        except ValueError:
            return x
    return {'columns': rows[0], 'rows': [[num(x) for x in r] for r in rows[1:] if r]}


def epoch_stage(run_dir, log, epochs_total):
    csv_ = read_csv(run_dir / 'train.csv') if run_dir else None
    done = int(csv_['rows'][-1][0]) + 1 if csv_ and csv_['rows'] else 0
    ep_s = read(run_dir / 'epoch_seconds.txt') if run_dir else None
    ep_s = float(ep_s) if ep_s else None
    return {
        'epochs_done': done, 'epochs_total': epochs_total, 'epoch_seconds': ep_s,
        'checkpoint_at': iso(mtime(run_dir / 'checkpoint.pth')) if run_dir else None,
        'exception_marker_at': iso(mtime(run_dir / 'exception')) if run_dir else None,
        'live': last_progress_line(log),
        'remaining_seconds': (epochs_total - done) * ep_s if ep_s and epochs_total else None,
    }, csv_


def main():
    host = socket.gethostname().split('.')[0]
    now = time.time()
    out = {'schema': SCHEMA, 'host': host, 'checked_at': iso(now), 'active': False}
    if len(sys.argv) < 2 or sys.argv[1] == '--none':
        print(json.dumps(out))
        return
    env = parse_env(sys.argv[1])
    root = Path(env['OUTPUT_ROOT'])
    state = root / 'state'
    cfg_text = read(root / 'run_config.txt', '')

    def flag(name, cast=str):
        m = re.search(r'--%s (\S+)' % name, cfg_text)
        return cast(m.group(1)) if m else None
    m = re.search(r'n_keys=(\d+)', cfg_text)
    n_keys = int(m.group(1)) if m else None
    epochs = flag('epochs', int)
    stages = env.get('STAGES', ' '.join(STAGES)).split()

    out.update(active=True, rearmed_cron='--rearmed' in sys.argv, output_root=str(root),
               run_name=root.name)
    out['config'] = {'model': flag('model'), 'opt': flag('opt'), 'lr': flag('lr'),
                     'epochs': epochs, 'n_keys': n_keys, 'stages': stages,
                     'workers': env.get('WORKERS'), 'reserve_gpu_mem': env.get('RESERVE_GPU_MEM'),
                     'run_config': cfg_text}

    # health
    try:
        cron = subprocess.run(['crontab', '-l'], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        cron = ''
    pid = read(state / 'pipeline.pid')
    running, uptime = False, None
    if pid and 'timed_pipeline' in (read('/proc/%s/cmdline' % pid, '') or ''):
        running = True
        uptime = now - os.stat('/proc/%s' % pid).st_mtime
    hb = read(state / 'heartbeat')
    gpu_mem = None
    try:
        r = subprocess.run(['nvidia-smi', '--query-compute-apps=used_memory',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
        gpu_mem = sum(int(x) for x in r.stdout.split() if x.isdigit())
    except Exception:
        pass
    try:
        import gpu_window
        ws, wts, wsec = gpu_window.status()
        window = {'state': ws, 'until': wts.astimezone().isoformat(timespec='seconds'),
                  'seconds': int(wsec)}
    except Exception as e:
        window = {'state': 'unknown', 'error': str(e)}
    failures = int(read(state / 'failures', '0') or 0)
    max_failures = int(env.get('MAX_FAILURES', 8))
    done = {s: (state / ('%s.done' % s)).exists() for s in stages}
    current = next((s for s in stages if not done[s]), None)
    out['health'] = {
        'cron_installed': TAG in cron,
        'heartbeat_at': hb,
        'heartbeat_age_seconds': int(now - datetime.fromisoformat(hb).timestamp()) if hb else None,
        'pipeline_running': running, 'pipeline_pid': int(pid) if running else None,
        'pipeline_uptime_seconds': int(uptime) if uptime else None,
        'failures': failures, 'max_failures': max_failures, 'gave_up': failures >= max_failures,
        'last_msg': read(state / 'last_msg'),
        'gpu_window': window, 'gpu_mem_used_mb': gpu_mem,
    }

    # progress
    pre_dir = next((Path(p) for p in glob.glob(str(root / 'pretrained' / '*' / 'seed0'))), None)
    val_dir = root / 'validation' if (root / 'validation').exists() else None
    pre, pre_csv = epoch_stage(pre_dir, root / 'pretrain.log', epochs)
    val, val_csv = epoch_stage(val_dir, root / 'validate.log', epochs)
    if val['remaining_seconds'] is None and pre['epoch_seconds'] and epochs:
        val['remaining_seconds'] = (epochs - val['epochs_done']) * pre['epoch_seconds']
        val['remaining_is_estimate_from_pretrain'] = True
    key_done = sorted(t for t in (mtime(p) for p in glob.glob(str(root / 'poisons' / 'key_*' / 'sign_done'))) if t)
    gaps = [b - a for a, b in zip(key_done, key_done[1:]) if b - a < 4 * 3600]
    key_s = statistics.median(gaps) if gaps else None
    sign = {'keys_done': len(key_done), 'n_keys': n_keys, 'seconds_per_key': key_s,
            'pbatch': read(state / 'pbatch'), 'pbatch_confirmed': read(state / 'pbatch_confirmed'),
            'live': last_progress_line(root / 'sign.log'),
            'remaining_seconds': (n_keys - len(key_done)) * key_s if key_s and n_keys else None}
    detect_tail = read(root / 'detect.log', '')
    detect = {'result_tail': '\n'.join(detect_tail.splitlines()[-15:]) if done.get('detect') else None,
              'remaining_seconds': 0 if done.get('detect') else 600}
    progress = {'pretrain': pre, 'sign': sign, 'validate': val, 'detect': detect}
    for s in progress:
        progress[s]['done'] = done.get(s, False)
        progress[s]['done_at'] = iso(mtime(state / ('%s.done' % s)))
        if progress[s]['done']:
            progress[s]['remaining_seconds'] = 0
    rem = [progress[s]['remaining_seconds'] for s in stages if s in progress]
    out['current_stage'] = current
    out['all_done'] = current is None
    out['progress'] = progress
    out['remaining_gpu_seconds'] = None if any(r is None for r in rem) else sum(rem)
    out['train_csv'] = {'pretrain': pre_csv, 'validate': val_csv}

    # timing attempts
    attempts, cur = [], None
    for line in read(root / 'timing.txt', '').splitlines():
        if line.startswith('start: '):
            cur = {'start': line[7:], 'end': None, 'stages': []}
            attempts.append(cur)
        elif cur is not None and line.startswith('end: '):
            cur['end'] = line[5:]
        elif cur is not None:
            m = re.match(r'^(\w+): (\d+)s \(exit (\d+)\)', line)
            if m:
                cur['stages'].append({'stage': m.group(1), 'seconds': int(m.group(2)), 'exit': int(m.group(3))})
    out['attempts'] = attempts

    # events: supervisor.log lines + file markers
    events = []
    for line in read(root / 'supervisor.log', '').splitlines()[-300:]:
        ts, _, msg = line.partition(' ')
        kind = ('launch' if msg.startswith('launching') else 'exit' if 'pipeline exited' in msg
                else 'killed' if 'killed externally' in msg else 'rearm' if 're-installed' in msg
                else 'window_kill' if msg.startswith('window closed') else 'gave_up' if msg.startswith('giving up')
                else 'info')
        events.append({'at': ts, 'kind': kind, 'msg': msg})
    for s in stages:
        t = mtime(state / ('%s.done' % s))
        if t:
            events.append({'at': iso(t), 'kind': 'stage_done', 'msg': '%s done' % s})
    for i, t in enumerate(key_done):
        events.append({'at': iso(t), 'kind': 'key_signed', 'msg': 'key %d/%s signed' % (i + 1, n_keys)})
    for name, d in (('pretrain', pre_dir), ('validate', val_dir)):
        t = mtime(d / 'exception') if d else None
        if t:
            events.append({'at': iso(t), 'kind': 'exception_marker', 'msg': '%s exception marker (last one)' % name})
    events.sort(key=lambda e: e['at'] or '')
    out['events'] = events
    print(json.dumps(out))


if __name__ == '__main__':
    main()

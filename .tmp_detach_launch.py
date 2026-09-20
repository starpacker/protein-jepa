#!/usr/bin/env python3
"""Fully-detached launcher for reviewer-experiment subprocesses.

Usage:
  python3 .tmp_detach_launch.py <log_path> -- <cmd> [args...]

Double-forks + setsid so the child survives the parent shell being killed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

if len(sys.argv) < 4 or sys.argv[2] != "--":
    print(__doc__)
    raise SystemExit(2)

log_path = Path(sys.argv[1])
cmd = sys.argv[3:]

log_path.parent.mkdir(parents=True, exist_ok=True)


def _fork_child(log_path: Path, cmd: list[str]) -> None:
    pid = os.fork()
    if pid > 0:
        print(f"detached pid={pid} log={log_path}")
        return
    os.setsid()
    pid2 = os.fork()
    if pid2 > 0:
        os._exit(0)
    with open(log_path, "w") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        devnull = os.open(os.devnull, os.O_RDONLY)
        os.dup2(devnull, 0)
        os.execvp(cmd[0], cmd)


_fork_child(log_path, cmd)

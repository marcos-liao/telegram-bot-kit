#!/usr/bin/env python3
"""
Memory-retention pruner for Telegram Bot Kit.

Deletes vector memories older than MEMORY_RETENTION_DAYS from the shared SQLite
DB (table `memories`). Cheap no-op when nothing is old enough -> safe to run
from cron daily.

Run:  ./venv/bin/python memory_prune.py
"""
import os
import time
import logging
import sqlite3
from contextlib import closing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE_DIR, "memory.db"))
MEMORY_RETENTION_DAYS = int(os.environ.get("MEMORY_RETENTION_DAYS", "180"))  # 0 = keep forever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("memory_prune")


def prune_memories_by_age(days=MEMORY_RETENTION_DAYS):
    """Delete memories older than `days`. Returns number removed (0 if disabled)."""
    if days <= 0:
        return 0
    cutoff = time.time() - days * 86400
    with closing(sqlite3.connect(DB_PATH, timeout=30)) as con:
        cur = con.execute("DELETE FROM memories WHERE ts < ?", (cutoff,))
        con.commit()
        return cur.rowcount


if __name__ == "__main__":
    n = prune_memories_by_age()
    if n:
        log.info("pruned %d memories older than %d days", n, MEMORY_RETENTION_DAYS)
    else:
        log.info("nothing to prune (retention=%d days)", MEMORY_RETENTION_DAYS)

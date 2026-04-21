from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

from service.worker_healthcheck import main as run_healthcheck


def run_worker() -> None:
    try:
        run_healthcheck()
    except Exception as exc:
        print(f"[healthcheck] FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    db_path = Path(os.getenv("UVR_JOB_DB_PATH", "data/separation_jobs.sqlite3"))
    db_path.parent.mkdir(parents=True, exist_ok=True)

    from service.api import DurableJobStore, SeparationWorker

    store = DurableJobStore(db_path=db_path)
    worker = SeparationWorker(store=store)
    worker.start()

    running = True

    def _shutdown(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    while running:
        time.sleep(0.5)

    worker.stop()


if __name__ == "__main__":
    run_worker()

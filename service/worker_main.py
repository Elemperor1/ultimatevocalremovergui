from __future__ import annotations

import signal
import time

from service.api import DurableJobStore, SeparationWorker
from service.worker_healthcheck import main as run_healthcheck


def run_worker() -> None:
    run_healthcheck()
    store = DurableJobStore(db_path="/tmp/separation_jobs.sqlite3")
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

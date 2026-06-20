from __future__ import annotations
import os, queue, threading
from datetime import datetime, timezone
from pathlib import Path
from . import optimizer, report
from .models import Job, JobStatus

DATA_DIR = Path(os.environ.get("NIGHTSHIFT_DATA_DIR", "data"))
_queue: "queue.Queue[str]" = queue.Queue()
_jobs: dict[str, Job] = {}
_lock = threading.Lock()
_worker_started = False

def job_dir(job_id: str) -> Path: return DATA_DIR / job_id

def save(job: Job) -> None:
    d = job_dir(job.id); d.mkdir(parents=True, exist_ok=True)
    (d / "job.json").write_text(job.model_dump_json(indent=2))

def get(job_id: str) -> Job | None:
    with _lock:
        if job_id in _jobs: return _jobs[job_id]
    path = job_dir(job_id) / "job.json"
    if path.exists():
        job = Job.model_validate_json(path.read_text())
        with _lock: _jobs[job_id] = job
        return job
    return None

def list_jobs() -> list[Job]:
    if not DATA_DIR.exists(): return []
    return [j for d in sorted(DATA_DIR.iterdir()) if (j := get(d.name))]

def submit(job: Job) -> None:
    with _lock: _jobs[job.id] = job
    save(job); _ensure_worker(); _queue.put(job.id)

def _ensure_worker() -> None:
    global _worker_started
    with _lock:
        if _worker_started: return
        _worker_started = True
    threading.Thread(target=_worker_loop, name="nightshift-worker", daemon=True).start()

def _worker_loop() -> None:
    while True:
        job_id = _queue.get()
        if job := get(job_id): _run_job(job)

def _run_job(job: Job) -> None:
    job.status = JobStatus.RUNNING
    job.started_at = datetime.now(timezone.utc)
    save(job)
    def on_result(r):
        job.results.append(r)
        if len(job.results) % 5 == 0: save(job)
    try:
        optimizer.run_optimization(job, on_result=on_result)
        kept = [r.metric_value for r in job.results if r.status == "keep"]
        if kept:
            job.baseline_metric = kept[0]
            from .models import MetricDirection
            lower = job.config.metric_direction == MetricDirection.LOWER_IS_BETTER
            job.best_metric = min(kept) if lower else max(kept)
        job.status = JobStatus.COMPLETED
    except Exception as e:
        job.status = JobStatus.FAILED; job.error = repr(e)
    finally:
        job.finished_at = datetime.now(timezone.utc)
        d = job_dir(job.id)
        (d / "results.tsv").write_text(report.results_tsv(job))
        (d / "report.md").write_text(report.report_markdown(job))
        save(job)

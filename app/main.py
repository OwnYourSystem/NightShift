from __future__ import annotations
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from . import jobs
from .models import Job, MetricDirection, OptimizationConfig

app = FastAPI(title="NightShift", description="Overnight ML model optimization.")
STATIC_DIR = Path(__file__).parent.parent / "static"
MAX_UPLOAD_BYTES = 2 * 1024**3
ALLOWED_EXTENSIONS = {".pt", ".pth", ".bin", ".safetensors", ".onnx", ".gguf", ".zip", ".tar", ".gz"}

@app.get("/", response_class=HTMLResponse)
def index(): return (STATIC_DIR / "index.html").read_text()

@app.post("/api/jobs")
async def create_job(
    customer_email: str = Form(...), metric_name: str = Form("val_bpb"),
    metric_direction: str = Form("lower"), max_experiments: int = Form(100),
    notes: str = Form(""), model_file: UploadFile = File(...),
):
    suffix = Path(model_file.filename or "model.bin").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type {suffix!r}.")
    config = OptimizationConfig(metric_name=metric_name,
        metric_direction=MetricDirection(metric_direction),
        max_experiments=max_experiments, notes=notes)
    job = Job(customer_email=customer_email,
        model_filename=model_file.filename or "model.bin", config=config)
    dest_dir = jobs.job_dir(job.id); dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"model{suffix}"; written = 0
    with open(dest, "wb") as f:
        while chunk := await model_file.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "Model exceeds 2 GB upload limit.")
            f.write(chunk)
    jobs.submit(job)
    return {"job_id": job.id, "status": job.status}

@app.get("/api/jobs")
def list_all_jobs():
    return [{"job_id": j.id, "status": j.status, "model": j.model_filename,
             "experiments_done": len(j.results), "best_metric": j.best_metric,
             "created_at": j.created_at} for j in jobs.list_jobs()]

@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    if not (job := jobs.get(job_id)): raise HTTPException(404, "Job not found")
    kept = [r for r in job.results if r.status == "keep"]
    return {"job_id": job.id, "status": job.status, "experiments_done": len(job.results),
            "experiments_total": job.config.max_experiments,
            "baseline_metric": kept[0].metric_value if kept else None,
            "best_metric": job.best_metric, "error": job.error}

@app.get("/api/jobs/{job_id}/results.tsv", response_class=PlainTextResponse)
def job_results_tsv(job_id: str):
    path = jobs.job_dir(job_id) / "results.tsv"
    if not path.exists(): raise HTTPException(404, "Results not ready")
    return path.read_text()

@app.get("/api/jobs/{job_id}/report.md")
def job_report(job_id: str):
    path = jobs.job_dir(job_id) / "report.md"
    if not path.exists():
        if not (job := jobs.get(job_id)): raise HTTPException(404, "Job not found")
        raise HTTPException(409, f"Report not ready (job is {job.status.value})")
    return FileResponse(path, media_type="text/markdown", filename=f"nightshift-{job_id}.md")

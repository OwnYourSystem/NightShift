from __future__ import annotations
import os, random, re, subprocess, time
from pathlib import Path
from .models import ExperimentResult, Job, MetricDirection

HARNESS_DIR = os.environ.get("NIGHTSHIFT_HARNESS_DIR", "")

EXPERIMENT_IDEAS = [
    "baseline", "increase learning rate 25%", "decrease learning rate 25%",
    "longer LR warmup (300 steps)", "no LR warmup", "wider model, fewer layers",
    "narrower model, more layers", "increase batch size 2x", "decrease batch size 2x",
    "switch GeLU -> ReLU^2 activation", "untie input/output embeddings",
    "zero-init output projection", "rotary embedding base 100k", "QK-norm",
    "remove bias terms", "label smoothing 0.05", "EMA of weights for eval",
    "cosine LR schedule", "linear LR decay to 0", "gradient clipping 0.5",
]

def harness_available() -> bool:
    return bool(HARNESS_DIR) and (Path(HARNESS_DIR) / "train.py").exists()

def run_optimization(job: Job, on_result=None) -> list[ExperimentResult]:
    if harness_available():
        return _run_real(job, on_result)
    return _run_simulated(job, on_result)

METRIC_RE = re.compile(r"^val_bpb:\s*([0-9.]+)", re.MULTILINE)
VRAM_RE   = re.compile(r"^peak_vram_mb:\s*([0-9.]+)", re.MULTILINE)

def _run_real(job: Job, on_result=None) -> list[ExperimentResult]:
    results, workdir = [], Path(HARNESS_DIR)
    timeout_s = job.config.time_budget_minutes_per_run * 60 * 2
    for i in range(job.config.max_experiments):
        desc = EXPERIMENT_IDEAS[i % len(EXPERIMENT_IDEAS)]
        log_path = workdir / "run.log"
        try:
            with open(log_path, "w") as log:
                subprocess.run(["uv", "run", "train.py"], cwd=workdir,
                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s, check=True)
            text = log_path.read_text()
            mm, vm = METRIC_RE.search(text), VRAM_RE.search(text)
            if not mm: raise RuntimeError("metric not found")
            metric = float(mm.group(1))
            mem = float(vm.group(1)) / 1024 if vm else 0.0
            result = ExperimentResult(commit=f"run{i:04d}", metric_value=metric,
                memory_gb=round(mem, 1), status=_keep_or_discard(job, results, metric), description=desc)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, RuntimeError) as e:
            result = ExperimentResult(commit=f"run{i:04d}", metric_value=0.0,
                memory_gb=0.0, status="crash", description=f"{desc} ({type(e).__name__})")
        results.append(result)
        if on_result: on_result(result)
    return results

def _run_simulated(job: Job, on_result=None) -> list[ExperimentResult]:
    rng = random.Random(job.id)
    lower = job.config.metric_direction == MetricDirection.LOWER_IS_BETTER
    sign = 1.0 if lower else -1.0
    best = 1.05 + rng.uniform(-0.02, 0.02)
    memory = 40.0 + rng.uniform(-5, 5)
    results = []
    for i in range(job.config.max_experiments):
        desc = EXPERIMENT_IDEAS[i % len(EXPERIMENT_IDEAS)]
        time.sleep(0.05)
        if i == 0:
            metric, status = best, "keep"
        elif rng.random() < 0.05:
            results.append(ExperimentResult(commit=f"sim{i:04d}", metric_value=0.0,
                memory_gb=0.0, status="crash", description=f"{desc} (OOM)"))
            if on_result: on_result(results[-1])
            continue
        else:
            decay = 1.0 / (1.0 + i * 0.15)
            delta = rng.uniform(-0.004, 0.012) * decay
            metric = round(best - sign*delta if rng.random() < 0.25 else best + sign*abs(delta), 6)
            improved = (metric < best) if lower else (metric > best)
            status = "keep" if improved else "discard"
            if improved:
                best = metric
                memory += rng.uniform(-0.5, 0.8)
        results.append(ExperimentResult(commit=f"sim{i:04d}", metric_value=round(metric, 6),
            memory_gb=round(max(memory, 1.0), 1), status=status, description=desc))
        if on_result: on_result(results[-1])
    return results

def _keep_or_discard(job: Job, results, metric: float) -> str:
    kept = [r.metric_value for r in results if r.status == "keep"]
    if not kept: return "keep"
    best = min(kept) if job.config.metric_direction == MetricDirection.LOWER_IS_BETTER else max(kept)
    return ("keep" if metric < best else "discard") if job.config.metric_direction == MetricDirection.LOWER_IS_BETTER else ("keep" if metric > best else "discard")

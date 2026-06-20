from __future__ import annotations
from .models import Job, JobStatus, MetricDirection

def results_tsv(job: Job) -> str:
    lines = ["commit\tmetric\tmemory_gb\tstatus\tdescription"]
    for r in job.results:
        lines.append(f"{r.commit}\t{r.metric_value:.6f}\t{r.memory_gb:.1f}\t{r.status}\t{r.description}")
    return "\n".join(lines) + "\n"

def report_markdown(job: Job) -> str:
    cfg = job.config
    lower = cfg.metric_direction == MetricDirection.LOWER_IS_BETTER
    completed = [r for r in job.results if r.status != "crash"]
    crashes   = [r for r in job.results if r.status == "crash"]
    kept      = [r for r in job.results if r.status == "keep"]
    baseline  = completed[0] if completed else None
    best = (min(kept, key=lambda r: r.metric_value) if lower else max(kept, key=lambda r: r.metric_value)) if kept else None
    lines = [
        f"# NightShift Optimization Report — job `{job.id}`", "",
        f"- **Customer**: {job.customer_email}",
        f"- **Model**: `{job.model_filename}`",
        f"- **Metric**: `{cfg.metric_name}` ({'lower' if lower else 'higher'} is better)",
        f"- **Experiments run**: {len(job.results)} ({len(crashes)} crashed)",
        f"- **Status**: {job.status.value}", "",
    ]
    if baseline and best:
        delta = best.metric_value - baseline.metric_value
        pct = (delta / baseline.metric_value * 100) if baseline.metric_value else 0.0
        ok = (delta < 0) if lower else (delta > 0)
        lines += ["## Headline", "",
            f"| | {cfg.metric_name} | memory (GB) |", "|---|---|---|",
            f"| Baseline (`{baseline.commit}`) | {baseline.metric_value:.6f} | {baseline.memory_gb:.1f} |",
            f"| **Best (`{best.commit}`)** | **{best.metric_value:.6f}** | {best.memory_gb:.1f} |", "",
            f"**Change: {delta:+.6f} ({pct:+.2f}%)** — " + ("improvement kept." if ok else "no improvement found."), "",
            f"Winning experiment: *{best.description}*", ""]
    if kept:
        lines += ["## Kept improvements", ""]
        for r in kept: lines.append(f"- `{r.commit}` {r.metric_value:.6f} — {r.description}")
        lines.append("")
    lines += ["## Full log", "", "See `results.tsv` for the complete experiment-by-experiment record.", ""]
    if job.status == JobStatus.FAILED and job.error:
        lines += ["## Error", "", f"```\n{job.error}\n```", ""]
    return "\n".join(lines)

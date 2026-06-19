# ☾ NightShift

**Overnight ML model optimization as a service.** Upload a model and a quality metric in the evening; an autonomous experiment loop runs ~100 optimization experiments overnight; you wake up to a report and a better model — smaller, faster, cheaper, or more accurate.

## Status

Phase 1 / Week 1 MVP: the full pipeline works end to end —

- **Upload** a model file + metric config via web form or API
- **Optimize** via a background worker running the experiment loop
  - *real mode*: drives an autoresearch harness (`uv run train.py`) when `NIGHTSHIFT_HARNESS_DIR` points at one
  - *simulate mode* (default): generates a realistic experiment trajectory so the whole product can be demoed with no GPU
- **Deliver** `results.tsv` (full experiment log) + `report.md` (headline numbers, kept improvements)

## Quick start

```bash
uv sync
uv run uvicorn app.main:app --reload
# open http://localhost:8000

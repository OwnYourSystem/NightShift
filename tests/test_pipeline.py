"""End-to-end pipeline tests for NightShift."""
from __future__ import annotations
import importlib
import io
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient wired to a clean temp data directory."""
    monkeypatch.setenv("NIGHTSHIFT_DATA_DIR", str(tmp_path))

    # Force reload so DATA_DIR picks up the monkeypatched env var
    import app.jobs as jobs_module
    jobs_module.DATA_DIR = tmp_path
    jobs_module._jobs.clear()
    jobs_module._worker_started = False
    import queue
    jobs_module._queue = queue.Queue()

    from app.main import app
    return TestClient(app)


def _fake_model_bytes() -> bytes:
    # Minimal .pt-looking payload — the server just stores the bytes
    return b"\x80\x02}q\x00."  # tiny pickle


def test_full_pipeline(client):
    """Upload a model, wait for completion, check artifacts."""
    payload = _fake_model_bytes()

    resp = client.post(
        "/api/jobs",
        data={
            "customer_email": "test@example.com",
            "metric_name": "val_loss",
            "metric_direction": "lower",
            "max_experiments": "10",
            "notes": "fast CI test",
        },
        files={"model_file": ("model.pt", io.BytesIO(payload), "application/octet-stream")},
    )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]
    assert job_id

    # Poll until completed or failed (max ~20 s; simulate mode is fast)
    deadline = time.time() + 20
    status = None
    while time.time() < deadline:
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        d = r.json()
        status = d["status"]
        if status in ("completed", "failed"):
            break
        time.sleep(0.3)

    assert status == "completed", f"Job did not complete in time; last status: {status}"

    # Confirm metric fields present
    d = client.get(f"/api/jobs/{job_id}").json()
    assert d["baseline_metric"] is not None
    assert d["best_metric"] is not None
    assert d["experiments_done"] == 10

    # results.tsv
    tsv_resp = client.get(f"/api/jobs/{job_id}/results.tsv")
    assert tsv_resp.status_code == 200
    assert tsv_resp.text.startswith("commit\t")

    # report.md
    md_resp = client.get(f"/api/jobs/{job_id}/report.md")
    assert md_resp.status_code == 200
    assert "NightShift Optimization Report" in md_resp.text


def test_rejects_bad_extension(client):
    """Files with disallowed extensions must get HTTP 400."""
    resp = client.post(
        "/api/jobs",
        data={
            "customer_email": "bad@example.com",
            "metric_name": "acc",
            "metric_direction": "higher",
            "max_experiments": "5",
        },
        files={"model_file": ("malware.exe", io.BytesIO(b"MZ"), "application/octet-stream")},
    )
    assert resp.status_code == 400
    assert ".exe" in resp.text


def test_list_jobs(client):
    """/api/jobs returns a list (possibly empty initially)."""
    resp = client.get("/api/jobs")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_unknown_job_404(client):
    """Unknown job IDs return 404."""
    resp = client.get("/api/jobs/doesnotexist")
    assert resp.status_code == 404

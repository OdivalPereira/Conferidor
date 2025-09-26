import importlib
import json
import sys
from pathlib import Path

import pytest


class _DummyTask:
    def __init__(self, done: bool):
        self._done = done

    def done(self) -> bool:  # pragma: no cover - simple helper
        return self._done


@pytest.fixture
def ui_server_cleanup(monkeypatch, tmp_path):
    data_dir = tmp_path / "out"
    uploads_dir = tmp_path / "dados"
    data_dir.mkdir()
    uploads_dir.mkdir()

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("UPLOADS_DIR", str(uploads_dir))

    from fastapi.dependencies import utils as fastapi_utils

    monkeypatch.setattr(fastapi_utils, "ensure_multipart_is_installed", lambda: None)

    if "ui_server" in sys.modules:
        module = importlib.reload(sys.modules["ui_server"])
    else:
        module = importlib.import_module("ui_server")

    try:
        yield module, data_dir, uploads_dir
    finally:
        importlib.reload(module)


def _write_status(jobs_dir: Path, job_id: str, status: str) -> None:
    job_dir = jobs_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": job_id,
        "status": status,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:10:00Z",
        "logs": [],
    }
    (job_dir / "status.json").write_text(json.dumps(payload), encoding="utf-8")


def test_delete_data_removes_completed_jobs_and_preserves_running(ui_server_cleanup):
    module, data_dir, uploads_dir = ui_server_cleanup

    jobs_dir = data_dir / "jobs"
    jobs_dir.mkdir(exist_ok=True)

    running_job = "running-job"
    finished_job = "finished-job"
    pending_job = "pending-job"

    # Create job outputs
    (data_dir / running_job).mkdir()
    (data_dir / running_job / "result.txt").write_text("ok", encoding="utf-8")

    finished_dir = data_dir / finished_job
    finished_dir.mkdir()
    (finished_dir / "output.txt").write_text("done", encoding="utf-8")

    (data_dir / "orphan.txt").write_text("data", encoding="utf-8")

    # Upload directories
    running_upload = uploads_dir / running_job
    running_upload.mkdir()
    (running_upload / "input.csv").write_text("a,b", encoding="utf-8")

    finished_upload = uploads_dir / finished_job
    finished_upload.mkdir()
    (finished_upload / "input.csv").write_text("1,2", encoding="utf-8")

    pending_upload = uploads_dir / pending_job
    pending_upload.mkdir()
    (pending_upload / "input.csv").write_text("x,y", encoding="utf-8")

    # Job statuses
    _write_status(jobs_dir, running_job, "running")
    _write_status(jobs_dir, finished_job, "success")
    _write_status(jobs_dir, pending_job, "queued")

    module.JOB_TASKS.clear()
    module.JOB_TASKS[running_job] = _DummyTask(done=False)

    response = module.api_delete_data()

    assert response == {"removed_files": 4}

    # Running job artifacts remain intact
    assert (data_dir / running_job / "result.txt").exists()
    assert (jobs_dir / running_job / "status.json").exists()
    assert (uploads_dir / running_job / "input.csv").exists()

    # Finished job artifacts removed
    assert not finished_dir.exists()
    assert not (jobs_dir / finished_job).exists()
    assert not finished_upload.exists()

    # Pending job uploads preserved
    assert (uploads_dir / pending_job / "input.csv").exists()
    # Loose files removed from out/
    assert not (data_dir / "orphan.txt").exists()

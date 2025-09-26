import asyncio
import importlib
import json
import sys

import pytest


async def _consume_streaming(response):
    body = bytearray()
    async for chunk in response.body_iterator:  # type: ignore[attr-defined]
        if isinstance(chunk, str):
            body.extend(chunk.encode("utf-8"))
        else:
            body.extend(chunk)
    return bytes(body)


@pytest.fixture
def ui_server_module(monkeypatch, tmp_path):
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


def test_process_status_returns_percent(ui_server_module):
    module, data_dir, _ = ui_server_module

    job_id = "job-123"
    status_path = data_dir / "jobs" / job_id / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_payload = {
        "job_id": job_id,
        "status": "running",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:05:00Z",
        "progress": {"completed": 2, "total": 4},
        "logs": [],
    }
    status_path.write_text(json.dumps(status_payload), encoding="utf-8")

    payload = module.api_process_status(job_id)
    assert payload["status"] == "running"
    assert payload["progress"]["percent"] == 50.0
    assert payload["log_url"] is None


def test_process_logs_streams_file(ui_server_module):
    module, data_dir, _ = ui_server_module

    job_id = "job-logs"
    job_dir = data_dir / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    status_path = job_dir / "status.json"
    status_path.write_text(json.dumps({"job_id": job_id, "status": "running", "logs": []}), encoding="utf-8")

    log_path = job_dir / "pipeline.log"
    log_content = "first line\nsecond line\n"
    log_path.write_text(log_content, encoding="utf-8")

    response = module.api_process_logs(job_id)
    body = asyncio.run(_consume_streaming(response))
    assert body.decode("utf-8") == log_content
    assert response.headers["X-Log-Size"] == str(len(log_content.encode("utf-8")))

    truncated = module.api_process_logs(job_id, offset=len("first line\n"))
    truncated_body = asyncio.run(_consume_streaming(truncated))
    assert truncated_body.decode("utf-8") == "second line\n"


def test_process_cancel_marks_status_and_event(ui_server_module):
    module, data_dir, _ = ui_server_module

    job_id = "job-cancel"
    job_dir = data_dir / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    status_path = job_dir / "status.json"
    status_path.write_text(
        json.dumps({
            "job_id": job_id,
            "status": "running",
            "logs": [],
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:10:00Z",
        }),
        encoding="utf-8",
    )

    module.JOB_TASKS.clear()
    module.JOB_CANCEL_EVENTS.clear()

    response = asyncio.run(module.api_process_cancel(job_id))
    assert response.status_code == 202
    payload = json.loads(response.body.decode("utf-8"))
    assert payload["cancel_requested"] is True
    assert payload["status"] == "cancelling"

    updated_status = json.loads(status_path.read_text(encoding="utf-8"))
    assert updated_status["status"] == "cancelling"
    assert updated_status["logs"]
    assert "Cancellation" in updated_status["logs"][-1]["message"]

    assert job_id in module.JOB_CANCEL_EVENTS
    assert module.JOB_CANCEL_EVENTS[job_id].is_set()


def test_cancel_finished_job_returns_conflict(ui_server_module):
    module, data_dir, _ = ui_server_module

    job_id = "job-finished"
    job_dir = data_dir / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    status_path = job_dir / "status.json"
    status_path.write_text(json.dumps({"job_id": job_id, "status": "success", "logs": []}), encoding="utf-8")

    with pytest.raises(module.HTTPException) as excinfo:
        asyncio.run(module.api_process_cancel(job_id))
    assert excinfo.value.status_code == 409

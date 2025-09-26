import importlib
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import pytest
from fastapi.testclient import TestClient


def _upload_sample_payload(client: TestClient) -> Dict[str, object]:
    files = [
        ("files", ("sucessor.csv", "id\n1\n", "text/csv")),
        ("files", ("suprema_entradas.csv", "id\n2\n", "text/csv")),
    ]

    response = client.post("/api/uploads", files=files)
    assert response.status_code == 200
    payload = response.json()
    assert "job_id" in payload
    return payload


def _wait_for_status(client: TestClient, job_id: str, expected: str, timeout: float = 5.0) -> Dict[str, object]:
    deadline = time.time() + timeout
    last_payload: Dict[str, object] = {}

    while time.time() < deadline:
        response = client.get(f"/api/process/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        last_payload = payload
        if payload.get("status") == expected:
            return payload
        time.sleep(0.05)

    pytest.fail(f"Timeout waiting for status '{expected}', last payload: {last_payload}")


@pytest.fixture
def client_with_temp_dirs(monkeypatch, tmp_path) -> Tuple[TestClient, object, Path, Path]:
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

    module.JOB_TASKS.clear()
    module.JOB_CANCEL_EVENTS.clear()

    client = TestClient(module.app)

    try:
        yield client, module, data_dir, uploads_dir
    finally:
        module.JOB_TASKS.clear()
        module.JOB_CANCEL_EVENTS.clear()
        module.app.dependency_overrides.clear()
        importlib.reload(module)


def test_pipeline_success_flow(client_with_temp_dirs, monkeypatch):
    """Simula um job bem-sucedido retornando ``0`` de ``run_pipeline.main``."""

    client, module, data_dir, _ = client_with_temp_dirs
    upload_payload = _upload_sample_payload(client)
    job_id = upload_payload["job_id"]

    recorded_args: Dict[str, object] = {}

    def fake_run_pipeline(argv):
        recorded_args["argv"] = argv
        # Devolver 0 sinaliza sucesso — a API deve concluir o polling com status ``success``.
        return 0

    monkeypatch.setattr(module, "run_pipeline_main", fake_run_pipeline)

    response = client.post("/api/process", json={"job_id": job_id})
    assert response.status_code == 202
    process_payload = response.json()
    assert process_payload["job_id"] == job_id
    assert process_payload["status"] == "queued"
    assert process_payload["status_url"].endswith(job_id)

    status_payload = _wait_for_status(client, job_id, expected="success")
    assert status_payload["status"] == "success"
    assert status_payload.get("exit_code") == 0
    assert status_payload.get("log_url") == f"/api/process/{job_id}/logs"

    job_status = client.get(f"/api/jobs/{job_id}")
    assert job_status.status_code == 200
    assert job_status.json()["status"] == "success"

    log_response = client.get(status_payload["log_url"])
    assert log_response.status_code == 200
    assert "Pipeline completed successfully" in log_response.text

    assert "argv" in recorded_args
    assert recorded_args["argv"][0] == "--dados-dir"
    job_out_dir = data_dir / job_id
    assert job_out_dir.exists()


def test_pipeline_error_flow(client_with_temp_dirs, monkeypatch):
    """Simula um job que falha retornando ``2`` — a API deve sinalizar ``error``."""

    client, module, data_dir, _ = client_with_temp_dirs
    upload_payload = _upload_sample_payload(client)
    job_id = upload_payload["job_id"]

    def fake_run_pipeline(_argv):
        # Ao devolver um código diferente de zero o worker registra erro e devolve HTTP 200 no polling.
        return 2

    monkeypatch.setattr(module, "run_pipeline_main", fake_run_pipeline)

    response = client.post("/api/process", json={"job_id": job_id})
    assert response.status_code == 202

    status_payload = _wait_for_status(client, job_id, expected="error")
    assert status_payload["status"] == "error"
    assert status_payload.get("exit_code") == 2
    assert status_payload.get("log_url") == f"/api/process/{job_id}/logs"

    log_response = client.get(status_payload["log_url"])
    assert log_response.status_code == 200
    assert "Pipeline exited with code 2" in log_response.text

    job_out_dir = data_dir / job_id
    assert job_out_dir.exists()

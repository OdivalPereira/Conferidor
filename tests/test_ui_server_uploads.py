import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from src import ui_server


@pytest.fixture
def client(tmp_path):
    app = ui_server.app
    app.dependency_overrides[ui_server.get_uploads_root] = lambda: tmp_path
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_api_uploads_success(client, tmp_path):
    files = [
        ("files", ("sucessor.csv", "col1\n1\n", "text/csv")),
        ("files", ("entradas.csv", "col1\n2\n", "text/csv")),
    ]

    response = client.post("/api/uploads", files=files)

    assert response.status_code == 200
    payload = response.json()
    assert "job_id" in payload
    assert payload["file_count"] == 2

    job_dir = tmp_path / payload["job_id"]
    assert job_dir.is_dir()

    manifest_path = job_dir / "manifest.json"
    assert manifest_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["job_id"] == payload["job_id"]
    assert len(manifest["files"]) == 2

    first_file = manifest["files"][0]
    expected_hash = hashlib.sha256(files[0][1][1].encode("utf-8")).hexdigest()
    assert first_file["hash"] == expected_hash
    assert (job_dir / first_file["stored_name"]).is_file()


def test_api_uploads_rejects_non_csv(client):
    files = [("files", ("document.txt", "irrelevant", "text/plain"))]

    response = client.post("/api/uploads", files=files)

    assert response.status_code == 400
    assert response.json()["detail"].startswith("Unsupported file extension")

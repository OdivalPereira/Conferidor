import importlib
import json
import sys

import pytest


@pytest.fixture
def reload_ui_server(monkeypatch, tmp_path):
    data_dir = tmp_path / "out"
    data_dir.mkdir()
    monkeypatch.setenv("DATA_DIR", str(data_dir))

    from fastapi.dependencies import utils as fastapi_utils

    monkeypatch.setattr(fastapi_utils, "ensure_multipart_is_installed", lambda: None)

    if "ui_server" in sys.modules:
        module = importlib.reload(sys.modules["ui_server"])
    else:
        module = importlib.import_module("ui_server")
    yield module, data_dir
    importlib.reload(module)


def test_api_grid_sorts_scores_with_none(reload_ui_server):
    module, data_dir = reload_ui_server
    grid_path = data_dir / "ui_grid.jsonl"
    rows = [
        {"id": 1, "score": 90},
        {"id": 2, "score": None},
        {"id": 3, "score": 75},
    ]
    grid_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    base_kwargs = {
        "limit": 200,
        "offset": 0,
        "status": None,
        "fonte_tipo": None,
        "cfop": None,
        "q": None,
    }

    ascending = module.api_grid(sort_by="score", sort_dir="asc", **base_kwargs)
    asc_scores = [item["score"] for item in ascending["items"]]
    assert asc_scores == [75, 90, None]

    descending = module.api_grid(sort_by="score", sort_dir="desc", **base_kwargs)
    desc_scores = [item["score"] for item in descending["items"]]
    assert desc_scores == [None, 90, 75]


def test_manual_override_persists_after_reload(reload_ui_server):
    module, data_dir = reload_ui_server
    grid_path = data_dir / "ui_grid.jsonl"
    base_row = {"id": "row-1", "status": "ALERTA", "motivos": "regra_a"}
    grid_path.write_text(json.dumps(base_row) + "\n", encoding="utf-8")

    payload = module.ManualStatusPayload(row_id="row-1", status="OK", original_status="ALERTA")
    response = module.api_manual_status(payload)
    assert response["ok"] is True

    base_kwargs = {
        "limit": 200,
        "offset": 0,
        "status": None,
        "fonte_tipo": None,
        "cfop": None,
        "q": None,
    }

    first_result = module.api_grid(sort_by=None, sort_dir="desc", **base_kwargs)
    assert first_result["items"][0]["status"] == "OK"
    assert first_result["items"][0]["original_status"] == "ALERTA"
    assert "ajuste_manual" in first_result["items"][0].get("motivos", "")

    module = importlib.reload(module)
    second_result = module.api_grid(sort_by=None, sort_dir="desc", **base_kwargs)
    assert second_result["items"][0]["status"] == "OK"
    assert second_result["items"][0]["original_status"] == "ALERTA"
    assert "ajuste_manual" in second_result["items"][0].get("motivos", "")


def test_manual_override_delete_removes_adjustment(reload_ui_server):
    module, data_dir = reload_ui_server
    grid_path = data_dir / "ui_grid.jsonl"
    base_row = {"id": "row-2", "status": "DIVERGENCIA", "motivos": "regra_b"}
    grid_path.write_text(json.dumps(base_row) + "\n", encoding="utf-8")

    payload = module.ManualStatusPayload(row_id="row-2", status="OK", original_status="DIVERGENCIA")
    module.api_manual_status(payload)

    delete_response = module.api_manual_status_delete(row_id="row-2")
    assert delete_response["ok"] is True

    base_kwargs = {
        "limit": 200,
        "offset": 0,
        "status": None,
        "fonte_tipo": None,
        "cfop": None,
        "q": None,
    }

    result = module.api_grid(sort_by=None, sort_dir="desc", **base_kwargs)
    row = result["items"][0]
    assert row["status"] == "DIVERGENCIA"
    assert "ajuste_manual" not in (row.get("motivos") or "")
    assert not row.get("_manual")
    assert row.get("original_status") in (None, "")


def test_ui_app_endpoint_serves_html(reload_ui_server):
    module, _ = reload_ui_server

    root_response = module.index()
    assert root_response.status_code == 200
    assert 'id="root"' in root_response.body.decode("utf-8")

    app_response = module.ui_app()
    assert app_response.status_code == 200
    assert 'id="root"' in app_response.body.decode("utf-8")

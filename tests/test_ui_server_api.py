import importlib
import json
import sys

import pytest


@pytest.fixture
def reload_ui_server(monkeypatch, tmp_path):
    data_dir = tmp_path / "out"
    data_dir.mkdir()
    monkeypatch.setenv("DATA_DIR", str(data_dir))

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


def test_index_serves_ui_html(reload_ui_server):
    module, _ = reload_ui_server
    response = module.index()
    assert response.status_code == 200
    assert 'id="root"' in response.body.decode("utf-8")

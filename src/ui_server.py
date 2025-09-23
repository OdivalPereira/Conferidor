﻿from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from export_xlsx import run as run_export_xlsx  # type: ignore
except Exception:  # pragma: no cover
    run_export_xlsx = None

try:
    from export_pdf import run as run_export_pdf  # type: ignore
except Exception:  # pragma: no cover
    run_export_pdf = None

APP_TITLE = "Conferidor UI"
DATA_DIR = Path(os.environ.get("DATA_DIR", "out")).resolve()
SCHEMA_PATH = Path(os.environ.get("UI_SCHEMA", "cfg/ui_schema.json")).resolve()

app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if DATA_DIR.exists():
    app.mount("/files", StaticFiles(directory=str(DATA_DIR)), name="files")


def _read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise HTTPException(404, detail=f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise HTTPException(404, detail=f"File not found: {path}")
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = f"""
    <html><head><meta charset='utf-8'><title>{APP_TITLE}</title>
    <style>
      body {{ font-family: system-ui, sans-serif; margin: 0; padding: 24px; background: #f9fafb; color: #1f2937; }}
      a {{ color: #2563eb; text-decoration: none; }}
      .card {{ background: #fff; border-radius: 12px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(15,23,42,0.08); }}
    </style></head><body>
      <h1>{APP_TITLE}</h1>
      <div class="card">
        <p><strong>DATA_DIR:</strong> {DATA_DIR}</p>
        <p>Downloads: <a href="/files/" target="_blank">/files/</a></p>
      </div>
      <div class="card">
        <h3>APIs</h3>
        <ul>
          <li><a href="/api/health" target="_blank">/api/health</a></li>
          <li><a href="/api/schema" target="_blank">/api/schema</a></li>
          <li><a href="/api/meta" target="_blank">/api/meta</a></li>
          <li><a href="/api/grid?limit=50" target="_blank">/api/grid</a></li>
          <li><a href="/api/files" target="_blank">/api/files</a></li>
        </ul>
      </div>
    </body></html>
    """
    return HTMLResponse(html)


@app.get("/api/health")
def api_health() -> Dict[str, object]:
    return {"ok": True, "data_dir": str(DATA_DIR)}


@app.get("/api/schema")
def api_schema() -> Dict[str, object]:
    if not SCHEMA_PATH.exists():
        return {"version": 1, "columns": []}
    return _read_json(SCHEMA_PATH)


@app.get("/api/meta")
def api_meta() -> Dict[str, object]:
    meta_path = DATA_DIR / "ui_meta.json"
    if not meta_path.exists():
        raise HTTPException(404, detail="ui_meta.json not found")
    return _read_json(meta_path)


@app.get("/api/grid")
def api_grid(
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    status: Optional[str] = None,
    fonte_tipo: Optional[str] = None,
    cfop: Optional[str] = None,
    q: Optional[str] = Query(None, description="Simple full text filter"),
    sort_by: Optional[str] = Query(None),
    sort_dir: str = Query("desc", regex="^(asc|desc)$"),
) -> Dict[str, object]:
    grid_path = DATA_DIR / "ui_grid.jsonl"
    rows = _read_jsonl(grid_path)

    def row_match(row: Dict[str, object]) -> bool:
        if status and str(row.get("status") or "").upper() != status.upper():
            return False
        if fonte_tipo and str(row.get("fonte_tipo") or "").upper() != fonte_tipo.upper():
            return False
        if cfop and str(row.get("F.cfop") or "") != cfop:
            return False
        if q:
            haystack = " ".join(
                str(row.get(key) or "")
                for key in ("S.historico", "S.doc", "F.doc", "F.participante", "motivos")
            ).lower()
            if q.lower() not in haystack:
                return False
        return True

    filtered = [row for row in rows if row_match(row)]

    if sort_by:
        reverse = sort_dir.lower() != "asc"
        filtered.sort(key=lambda item: item.get(sort_by), reverse=reverse)

    paginated = filtered[offset : offset + limit]
    return {
        "total": len(rows),
        "total_filtered": len(filtered),
        "returned": len(paginated),
        "offset": offset,
        "limit": limit,
        "items": paginated,
    }


@app.get("/api/files")
def api_files() -> Dict[str, object]:
    if not DATA_DIR.exists():
        raise HTTPException(404, detail=f"Directory not found: {DATA_DIR}")
    items: List[Dict[str, object]] = []
    for path in DATA_DIR.rglob("*"):
        if path.is_file():
            rel = path.relative_to(DATA_DIR).as_posix()
            items.append({
                "name": path.name,
                "path": rel,
                "size": path.stat().st_size,
                "download": f"/files/{rel}",
            })
    items.sort(key=lambda item: item["path"].lower())
    return {"count": len(items), "items": items}


@app.post("/api/export/xlsx")
def api_export_xlsx(payload: Dict[str, object]):
    if run_export_xlsx is None:
        raise HTTPException(500, detail="export_xlsx is not available in this runtime")
    grid = payload.get("grid", "out/match/reconc_grid.csv")
    sem_fonte = payload.get("sem_fonte", "out/match/reconc_sem_fonte.csv")
    sem_sucessor = payload.get("sem_sucessor", "out/match/reconc_sem_sucessor.csv")
    out = payload.get("out", "out/relatorio_conferencia.xlsx")
    result = run_export_xlsx(
        grid_csv=str(Path(grid)),
        sem_fonte_csv=str(Path(sem_fonte)),
        sem_sucessor_csv=str(Path(sem_sucessor)),
        out_path=str(Path(out)),
    )
    rel = Path(result["out"]).resolve().relative_to(DATA_DIR)
    result["download"] = f"/files/{rel.as_posix()}"
    return result


@app.post("/api/export/pdf")
def api_export_pdf(payload: Dict[str, object]):
    if run_export_pdf is None:
        raise HTTPException(500, detail="export_pdf is not available in this runtime")
    grid = payload.get("grid", "out/match/reconc_grid.csv")
    out = payload.get("out", "out/relatorio_conferencia.pdf")
    cliente = payload.get("cliente")
    periodo = payload.get("periodo")
    result = run_export_pdf(
        grid_csv=str(Path(grid)),
        out_path=str(Path(out)),
        cliente=cliente,
        periodo=periodo,
    )
    if result.get("pdf"):
        rel = Path(result["pdf"]).resolve().relative_to(DATA_DIR)
        result["download"] = f"/files/{rel.as_posix()}"
    if result.get("html"):
        rel = Path(result["html"]).resolve().relative_to(DATA_DIR)
        result["download_html"] = f"/files/{rel.as_posix()}"
    return result


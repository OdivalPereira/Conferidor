from __future__ import annotations

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
DATA_DIR = Path(os.environ.get("DATA_DIR", "out")).expanduser().resolve()
SCHEMA_PATH = Path(os.environ.get("UI_SCHEMA", "cfg/ui_schema.json")).expanduser().resolve()
UI_APP_PATH = Path(__file__).resolve().parent / "ui_app.html"

DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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


def _coerce_path(value: object | None) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)


def _resolve_data_path(value: Optional[str], default_relative: str) -> Path:
    if value:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (DATA_DIR / candidate).resolve()
        else:
            candidate = candidate.resolve()
    else:
        candidate = (DATA_DIR / default_relative).resolve()
    return candidate


def _relative_download(path: Path) -> Optional[str]:
    try:
        rel = path.resolve().relative_to(DATA_DIR)
    except ValueError:
        return None
    return f"/files/{rel.as_posix()}"


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    if not UI_APP_PATH.exists():
        raise HTTPException(500, detail=f"ui_app.html not found at {UI_APP_PATH}")
    html = UI_APP_PATH.read_text(encoding="utf-8")
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
        filtered.sort(
            key=lambda item: (item.get(sort_by) is None, item.get(sort_by)),
            reverse=reverse,
        )

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
    items: List[Dict[str, object]] = []
    for path in DATA_DIR.rglob("*"):
        if path.is_file():
            rel = path.relative_to(DATA_DIR).as_posix()
            items.append(
                {
                    "name": path.name,
                    "path": rel,
                    "size": path.stat().st_size,
                    "download": f"/files/{rel}",
                }
            )
    items.sort(key=lambda item: item["path"].lower())
    return {"count": len(items), "items": items}


@app.post("/api/export/xlsx")
def api_export_xlsx(payload: Dict[str, object]):
    if run_export_xlsx is None:
        raise HTTPException(500, detail="export_xlsx is not available in this runtime")
    grid = _resolve_data_path(_coerce_path(payload.get("grid")), "match/reconc_grid.csv")
    sem_fonte = _resolve_data_path(_coerce_path(payload.get("sem_fonte")), "match/reconc_sem_fonte.csv")
    sem_sucessor = _resolve_data_path(_coerce_path(payload.get("sem_sucessor")), "match/reconc_sem_sucessor.csv")
    out = _resolve_data_path(_coerce_path(payload.get("out")), "relatorio_conferencia.xlsx")
    result = run_export_xlsx(
        grid_csv=str(grid),
        sem_fonte_csv=str(sem_fonte),
        sem_sucessor_csv=str(sem_sucessor),
        out_path=str(out),
    )
    out_path = Path(str(result.get("out", out))).resolve()
    download = _relative_download(out_path)
    if download:
        result["download"] = download
    else:
        result["absolute_out"] = str(out_path)
    return result


@app.post("/api/export/pdf")
def api_export_pdf(payload: Dict[str, object]):
    if run_export_pdf is None:
        raise HTTPException(500, detail="export_pdf is not available in this runtime")
    grid = _resolve_data_path(_coerce_path(payload.get("grid")), "match/reconc_grid.csv")
    out = _resolve_data_path(_coerce_path(payload.get("out")), "relatorio_conferencia.pdf")
    cliente = payload.get("cliente")
    periodo = payload.get("periodo")
    result = run_export_pdf(
        grid_csv=str(grid),
        out_path=str(out),
        cliente=cliente,
        periodo=periodo,
    )
    if result.get("pdf"):
        pdf_path = Path(str(result["pdf"])).resolve()
        download = _relative_download(pdf_path)
        if download:
            result["download"] = download
        else:
            result["absolute_pdf"] = str(pdf_path)
    if result.get("html"):
        html_path = Path(str(result["html"])).resolve()
        download_html = _relative_download(html_path)
        if download_html:
            result["download_html"] = download_html
        else:
            result["absolute_html"] = str(html_path)
    return result

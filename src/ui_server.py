from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from export_xlsx import run as run_export_xlsx  # type: ignore
except Exception:  # pragma: no cover
    run_export_xlsx = None

try:
    from export_pdf import run as run_export_pdf  # type: ignore
except Exception:  # pragma: no cover
    run_export_pdf = None

try:
    from run_pipeline import main as run_pipeline_main  # type: ignore
except Exception:  # pragma: no cover
    run_pipeline_main = None

APP_TITLE = "Conferidor UI"
DATA_DIR = Path(os.environ.get("DATA_DIR", "out")).expanduser().resolve()
UPLOADS_DIR = Path(os.environ.get("UPLOADS_DIR", "dados")).expanduser().resolve()
SCHEMA_PATH = Path(os.environ.get("UI_SCHEMA", "cfg/ui_schema.json")).expanduser().resolve()
CFG_DIR = Path(os.environ.get("CFG_DIR", "cfg")).expanduser().resolve()
UI_APP_PATH = Path(__file__).resolve().parent / "ui_app.html"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR = DATA_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

JOB_TASKS: Dict[str, asyncio.Task[Any]] = {}


class ProcessPayload(BaseModel):
    job_id: str
    dados_dir: Optional[str] = None
    cfg_dir: Optional[str] = None
    out_dir: Optional[str] = None
    pipeline_params: Dict[str, Any] = Field(default_factory=dict)

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


def _ui_app_html() -> str:
    if not UI_APP_PATH.exists():
        raise HTTPException(500, detail=f"ui_app.html not found at {UI_APP_PATH}")
    return UI_APP_PATH.read_text(encoding="utf-8")


def get_uploads_root() -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOADS_DIR


def _ensure_csv(filename: Optional[str]) -> None:
    if not filename:
        raise HTTPException(400, detail="Missing filename")
    if Path(filename).suffix.lower() != ".csv":
        raise HTTPException(400, detail=f"Unsupported file extension for {filename}")


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _job_dir(job_id: str) -> Path:
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def _job_status_path(job_id: str) -> Path:
    return _job_dir(job_id) / "status.json"


def _resolve_user_path(value: Optional[str], default: Path) -> Path:
    if not value:
        return default
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _load_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    status_path = _job_status_path(job_id)
    if not status_path.exists():
        return None
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_job_status(job_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    status_path = _job_status_path(job_id)
    status_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def _append_job_log(data: Dict[str, Any], message: str, level: str = "info") -> None:
    logs = data.setdefault("logs", [])
    logs.append({"timestamp": _now_iso(), "level": level, "message": message})


def _update_job_status(
    job_id: str,
    *,
    status: Optional[str] = None,
    message: Optional[str] = None,
    level: str = "info",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    data = _load_job_status(job_id) or {
        "job_id": job_id,
        "created_at": _now_iso(),
        "logs": [],
    }
    now = _now_iso()
    data["updated_at"] = now
    if status:
        data["status"] = status
        if status in {"success", "error"}:
            data["finished_at"] = now
    if extra:
        for key, value in extra.items():
            data[str(key)] = _json_safe(value)
    if message:
        _append_job_log(data, message, level=level)
    return _write_job_status(job_id, data)


def _build_pipeline_argv(
    dados_dir: Path,
    out_dir: Path,
    cfg_dir: Path,
    params: Dict[str, Any],
) -> List[str]:
    argv: List[str] = [
        "--dados-dir",
        str(dados_dir),
        "--out-dir",
        str(out_dir),
        "--cfg-dir",
        str(cfg_dir),
    ]

    for key, value in params.items():
        flag = f"--{str(key).replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            argv.append(flag)
            argv.extend(str(item) for item in value)
            continue
        argv.extend([flag, str(value)])

    return argv


def _invoke_pipeline(argv: List[str]) -> int:
    if run_pipeline_main is None:
        raise RuntimeError("run_pipeline.main is not available in this runtime")
    try:
        return run_pipeline_main(argv)
    except SystemExit as exc:  # pragma: no cover - defensive
        code = exc.code if isinstance(exc.code, int) else 1
        return int(code)


async def _pipeline_worker(job_id: str, argv: List[str]) -> None:
    try:
        _update_job_status(job_id, status="running", message="Pipeline execution started")
        exit_code = await asyncio.to_thread(_invoke_pipeline, argv)
        if exit_code == 0:
            _update_job_status(job_id, status="success", message="Pipeline completed successfully", extra={"exit_code": 0})
        else:
            _update_job_status(
                job_id,
                status="error",
                message=f"Pipeline exited with code {exit_code}",
                level="error",
                extra={"exit_code": exit_code},
            )
    except Exception as exc:  # pragma: no cover - defensive
        _update_job_status(
            job_id,
            status="error",
            message=f"Unhandled error during pipeline execution: {exc}",
            level="error",
            extra={"error": str(exc)},
        )
    finally:
        JOB_TASKS.pop(job_id, None)


async def _monitor_task(task: asyncio.Task[Any]) -> None:
    try:
        await task
    except Exception:  # pragma: no cover - defensive
        pass


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(_ui_app_html())


@app.get("/app", response_class=HTMLResponse)
def ui_app() -> HTMLResponse:
    return HTMLResponse(_ui_app_html())


@app.get("/ui", response_class=HTMLResponse)
def ui_route() -> HTMLResponse:
    return HTMLResponse(_ui_app_html())


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


@app.post("/api/uploads")
def api_uploads(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    uploads_root: Path = Depends(get_uploads_root),
) -> Dict[str, object]:
    if not files:
        raise HTTPException(400, detail="No files received")

    for upload in files:
        _ensure_csv(upload.filename)

    job_id = uuid4().hex
    job_dir = uploads_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    manifest_files: List[Dict[str, object]] = []
    created_at = _now_iso()

    for upload in files:
        background_tasks.add_task(upload.file.close)
        filename = upload.filename
        _ensure_csv(filename)
        safe_name = Path(filename).name
        dest_path = job_dir / safe_name

        hasher = hashlib.sha256()
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=job_dir) as tmp:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8")
                hasher.update(chunk)
                tmp.write(chunk)
        shutil.move(tmp.name, dest_path)

        manifest_files.append(
            {
                "original_name": filename,
                "stored_name": safe_name,
                "hash": hasher.hexdigest(),
                "timestamp": _now_iso(),
            }
        )

    manifest = {
        "job_id": job_id,
        "created_at": created_at,
        "files": manifest_files,
    }
    manifest_path = job_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"job_id": job_id, "file_count": len(manifest_files)}


@app.get("/api/jobs/{job_id}")
def api_job_status(job_id: str) -> Dict[str, Any]:
    status = _load_job_status(job_id)
    if not status:
        raise HTTPException(404, detail=f"Job '{job_id}' not found")
    return status


@app.post("/api/process")
async def api_process(
    payload: ProcessPayload,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    job_id = payload.job_id.strip()
    if not job_id:
        raise HTTPException(400, detail="job_id is required")

    uploads_dir = UPLOADS_DIR / job_id
    if not uploads_dir.exists() or not uploads_dir.is_dir():
        raise HTTPException(404, detail=f"Upload job '{job_id}' not found")

    existing = _load_job_status(job_id)
    if existing and existing.get("status") in {"queued", "running"}:
        raise HTTPException(409, detail=f"Job '{job_id}' is already in progress")

    dados_dir = _resolve_user_path(payload.dados_dir, uploads_dir)
    if not dados_dir.exists():
        raise HTTPException(400, detail=f"dados_dir not found: {dados_dir}")

    cfg_dir = _resolve_user_path(payload.cfg_dir, CFG_DIR)
    if not cfg_dir.exists():
        raise HTTPException(400, detail=f"cfg_dir not found: {cfg_dir}")

    out_dir = _resolve_user_path(payload.out_dir, DATA_DIR / job_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    _job_dir(job_id)

    params = {str(key): value for key, value in payload.pipeline_params.items()}
    argv = _build_pipeline_argv(dados_dir, out_dir, cfg_dir, params)

    initial_status = {
        "job_id": job_id,
        "status": "queued",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "dados_dir": str(dados_dir),
        "cfg_dir": str(cfg_dir),
        "out_dir": str(out_dir),
        "params": _json_safe(params),
        "argv": list(argv),
        "logs": [],
    }
    _append_job_log(initial_status, "Job queued")
    _write_job_status(job_id, initial_status)

    task = asyncio.create_task(_pipeline_worker(job_id, argv))
    JOB_TASKS[job_id] = task
    background_tasks.add_task(_monitor_task, task)

    response = JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "queued",
            "status_url": f"/api/jobs/{job_id}",
        },
    )
    response.headers["Location"] = f"/api/jobs/{job_id}"
    return response


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

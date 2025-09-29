from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import re
import shutil
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from export_xlsx import run as run_export_xlsx  # type: ignore
except Exception:  # pragma: no cover
    run_export_xlsx = None

try:
    from export_json import run as run_export_json  # type: ignore
except Exception:  # pragma: no cover
    run_export_json = None

try:
    from export_pdf import run as run_export_pdf  # type: ignore
except Exception:  # pragma: no cover
    run_export_pdf = None

try:
    from run_pipeline import (  # type: ignore
        PipelineCancelled as RunPipelineCancelled,
        main as run_pipeline_main,
        set_cancel_check as run_pipeline_set_cancel_check,
    )
except Exception:  # pragma: no cover
    run_pipeline_main = None
    RunPipelineCancelled = None
    run_pipeline_set_cancel_check = None

APP_TITLE = "Conferidor UI"
DATA_DIR = Path(os.environ.get("DATA_DIR", "out")).expanduser().resolve()
UPLOADS_DIR = Path(os.environ.get("UPLOADS_DIR", "dados")).expanduser().resolve()
SCHEMA_PATH = Path(os.environ.get("UI_SCHEMA", "cfg/ui_schema.json")).expanduser().resolve()
CFG_DIR = Path(os.environ.get("CFG_DIR", "cfg")).expanduser().resolve()
UI_APP_PATH = Path(__file__).resolve().parent / "ui_app.html"
STATIC_DIR = Path(__file__).resolve().parent / "static"
MANUAL_OVERRIDES_PATH = DATA_DIR / "manual_overrides.jsonl"
MANUAL_OVERRIDE_TAG = "ajuste_manual"
REQUIREMENTS_PATH = Path(__file__).resolve().parent.parent / "requirements.txt"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR = DATA_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

JOB_TASKS: Dict[str, asyncio.Task[Any]] = {}
JOB_CANCEL_EVENTS: Dict[str, asyncio.Event] = {}


def _count_files(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        return 1
    count = 0
    for entry in path.rglob("*"):
        if entry.is_file() or entry.is_symlink():
            count += 1
    return count


def _safe_rmtree(path: Path) -> int:
    if not path.exists():
        return 0
    removed = _count_files(path)
    shutil.rmtree(path)
    return removed


def _delete_path(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_dir() and not path.is_symlink():
        return _safe_rmtree(path)
    path.unlink()
    return 1


_RE_REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9_.-]+")


def _parse_requirement_names(path: Path = REQUIREMENTS_PATH) -> List[str]:
    if not path.exists():
        return []
    names: List[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split(";", 1)[0].strip()
        match = _RE_REQUIREMENT_NAME.match(line)
        if not match:
            continue
        name = match.group(0)
        if name:
            names.append(name)
    return names


def _dependency_versions(names: Iterable[str]) -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    unique = sorted({name for name in names}, key=lambda value: value.lower())
    for name in unique:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _path_status(path: Path) -> Dict[str, object]:
    exists = path.exists()
    status: Dict[str, object] = {"path": str(path), "exists": exists}
    if exists:
        status["is_dir"] = path.is_dir()
        try:
            status["modified_at"] = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        except OSError:  # pragma: no cover - filesystem edge case
            pass
    return status


def _system_status() -> Dict[str, object]:
    running_jobs = sum(1 for task in JOB_TASKS.values() if not task.done())
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python_version": platform.python_version(),
        "paths": {
            "data_dir": _path_status(DATA_DIR),
            "uploads_dir": _path_status(UPLOADS_DIR),
            "schema": _path_status(SCHEMA_PATH),
        },
        "jobs": {
            "total": len(JOB_TASKS),
            "running": running_jobs,
        },
    }


class ProcessPayload(BaseModel):
    job_id: str
    dados_dir: Optional[str] = None
    cfg_dir: Optional[str] = None
    out_dir: Optional[str] = None
    pipeline_params: Dict[str, Any] = Field(default_factory=dict)


class ManualStatusPayload(BaseModel):
    row_id: str
    status: str
    original_status: Optional[str] = None

app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
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


def _append_manual_override_record(record: Dict[str, object]) -> None:
    try:
        MANUAL_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MANUAL_OVERRIDES_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # pragma: no cover - best effort persistence
        pass


def _load_manual_overrides(path: Path = MANUAL_OVERRIDES_PATH) -> Dict[str, Dict[str, object]]:
    overrides: Dict[str, Dict[str, object]] = {}
    if not path.exists():
        return overrides
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row_id = str(data.get("row_id") or "").strip()
                if not row_id:
                    continue
                status_value = data.get("status")
                if status_value in (None, ""):
                    overrides.pop(row_id, None)
                    continue
                normalized = dict(data)
                normalized["row_id"] = row_id
                normalized["status"] = str(status_value).upper()
                original = normalized.get("original_status")
                if original:
                    normalized["original_status"] = str(original).upper()
                else:
                    normalized.pop("original_status", None)
                overrides[row_id] = normalized
    except Exception:  # pragma: no cover - defensive
        return overrides
    return overrides


def _infer_row_id_from_row(row: Dict[str, object]) -> Optional[str]:
    identifier = row.get("id")
    if identifier not in (None, ""):
        return str(identifier)
    parts: List[str] = []
    for key in ("sucessor_idx", "fonte_tipo", "fonte_idx"):
        value = row.get(key)
        if value in (None, ""):
            continue
        parts.append(str(value))
    if parts:
        return "-".join(parts)
    return None


def _apply_manual_override_to_row(
    row: Dict[str, object], override: Dict[str, object]
) -> Dict[str, object]:
    status_value = str(override.get("status") or "").upper()
    if not status_value:
        return row
    updated = dict(row)
    original_status = override.get("original_status")
    if original_status:
        updated["original_status"] = str(original_status).upper()
    else:
        previous = (
            updated.get("original_status")
            or updated.get("match.status")
            or updated.get("status")
            or ""
        )
        previous = str(previous).upper()
        if previous:
            updated["original_status"] = previous
        else:
            updated.pop("original_status", None)
    updated["status"] = status_value
    updated["match.status"] = status_value
    motivos_raw = updated.get("motivos")
    if motivos_raw is None and updated.get("match.motivos") is not None:
        motivos_raw = updated.get("match.motivos")
    motivos: List[str] = []
    if motivos_raw:
        motivos = [part for part in str(motivos_raw).split(";") if part]
    if MANUAL_OVERRIDE_TAG not in motivos:
        motivos.append(MANUAL_OVERRIDE_TAG)
    updated["motivos"] = ";".join(motivos)
    updated["_manual"] = True
    return updated


def _apply_manual_overrides(
    rows: List[Dict[str, object]], overrides: Optional[Dict[str, Dict[str, object]]] = None
) -> List[Dict[str, object]]:
    if not rows:
        return rows
    overrides = overrides or _load_manual_overrides()
    if not overrides:
        cleaned: List[Dict[str, object]] = []
        for row in rows:
            cleaned_row = dict(row)
            original_status = cleaned_row.get("original_status")
            if original_status:
                cleaned_row["status"] = str(original_status).upper()
                cleaned_row["match.status"] = str(original_status).upper()
            motivos_raw = cleaned_row.get("motivos") or cleaned_row.get("match.motivos")
            if motivos_raw:
                motivos = [part for part in str(motivos_raw).split(";") if part and part != MANUAL_OVERRIDE_TAG]
                cleaned_row["motivos"] = ";".join(motivos)
            cleaned_row.pop("_manual", None)
            cleaned_row.pop("original_status", None)
            cleaned.append(cleaned_row)
        return cleaned
    applied: List[Dict[str, object]] = []
    for row in rows:
        row_id = _infer_row_id_from_row(row)
        override = overrides.get(row_id) if row_id else None
        if override:
            applied.append(_apply_manual_override_to_row(row, override))
        else:
            cleaned_row = dict(row)
            original_status = cleaned_row.get("original_status")
            if original_status:
                cleaned_row["status"] = str(original_status).upper()
                cleaned_row["match.status"] = str(original_status).upper()
            motivos_raw = cleaned_row.get("motivos") or cleaned_row.get("match.motivos")
            if motivos_raw:
                motivos = [part for part in str(motivos_raw).split(";") if part and part != MANUAL_OVERRIDE_TAG]
                cleaned_row["motivos"] = ";".join(motivos)
            cleaned_row.pop("_manual", None)
            cleaned_row.pop("original_status", None)
            applied.append(cleaned_row)
    return applied


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


def _job_log_path(job_id: str) -> Path:
    return _job_dir(job_id) / "pipeline.log"


def _append_plain_log(log_path: Path, message: str) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{_now_iso()}] {message}\n")
    except Exception:  # pragma: no cover - best effort logging
        pass


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
        if status in {"success", "error", "cancelled"}:
            data["finished_at"] = now
    if extra:
        for key, value in extra.items():
            data[str(key)] = _json_safe(value)
    if message:
        _append_job_log(data, message, level=level)
    return _write_job_status(job_id, data)


def _job_status_payload(job_id: str) -> Dict[str, Any]:
    status = _load_job_status(job_id)
    if not status:
        raise HTTPException(404, detail=f"Job '{job_id}' not found")

    payload: Dict[str, Any] = json.loads(json.dumps(status))
    payload.setdefault("job_id", job_id)

    progress = payload.get("progress")
    if isinstance(progress, dict):
        percent = progress.get("percent")
        completed = progress.get("completed")
        total = progress.get("total")
        if percent is None and isinstance(completed, (int, float)) and isinstance(total, (int, float)) and total:
            try:
                progress["percent"] = round((float(completed) / float(total)) * 100, 2)
            except ZeroDivisionError:
                progress["percent"] = 0.0

    log_path = _job_log_path(job_id)
    if log_path.exists():
        payload["log_url"] = f"/api/process/{job_id}/logs"
        payload["log_size"] = log_path.stat().st_size
    else:
        payload["log_url"] = None

    return payload


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


def _run_pipeline_sync(job_id: str, argv: List[str], cancel_event: asyncio.Event) -> int:
    if run_pipeline_main is None:
        raise RuntimeError("run_pipeline.main is not available in this runtime")

    log_path = _job_log_path(job_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def check_cancel() -> None:
        if cancel_event.is_set():
            if RunPipelineCancelled is not None:
                raise RunPipelineCancelled()
            raise RuntimeError("Pipeline cancelled")

    with open(log_path, "a", encoding="utf-8") as log_file:
        if run_pipeline_set_cancel_check is not None and RunPipelineCancelled is not None:
            run_pipeline_set_cancel_check(check_cancel)
        try:
            with redirect_stdout(log_file), redirect_stderr(log_file):
                return _invoke_pipeline(argv)
        finally:
            if run_pipeline_set_cancel_check is not None:
                run_pipeline_set_cancel_check(None)
            log_file.flush()


async def _pipeline_worker(job_id: str, argv: List[str]) -> None:
    cancel_event = JOB_CANCEL_EVENTS.setdefault(job_id, asyncio.Event())
    log_path = _job_log_path(job_id)

    try:
        if cancel_event.is_set():
            _append_plain_log(log_path, "Job cancelled before start")
            _update_job_status(
                job_id,
                status="cancelled",
                message="Job cancelled before start",
                level="warning",
                extra={"cancelled": True},
            )
            return

        _append_plain_log(log_path, "Pipeline execution started")
        _update_job_status(
            job_id,
            status="running",
            message="Pipeline execution started",
            extra={"log_path": str(log_path)},
        )
        exit_code = await asyncio.to_thread(_run_pipeline_sync, job_id, argv, cancel_event)
        if exit_code == 0:
            _append_plain_log(log_path, "Pipeline completed successfully")
            _update_job_status(job_id, status="success", message="Pipeline completed successfully", extra={"exit_code": 0})
        else:
            _append_plain_log(log_path, f"Pipeline exited with code {exit_code}")
            _update_job_status(
                job_id,
                status="error",
                message=f"Pipeline exited with code {exit_code}",
                level="error",
                extra={"exit_code": exit_code},
            )
    except asyncio.CancelledError:
        _append_plain_log(log_path, "Pipeline task cancelled")
        _update_job_status(
            job_id,
            status="cancelled",
            message="Pipeline task cancelled",
            level="warning",
            extra={"cancelled": True},
        )
        raise
    except Exception as exc:  # pragma: no cover - defensive
        if RunPipelineCancelled is not None and isinstance(exc, RunPipelineCancelled):
            _append_plain_log(log_path, "Pipeline cancelled by user")
            _update_job_status(
                job_id,
                status="cancelled",
                message="Pipeline cancelled by user",
                level="warning",
                extra={"cancelled": True},
            )
            return
        if isinstance(exc, RuntimeError) and str(exc) == "Pipeline cancelled":
            _append_plain_log(log_path, "Pipeline cancelled by user")
            _update_job_status(
                job_id,
                status="cancelled",
                message="Pipeline cancelled by user",
                level="warning",
                extra={"cancelled": True},
            )
            return
        _append_plain_log(log_path, f"Pipeline error: {exc}")
        _update_job_status(
            job_id,
            status="error",
            message=f"Unhandled error during pipeline execution: {exc}",
            level="error",
            extra={"error": str(exc)},
        )
    finally:
        JOB_TASKS.pop(job_id, None)
        cancel_event = JOB_CANCEL_EVENTS.pop(job_id, None)
        if cancel_event is not None:
            cancel_event.set()


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


@app.get("/api/version")
def api_version() -> Dict[str, object]:
    dependency_names = _parse_requirement_names()
    return {
        "app": {"title": APP_TITLE},
        "dependencies": _dependency_versions(dependency_names),
        "status": _system_status(),
    }


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
    rows = _apply_manual_overrides(rows)

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


@app.post("/api/manual-status")
def api_manual_status(payload: ManualStatusPayload) -> Dict[str, object]:
    row_id = payload.row_id.strip()
    if not row_id:
        raise HTTPException(400, detail="row_id is required")
    status_value = payload.status.strip().upper()
    if not status_value:
        raise HTTPException(400, detail="status is required")

    overrides = _load_manual_overrides()
    existing = overrides.get(row_id)
    original = payload.original_status.strip().upper() if payload.original_status else None
    if not original and existing:
        original = existing.get("original_status")

    record: Dict[str, object] = {
        "row_id": row_id,
        "status": status_value,
        "updated_at": _now_iso(),
    }
    if original:
        record["original_status"] = original

    _append_manual_override_record(record)
    overrides[row_id] = record
    return {"ok": True, "override": record}


@app.delete("/api/manual-status")
def api_manual_status_delete(row_id: str = Query(..., description="Row identifier")) -> Dict[str, object]:
    row_key = row_id.strip()
    if not row_key:
        raise HTTPException(400, detail="row_id is required")
    overrides = _load_manual_overrides()
    removed = row_key in overrides
    record = {"row_id": row_key, "status": None, "updated_at": _now_iso()}
    _append_manual_override_record(record)
    overrides.pop(row_key, None)
    return {"ok": True, "removed": removed}


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


@app.delete("/api/data")
def api_delete_data() -> Dict[str, object]:
    active_jobs = {
        job_id
        for job_id, task in JOB_TASKS.items()
        if not getattr(task, "done", lambda: True)()
    }

    known_statuses: Dict[str, Optional[Dict[str, Any]]] = {}
    if JOBS_DIR.exists():
        for entry in JOBS_DIR.iterdir():
            if entry.is_dir():
                job_id = entry.name
                known_statuses[job_id] = _load_job_status(job_id)

    active_statuses = {"queued", "running", "cancelling"}
    for job_id, status in known_statuses.items():
        state = str(status.get("status", "") if status else "").lower()
        if state in active_statuses:
            active_jobs.add(job_id)

    removed_files = 0

    for entry in DATA_DIR.iterdir():
        if entry == JOBS_DIR:
            continue
        if entry.name in active_jobs:
            continue
        removed_files += _delete_path(entry)

    if JOBS_DIR.exists():
        for entry in JOBS_DIR.iterdir():
            if entry.name in active_jobs:
                continue
            removed_files += _delete_path(entry)

    if UPLOADS_DIR.exists():
        for entry in UPLOADS_DIR.iterdir():
            if not entry.is_dir():
                continue
            job_id = entry.name
            if job_id in active_jobs:
                continue
            status = known_statuses.get(job_id)
            state = str(status.get("status", "") if status else "").lower()
            if not status or state in active_statuses:
                continue
            removed_files += _delete_path(entry)

    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    return {"removed_files": removed_files}


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
    return _job_status_payload(job_id)


@app.get("/api/process/{job_id}")
def api_process_status(job_id: str) -> Dict[str, Any]:
    return _job_status_payload(job_id)


@app.get("/api/process/{job_id}/logs")
def api_process_logs(job_id: str, offset: int = Query(0, ge=0)) -> StreamingResponse:
    log_path = _job_log_path(job_id)
    if not log_path.exists():
        raise HTTPException(404, detail=f"Logs for job '{job_id}' not found")

    if not isinstance(offset, int):
        offset = getattr(offset, "default", 0)  # Handles direct invocation outside FastAPI

    try:
        offset_value = max(int(offset), 0)
    except (TypeError, ValueError):
        offset_value = 0

    size = log_path.stat().st_size
    start = min(offset_value, size)

    def iterator() -> Iterator[bytes]:
        with log_path.open("rb") as handle:
            if start:
                handle.seek(start)
            while True:
                chunk = handle.read(65536)
                if not chunk:
                    break
                yield chunk

    response = StreamingResponse(iterator(), media_type="text/plain; charset=utf-8")
    response.headers["X-Log-Size"] = str(size)
    response.headers["X-Log-Offset"] = str(start)
    response.headers["X-Log-Path"] = str(log_path)
    return response


@app.delete("/api/process/{job_id}")
async def api_process_cancel(job_id: str) -> JSONResponse:
    status = _load_job_status(job_id)
    if not status:
        raise HTTPException(404, detail=f"Job '{job_id}' not found")

    state = str(status.get("status") or "").lower()
    if state in {"success", "error", "cancelled"}:
        raise HTTPException(409, detail=f"Job '{job_id}' is already finished")

    cancel_event = JOB_CANCEL_EVENTS.get(job_id)
    if cancel_event is None:
        cancel_event = JOB_CANCEL_EVENTS[job_id] = asyncio.Event()

    already_requested = cancel_event.is_set()
    cancel_event.set()

    updated = _update_job_status(
        job_id,
        status="cancelling",
        message="Cancellation requested" if not already_requested else "Cancellation already requested",
        level="warning",
        extra={"cancel_requested": True},
    )

    response = JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": updated.get("status"),
            "cancel_requested": True,
        },
    )
    response.headers["Location"] = f"/api/process/{job_id}"
    return response


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


@app.post("/api/export/json")
def api_export_json(payload: Dict[str, object]):
    if run_export_json is None:
        raise HTTPException(500, detail="export_json is not available in this runtime")

    grid = _resolve_data_path(_coerce_path(payload.get("grid")), "match/reconc_grid.csv")
    out = _resolve_data_path(_coerce_path(payload.get("out")), "match/reconc_grid.json")

    indent_value = payload.get("indent")
    indent = 2
    if indent_value not in (None, ""):
        try:
            indent = int(indent_value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, detail="indent must be an integer") from exc
        if indent < 0:
            indent = 0

    ensure_ascii_value = payload.get("ensure_ascii")
    ensure_ascii = False
    if ensure_ascii_value not in (None, ""):
        if isinstance(ensure_ascii_value, bool):
            ensure_ascii = ensure_ascii_value
        elif isinstance(ensure_ascii_value, (int, float)):
            ensure_ascii = bool(ensure_ascii_value)
        else:
            ensure_ascii = str(ensure_ascii_value).strip().lower() in {"1", "true", "yes", "sim", "on"}

    result = run_export_json(
        grid_csv=str(grid),
        out_path=str(out),
        indent=indent,
        ensure_ascii=ensure_ascii,
    )

    out_path = Path(str(result.get("out", out))).resolve()
    download = _relative_download(out_path)
    if download:
        result["download"] = download
    else:
        result["absolute_out"] = str(out_path)
    return result


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

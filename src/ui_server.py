# ui_server.py — 23/28
# Servidor local (FastAPI) que expõe:
#   - /api/schema        → ui_schema.json
#   - /api/meta          → ui_meta.json
#   - /api/grid          → ui_grid.jsonl (com filtros/ paginação simples)
#   - /api/files         → lista arquivos na pasta de dados
#   - /api/export/xlsx   → chama export_xlsx.py
#   - /api/export/pdf    → chama export_pdf.py
#   - /api/health        → ping
#   - /                  → página simples com links úteis
#
# Observação: upload manual de arquivos é feito fora (por enquanto). Este servidor só lê/serve e dispara exports.
# Use: uvicorn ui_server:app --reload --port 8000
from __future__ import annotations
import os, json, io
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Dependências opcionais (presentes nas etapas anteriores)
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore

# Importa módulos dos arquivos já gerados (opcionais)
try:
    from export_xlsx import run_export as run_export_xlsx  # type: ignore
except Exception:
    run_export_xlsx = None  # type: ignore

try:
    from export_pdf import run_export_pdf  # type: ignore
except Exception:
    run_export_pdf = None  # type: ignore

APP_TITLE = "App de Conferência — Servidor Local"
DATA_DIR = os.environ.get("DATA_DIR", ".")  # por padrão, diretório atual
DATA_DIR = os.path.abspath(DATA_DIR)

app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monta arquivos estáticos para facilitar download (CSV/XLSX/PDF/JSON)
if os.path.isdir(DATA_DIR):
    app.mount("/files", StaticFiles(directory=DATA_DIR), name="files")


def _path(*parts: str) -> str:
    return os.path.join(DATA_DIR, *parts)


def _safe_read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


@app.get("/", response_class=HTMLResponse)
def index():
    html = f"""
    <html><head><meta charset='utf-8'><title>{APP_TITLE}</title>
    <style>
      body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto; padding: 20px; }}
      a {{ color: #2563eb; text-decoration: none; }}
      code {{ background: #f3f4f6; padding: 2px 6px; border-radius: 4px; }}
      .card {{ border: 1px solid #e5e7eb; padding: 16px; border-radius: 8px; margin: 12px 0; }}
    </style></head><body>
      <h2>{APP_TITLE}</h2>
      <div class="card">
        <p><b>DATA_DIR:</b> {DATA_DIR}</p>
        <p>Downloads diretos: <a href="/files/" target="_blank">/files/</a></p>
        <p>APIs: 
          <a href="/api/schema" target="_blank">/api/schema</a> · 
          <a href="/api/meta" target="_blank">/api/meta</a> · 
          <a href="/api/grid?limit=50" target="_blank">/api/grid</a> · 
          <a href="/api/files" target="_blank">/api/files</a> · 
          <a href="/api/health" target="_blank">/api/health</a>
        </p>
        <p>Swagger: <a href="/docs" target="_blank">/docs</a></p>
      </div>
      <div class="card">
        <h3>Exports</h3>
        <p>Excel: <code>POST /api/export/xlsx</code> com JSON:
        <pre>{{"grid":"reconc_grid.csv","sem_fonte":"reconc_sem_fonte.csv","sem_sucessor":"reconc_sem_sucessor.csv","out":"relatorio.xlsx"}}</pre></p>
        <p>PDF: <code>POST /api/export/pdf</code> com JSON:
        <pre>{{"grid":"reconc_grid.csv","out":"relatorio.pdf","cliente":"Nome","periodo":"08/2025"}}</pre></p>
      </div>
    </body></html>
    """
    return HTMLResponse(html)


@app.get("/api/health")
def health():
    return {"ok": True, "data_dir": DATA_DIR}


@app.get("/api/files")
def list_files(pattern: Optional[str] = Query(default=None, description="Filtro simples por sufixo, ex.: .csv, .json, .xlsx, .pdf")):
    if not os.path.isdir(DATA_DIR):
        raise HTTPException(500, detail=f"DATA_DIR não existe: {DATA_DIR}")
    out = []
    for root, _, files in os.walk(DATA_DIR):
        for fn in files:
            if pattern and not fn.lower().endswith(pattern.lower()):
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, DATA_DIR).replace("\\", "/")
            out.append({
                "name": fn,
                "relpath": rel,
                "size": os.path.getsize(full),
                "download": f"/files/{rel}"
            })
    out.sort(key=lambda x: x["relpath"])
    return {"count": len(out), "items": out}


@app.get("/api/schema")
def get_schema():
    p = _path("ui_schema.json")
    data = _safe_read_json(p)
    if data is None:
        # fallback mínimo
        data = {"version": 1, "columns": [{"id":"status","label":"Status"},{"id":"S.doc","label":"Nº Docto (S)"}]}
    return data


@app.get("/api/meta")
def get_meta():
    p = _path("ui_meta.json")
    data = _safe_read_json(p)
    if data is None:
        data = {"columns": [], "presets": [], "legend": [], "stats": {}}
    return data


@app.get("/api/grid")
def get_grid(
    limit: int = Query(200, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None, description="Filtra por status ex.: OK, ALERTA, DIVERGENCIA, SEM_FONTE, SEM_SUCESSOR"),
    fonte_tipo: Optional[str] = Query(None, description="ENTRADA, SAIDA, SERVICO"),
    cfop: Optional[str] = Query(None, description="Filtra por F.cfop"),
    q: Optional[str] = Query(None, description="Busca textual simples em S.historico/S.doc/F.doc/Tags"),
    sort_by: Optional[str] = Query(None, description="Campo para sort ex.: score, delta.valor, S.data"),
    sort_dir: Optional[str] = Query("desc", description="asc/desc"),
):
    p = _path("ui_grid.jsonl")
    if not os.path.exists(p):
        raise HTTPException(404, detail="ui_grid.jsonl não encontrado. Gere com ui_dataset_builder.py")

    # stream e filtra
    rows = []
    total = 0
    q_low = q.lower() if q else None
    for row in _read_jsonl(p):
        total += 1
        if status and str(row.get("status")).upper() != status.upper():
            continue
        if fonte_tipo and str(row.get("fonte_tipo") or "").upper() != fonte_tipo.upper():
            continue
        if cfop and str(row.get("F.cfop") or "") != cfop:
            continue
        if q_low:
            buf = " ".join([
                str(row.get("S.historico") or ""),
                str(row.get("S.doc") or ""),
                str(row.get("F.doc") or ""),
                " ".join([str(x) for x in row.get("tags") or []])
            ]).lower()
            if q_low not in buf:
                continue
        rows.append(row)

    # sort opcional (em memória)
    if sort_by:
        try:
            reverse = (str(sort_dir or "desc").lower() != "asc")
            rows.sort(key=lambda r: (r.get(sort_by) is None, r.get(sort_by)), reverse=reverse)
        except Exception:
            pass

    # paginação
    page = rows[offset:offset+limit]
    return {"total_filtered": len(rows), "returned": len(page), "offset": offset, "limit": limit, "items": page}


@app.post("/api/export/xlsx")
def export_xlsx(payload: Dict[str, Any]):
    if run_export_xlsx is None:
        raise HTTPException(500, detail="export_xlsx não disponível neste ambiente.")
    grid = _path(payload.get("grid", "reconc_grid.csv"))
    sem_fonte = _path(payload.get("sem_fonte", "reconc_sem_fonte.csv"))
    sem_sucessor = _path(payload.get("sem_sucessor", "reconc_sem_sucessor.csv"))
    out = _path(payload.get("out", "relatorio_conferencia.xlsx"))
    try:
        res = run_export_xlsx(grid_csv=grid, sem_fonte_csv=sem_fonte, sem_sucessor_csv=sem_sucessor, out_xlsx=out)
    except Exception as e:
        raise HTTPException(500, detail=f"Falha no export_xlsx: {e}")
    rel = os.path.relpath(res["out"], DATA_DIR).replace("\\", "/")
    res["download"] = f"/files/{rel}"
    return res


@app.post("/api/export/pdf")
def export_pdf(payload: Dict[str, Any]):
    if run_export_pdf is None:
        raise HTTPException(500, detail="export_pdf não disponível neste ambiente.")
    grid = _path(payload.get("grid", "reconc_grid.csv"))
    out = _path(payload.get("out", "relatorio_conferencia.pdf"))
    cliente = payload.get("cliente")
    periodo = payload.get("periodo")
    try:
        res = run_export_pdf(grid_csv=grid, out_pdf=out, cliente=cliente, periodo=periodo)
    except Exception as e:
        raise HTTPException(500, detail=f"Falha no export_pdf: {e}")
    if res.get("pdf"):
        rel = os.path.relpath(res["pdf"], DATA_DIR).replace("\\", "/")
        res["download"] = f"/files/{rel}"
    elif res.get("html"):
        rel = os.path.relpath(res["html"], DATA_DIR).replace("\\", "/")
        res["download_html"] = f"/files/{rel}"
    # gráficos (se existirem)
    charts = res.get("charts") or []
    res["charts_download"] = [f"/files/{os.path.relpath(c, DATA_DIR).replace('\\','/')}" for c in charts if os.path.exists(c)]
    return res

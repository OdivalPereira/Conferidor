# export_xlsx.py — 17/28
# Gera um Excel com 5 abas a partir dos CSVs do reconciler:
#   OK, Alertas, Divergências, Sem Fonte, Sem Sucessor
# Formatação condicional alinhada à UI (verde/amarelo/vermelho/cinza), filtros e freeze panes.
from __future__ import annotations
import argparse, json, sys
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore

def _read_csv(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8")

def _to_float(x) -> Optional[float]:
    if x in (None, ""):
        return None
    try:
        return float(x)
    except Exception:
        s = str(x).strip().replace(".", "").replace(",", ".")
        try:
            return float(s)
        except Exception:
            return None

def _prep_types(df: pd.DataFrame) -> pd.DataFrame:
    # Converte colunas padrão para tipos numéricos (sem explodir em caso de erro)
    num_cols = [c for c in df.columns if c.lower().endswith("valor") or "valor" in c.lower() or c.lower().endswith("score")]
    for c in num_cols:
        try:
            df[c] = df[c].map(_to_float)
        except Exception:
            pass
    return df

def _auto_fit(ws, df: pd.DataFrame, wb):
    # Ajuste simples de largura por tamanho de texto (cap em 60)
    from xlsxwriter.utility import xl_rowcol_to_cell
    max_widths = {}
    # cabeçalhos
    for j, col in enumerate(df.columns):
        max_widths[j] = max(max_widths.get(j, 0), min(60, max(10, len(str(col)) + 2)))
    # amostra de linhas (limita a 500 para performance)
    limit = min(len(df), 500)
    for i in range(limit):
        row = df.iloc[i]
        for j, col in enumerate(df.columns):
            val = row.get(col)
            if val is None:
                continue
            txt = f"{val}"
            max_widths[j] = max(max_widths.get(j, 0), min(60, len(txt) + 2))
    for j, w in max_widths.items():
        ws.set_column(j, j, w)

def _write_sheet(writer: 'pd.ExcelWriter', name: str, df_in: pd.DataFrame, money_cols_guess: List[str]):
    wb = writer.book
    ws = wb.add_worksheet(name)
    writer.sheets[name] = ws

    # formatos
    fmt_header = wb.add_format({"bold": True, "bg_color": "#111827", "font_color": "#ffffff", "border": 0})
    fmt_money = wb.add_format({"num_format": '_-* #,##0.00_-;_-* -#,##0.00_-;_-* "-"_-;_-@_-'})
    fmt_date = wb.add_format({"num_format": "dd/mm/yyyy"})
    fmt_ok = wb.add_format({"bg_color": "#d1fae5"})
    fmt_alerta = wb.add_format({"bg_color": "#fef3c7"})
    fmt_div = wb.add_format({"bg_color": "#fee2e2"})
    fmt_neutro = wb.add_format({"bg_color": "#e5e7eb"})

    # escrever cabeçalho
    for j, col in enumerate(df_in.columns):
        ws.write(0, j, col, fmt_header)

    # escrever linhas
    for i in range(len(df_in)):
        for j, col in enumerate(df_in.columns):
            val = df_in.iat[i, j]
            ws.write(i+1, j, val)

    # aplicar formatos por coluna
    # heurística: colunas com nome que contém 'valor' usam formato monetário
    for j, col in enumerate(df_in.columns):
        if any(tok in col.lower() for tok in ["valor", "amount", "total"]):
            ws.set_column(j, j, None, fmt_money)
        if col in ("S.data", "F.data"):
            ws.set_column(j, j, None, fmt_date)

    # localizar coluna de status para condicional
    status_col_idx = None
    for j, col in enumerate(df_in.columns):
        if col.lower() in ("status", "match.status"):
            status_col_idx = j
            break
    if status_col_idx is not None:
        last_row = len(df_in) + 1
        # texto contendo é robusto e simples
        ws.conditional_format(1, status_col_idx, last_row, status_col_idx, {"type": "text", "criteria": "containing", "value": "OK", "format": fmt_ok})
        ws.conditional_format(1, status_col_idx, last_row, status_col_idx, {"type": "text", "criteria": "containing", "value": "ALERTA", "format": fmt_alerta})
        ws.conditional_format(1, status_col_idx, last_row, status_col_idx, {"type": "text", "criteria": "containing", "value": "DIVERGENCIA", "format": fmt_div})
        ws.conditional_format(1, status_col_idx, last_row, status_col_idx, {"type": "text", "criteria": "containing", "value": "SEM_FONTE", "format": fmt_neutro})
        ws.conditional_format(1, status_col_idx, last_row, status_col_idx, {"type": "text", "criteria": "containing", "value": "SEM_SUCESSOR", "format": fmt_neutro})

    # autofilter e freeze
    ws.autofilter(0, 0, len(df_in), max(len(df_in.columns)-1, 0))
    ws.freeze_panes(1, 0)

    # auto fit
    _auto_fit(ws, df_in, wb)

def _filter_by_status(df: pd.DataFrame, status: str) -> pd.DataFrame:
    col = None
    for c in ["match.status", "status"]:
        if c in df.columns:
            col = c; break
    if col is None:
        return pd.DataFrame(columns=df.columns)
    out = df[df[col] == status].copy()
    return out

def run_export(
    grid_csv: str,
    sem_fonte_csv: Optional[str],
    sem_sucessor_csv: Optional[str],
    out_xlsx: str,
) -> Dict[str, Any]:
    assert pd is not None, "pandas requerido"
    df_grid = _read_csv(grid_csv)
    df_grid = _prep_types(df_grid)

    # se houver CSVs dedicados, use-os; senão derive da grid
    df_sem_fonte = _read_csv(sem_fonte_csv) if sem_fonte_csv else _filter_by_status(df_grid, "SEM_FONTE")
    df_sem_sucessor = _read_csv(sem_sucessor_csv) if sem_sucessor_csv else _filter_by_status(df_grid, "SEM_SUCESSOR")

    df_ok = _filter_by_status(df_grid, "OK")
    df_alerta = _filter_by_status(df_grid, "ALERTA")
    df_div = _filter_by_status(df_grid, "DIVERGENCIA")

    # manter a mesma ordem de colunas da grid para consistência
    cols = list(df_grid.columns)

    # garantir colunas mínimas caso sem_fonte/sem_sucessor vierem de arquivos separados
    def _align(df: pd.DataFrame) -> pd.DataFrame:
        for c in cols:
            if c not in df.columns:
                df[c] = None
        return df[cols].copy()

    df_ok = _align(df_ok)
    df_alerta = _align(df_alerta)
    df_div = _align(df_div)

    # Preparar SEM_FONTE e SEM_SUCESSOR com esquema da grid
    if not df_sem_fonte.empty:
        df_sem_fonte = _align(df_sem_fonte)
        if "match.status" in df_sem_fonte.columns:
            df_sem_fonte["match.status"] = "SEM_FONTE"
        elif "status" in df_sem_fonte.columns:
            df_sem_fonte["status"] = "SEM_FONTE"
    else:
        df_sem_fonte = pd.DataFrame(columns=cols)

    if not df_sem_sucessor.empty:
        df_sem_sucessor = _align(df_sem_sucessor)
        if "match.status" in df_sem_sucessor.columns:
            df_sem_sucessor["match.status"] = "SEM_SUCESSOR"
        elif "status" in df_sem_sucessor.columns:
            df_sem_sucessor["status"] = "SEM_SUCESSOR"
    else:
        df_sem_sucessor = pd.DataFrame(columns=cols)

    # escrever Excel
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        _write_sheet(writer, "OK", df_ok, ["S.valor","F.valor","delta.valor"])
        _write_sheet(writer, "Alertas", df_alerta, ["S.valor","F.valor","delta.valor"])
        _write_sheet(writer, "Divergencias", df_div, ["S.valor","F.valor","delta.valor"])
        _write_sheet(writer, "Sem Fonte", df_sem_fonte, ["S.valor"])
        _write_sheet(writer, "Sem Sucessor", df_sem_sucessor, ["F.valor"])

    return {
        "ok": len(df_ok),
        "alerta": len(df_alerta),
        "divergencia": len(df_div),
        "sem_fonte": len(df_sem_fonte),
        "sem_sucessor": len(df_sem_sucessor),
        "out": out_xlsx,
    }

def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="export_xlsx.py — gera Excel com 5 abas (OK/Alertas/Divergências/Sem Fonte/Sem Sucessor)")
    p.add_argument("--grid", required=True, help="CSV da grade (reconciler --out-grid)")
    p.add_argument("--sem-fonte", help="CSV dos sem fonte (reconciler --out-sem-fonte) — opcional")
    p.add_argument("--sem-sucessor", help="CSV dos sem sucessor (reconciler --out-sem-sucessor) — opcional")
    p.add_argument("--out", required=True, help="Arquivo XLSX de saída")
    args = p.parse_args(argv)

    res = run_export(
        grid_csv=args.grid,
        sem_fonte_csv=args.sem_fonte,
        sem_sucessor_csv=args.sem_sucessor,
        out_xlsx=args.out,
    )
    sys.stdout.write(json.dumps(res, ensure_ascii=False, indent=2) + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

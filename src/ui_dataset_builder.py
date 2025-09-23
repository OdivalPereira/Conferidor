# ui_dataset_builder.py — 16/28
# Constrói dataset para UI (grid) a partir dos CSVs de reconciliação.
# Saídas:
#  - ui_grid.jsonl  : linhas prontas para a grade (um JSON por linha)
#  - ui_meta.json   : metadados (colunas, presets de filtros, legendas, stats)
from __future__ import annotations
import json, sys, re
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore


COLOR_OK = "#16a34a"       # verde
COLOR_ALERTA = "#f59e0b"   # amarelo
COLOR_DIVERG = "#dc2626"   # vermelho
COLOR_NEUTRO = "#6b7280"   # cinza
COLOR_INFO = "#2563eb"     # azul para destaque


def _color(status: Optional[str]) -> str:
    s = (status or "").upper()
    if s == "OK": return COLOR_OK
    if s == "ALERTA": return COLOR_ALERTA
    if s == "DIVERGENCIA": return COLOR_DIVERG
    if s in ("SEM_FONTE","SEM_SUCESSOR"): return COLOR_NEUTRO
    return COLOR_NEUTRO


def _to_num(x) -> Optional[float]:
    if x in (None, ""): return None
    try:
        return float(x)
    except Exception:
        s = str(x).strip().replace(".", "").replace(",", ".")
        try:
            return float(s)
        except Exception:
            return None


def _abs_or_none(x) -> Optional[float]:
    if x in (None, ""): return None
    try:
        return abs(float(x))
    except Exception:
        try:
            return abs(float(str(x).replace(",", ".")))
        except Exception:
            return None


def _norm_text(x: Any) -> Optional[str]:
    if x is None: return None
    s = str(x).strip()
    return re.sub(r"\s+", " ", s) if s else None


def _bool(x: Any) -> bool:
    return str(x).strip().lower() in ("1", "true", "t", "yes", "y", "sim")


def _mk_row_id(i: int, fonte_tipo: Optional[str], s_idx: Optional[int]) -> str:
    # id estável por período/projeto: <S-idx>-<fonte_tipo>
    prefix = "S" + (str(s_idx) if s_idx is not None else "NA")
    ft = (fonte_tipo or "NA")[:3].upper()
    return f"{prefix}-{ft}-{i}"


def build_ui_rows(grid_df: 'pd.DataFrame') -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, r in grid_df.iterrows():
        sidx = r.get("sucessor_idx")
        fonte_tipo = r.get("fonte_tipo")
        status = r.get("match.status")
        cor = _color(status)

        # campos principais (com prefixos S. e F.)
        row = {
            "id": _mk_row_id(i, fonte_tipo, sidx if sidx not in (None, "") else None),
            "status": status,
            "color": cor,
            "strategy": r.get("match.strategy"),
            "score": _to_num(r.get("match.score")),
            "motivos": _norm_text(r.get("match.motivos")),
            "fonte_tipo": fonte_tipo,
            # Sucessor
            "S.data": r.get("S.data"),
            "S.debito": r.get("S.debito"),
            "S.credito": r.get("S.credito"),
            "S.part_d": r.get("S.part_d"),
            "S.part_c": r.get("S.part_c"),
            "S.doc": r.get("S.doc"),
            "S.valor": _to_num(r.get("S.valor")),
            "S.historico": _norm_text(r.get("S.historico")),
            # Fonte
            "F.data": r.get("F.data"),
            "F.doc": r.get("F.doc"),
            "F.participante": r.get("F.participante"),
            "F.valor": _to_num(r.get("F.valor")),
            "F.cfop": r.get("F.cfop"),
            "F.situacao": r.get("F.situacao"),
            # Deltas
            "delta.valor": _to_num(r.get("delta.valor")),
            "delta.dias": _to_num(r.get("delta.dias")),
        }

        # quick-tags para filtros client-side
        tags = []
        if status: tags.append(status)
        if fonte_tipo: tags.append(fonte_tipo)
        if r.get("F.cfop"): tags.append(f"CFOP:{r.get('F.cfop')}")
        if r.get("S.debito"): tags.append(f"DEB:{str(r.get('S.debito'))[:12]}")
        if r.get("S.credito"): tags.append(f"CRE:{str(r.get('S.credito'))[:12]}")
        if r.get("S.doc"): tags.append(f"DOC:{str(r.get('S.doc'))}")
        if r.get("F.situacao"): tags.append(f"SIT:{str(r.get('F.situacao')).upper()}")
        row["tags"] = tags

        rows.append(row)
    return rows


def build_meta(grid_df: 'pd.DataFrame', sem_fonte_df: 'pd.DataFrame', sem_sucessor_df: 'pd.DataFrame') -> Dict[str, Any]:
    # colunas da UI (sugestão TanStack/AG)
    columns = [
        {"id": "status", "label": "Status", "type": "badge", "width": 100},
        {"id": "strategy", "label": "Regra", "type": "text", "width": 80},
        {"id": "score", "label": "Score", "type": "number", "width": 80},
        {"id": "S.data", "label": "Data (S)", "type": "date", "width": 100, "group": "Sucessor"},
        {"id": "S.debito", "label": "Débito", "type": "text", "width": 220, "group": "Sucessor"},
        {"id": "S.credito", "label": "Crédito", "type": "text", "width": 220, "group": "Sucessor"},
        {"id": "S.part_d", "label": "Part. D", "type": "text", "width": 160, "group": "Sucessor"},
        {"id": "S.part_c", "label": "Part. C", "type": "text", "width": 160, "group": "Sucessor"},
        {"id": "S.doc", "label": "Nº Docto (S)", "type": "text", "width": 140, "group": "Sucessor"},
        {"id": "S.valor", "label": "Valor (S)", "type": "money", "width": 120, "group": "Sucessor"},
        {"id": "S.historico", "label": "Histórico", "type": "text", "width": 360, "group": "Sucessor"},
        {"id": "fonte_tipo", "label": "Fonte", "type": "text", "width": 100, "group": "Fonte"},
        {"id": "F.data", "label": "Data (F)", "type": "date", "width": 100, "group": "Fonte"},
        {"id": "F.doc", "label": "Nº Docto (F)", "type": "text", "width": 140, "group": "Fonte"},
        {"id": "F.participante", "label": "Participante (F)", "type": "text", "width": 200, "group": "Fonte"},
        {"id": "F.valor", "label": "Valor (F)", "type": "money", "width": 120, "group": "Fonte"},
        {"id": "F.cfop", "label": "CFOP", "type": "text", "width": 90, "group": "Fonte"},
        {"id": "F.situacao", "label": "Situação", "type": "text", "width": 120, "group": "Fonte"},
        {"id": "delta.valor", "label": "Δ Valor", "type": "money", "width": 120, "group": "Diferenças"},
        {"id": "delta.dias", "label": "Δ Dias", "type": "number", "width": 90, "group": "Diferenças"},
        {"id": "motivos", "label": "Motivos", "type": "text", "width": 360},
        {"id": "tags", "label": "Tags", "type": "array", "width": 260},
    ]

    # presets de filtros úteis
    presets = [
        {"id": "only_ok", "label": "Somente OK", "filters": [{"field": "status", "op": "=", "value": "OK"}]},
        {"id": "only_alerta", "label": "Somente Alertas", "filters": [{"field": "status", "op": "=", "value": "ALERTA"}]},
        {"id": "only_diverg", "label": "Somente Divergências", "filters": [{"field": "status", "op": "=", "value": "DIVERGENCIA"}]},
        {"id": "only_sem_fonte", "label": "Sem Fonte", "filters": [{"field": "status", "op": "=", "value": "SEM_FONTE"}]},
        {"id": "only_sem_sucessor", "label": "Sem Sucessor", "filters": [{"field": "status", "op": "=", "value": "SEM_SUCESSOR"}]},
        {"id": "entrada_diverg", "label": "Divergências (Entradas)", "filters": [{"field": "status", "op": "=", "value": "DIVERGENCIA"}, {"field": "fonte_tipo", "op": "=", "value": "ENTRADA"}]},
        {"id": "saida_diverg", "label": "Divergências (Saídas)", "filters": [{"field": "status", "op": "=", "value": "DIVERGENCIA"}, {"field": "fonte_tipo", "op": "=", "value": "SAIDA"}]},
    ]

    legend = [
        {"status": "OK", "color": COLOR_OK},
        {"status": "ALERTA", "color": COLOR_ALERTA},
        {"status": "DIVERGENCIA", "color": COLOR_DIVERG},
        {"status": "SEM_FONTE/SEM_SUCESSOR", "color": COLOR_NEUTRO},
    ]

    # estatísticas rápidas
    def _count(df: 'pd.DataFrame', val: str) -> int:
        if "match.status" not in df.columns: return 0
        return int((df["match.status"] == val).sum())

    stats = {
        "linhas_grid": len(grid_df),
        "ok": _count(grid_df, "OK"),
        "alerta": _count(grid_df, "ALERTA"),
        "divergencia": _count(grid_df, "DIVERGENCIA"),
        "sem_fonte": len(sem_fonte_df),
        "sem_sucessor": len(sem_sucessor_df),
    }

    return {
        "columns": columns,
        "presets": presets,
        "legend": legend,
        "stats": stats,
        "version": 1,
    }


def write_jsonl(rows: List[Dict[str, Any]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_builder(
    grid_csv: str,
    sem_fonte_csv: str,
    sem_sucessor_csv: str,
    out_jsonl: str,
    out_meta: str,
) -> Dict[str, Any]:
    assert pd is not None, "pandas requerido"
    grid_df = pd.read_csv(grid_csv, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8")
    sem_fonte_df = pd.read_csv(sem_fonte_csv, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8")
    sem_sucessor_df = pd.read_csv(sem_sucessor_csv, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8")

    # construir linhas para UI
    rows = build_ui_rows(grid_df)

    # anexar listas auxiliares como estados especiais
    # Sem Fonte
    for i, r in sem_fonte_df.iterrows():
        rows.append({
            "id": f"SF-{i}",
            "status": "SEM_FONTE",
            "color": _color("SEM_FONTE"),
            "strategy": None,
            "score": None,
            "motivos": None,
            "fonte_tipo": None,
            "S.data": r.get("S.data"),
            "S.debito": r.get("S.debito"),
            "S.credito": r.get("S.credito"),
            "S.part_d": r.get("S.part_d"),
            "S.part_c": r.get("S.part_c"),
            "S.doc": r.get("S.doc"),
            "S.valor": _to_num(r.get("S.valor")),
            "S.historico": _norm_text(r.get("S.historico")),
            "F.data": None,
            "F.doc": None,
            "F.participante": None,
            "F.valor": None,
            "F.cfop": None,
            "F.situacao": None,
            "delta.valor": None,
            "delta.dias": None,
            "tags": ["SEM_FONTE"],
        })

    # Sem Sucessor
    for i, r in sem_sucessor_df.iterrows():
        rows.append({
            "id": f"SS-{i}",
            "status": "SEM_SUCESSOR",
            "color": _color("SEM_SUCESSOR"),
            "strategy": None,
            "score": None,
            "motivos": None,
            "fonte_tipo": r.get("fonte_tipo") or r.get("F.fonte_tipo") or None,
            "S.data": None,
            "S.debito": None,
            "S.credito": None,
            "S.part_d": None,
            "S.part_c": None,
            "S.doc": None,
            "S.valor": None,
            "S.historico": None,
            "F.data": r.get("F.data"),
            "F.doc": r.get("F.doc"),
            "F.participante": r.get("F.participante"),
            "F.valor": _to_num(r.get("F.valor")),
            "F.cfop": r.get("F.cfop"),
            "F.situacao": r.get("F.situacao"),
            "delta.valor": None,
            "delta.dias": None,
            "tags": ["SEM_SUCESSOR", f"CFOP:{r.get('F.cfop')}"] if r.get("F.cfop") else ["SEM_SUCESSOR"],
        })

    # meta
    meta = build_meta(grid_df, sem_fonte_df, sem_sucessor_df)

    # gravar
    write_jsonl(rows, out_jsonl)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {"rows": len(rows), "out_jsonl": out_jsonl, "out_meta": out_meta, "stats": meta.get("stats")}


def main(argv: List[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(description="ui_dataset_builder.py — monta dataset e metadados para a UI")
    p.add_argument("--grid", required=True, help="CSV gerado pelo reconciler (--out-grid)")
    p.add_argument("--sem-fonte", required=True, help="CSV gerado pelo reconciler (--out-sem-fonte)")
    p.add_argument("--sem-sucessor", required=True, help="CSV gerado pelo reconciler (--out-sem-sucessor)")
    p.add_argument("--out-jsonl", default="ui_grid.jsonl")
    p.add_argument("--out-meta", default="ui_meta.json")
    args = p.parse_args(argv)

    res = run_builder(
        grid_csv=args.grid,
        sem_fonte_csv=args.sem_fonte,
        sem_sucessor_csv=args.sem_sucessor,
        out_jsonl=args.out_jsonl,
        out_meta=args.out_meta,
    )
    sys.stdout.write(json.dumps(res, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

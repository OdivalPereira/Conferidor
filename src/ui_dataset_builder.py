from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

STATUS_COLOR = {
    "OK": "#16a34a",
    "ALERTA": "#f59e0b",
    "DIVERGENCIA": "#dc2626",
    "SEM_FONTE": "#64748b",
    "SEM_SUCESSOR": "#64748b",
}

MANUAL_OVERRIDE_TAG = "ajuste_manual"


def read_csv(path: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path, dtype=str, keep_default_na=False, encoding="utf-8")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=[])


def to_number(cell: Optional[str]) -> Optional[float]:
    if cell in (None, ""):
        return None
    try:
        return float(cell)
    except Exception:
        try:
            return float(str(cell).replace(" ", "").replace(".", "").replace(",", "."))
        except Exception:
            return None


def build_base_row(row: pd.Series) -> Dict[str, object]:
    status = str(row.get("match.status") or row.get("status") or "").upper()
    return {
        "status": status,
        "color": STATUS_COLOR.get(status, "#1f2937"),
        "strategy": row.get("match.strategy"),
        "score": to_number(row.get("match.score")),
        "motivos": row.get("match.motivos"),
        "sucessor_idx": row.get("sucessor_idx"),
        "fonte_idx": row.get("fonte_idx"),
        "fonte_tipo": row.get("fonte_tipo"),
        "S.data": row.get("S.data"),
        "S.doc": row.get("S.doc"),
        "S.valor": to_number(row.get("S.valor")),
        "S.part_d": row.get("S.part_d"),
        "S.part_c": row.get("S.part_c"),
        "S.historico": row.get("S.historico"),
        "F.data": row.get("F.data"),
        "F.doc": row.get("F.doc"),
        "F.valor": to_number(row.get("F.valor")),
        "F.cfop": row.get("F.cfop"),
        "F.participante": row.get("F.participante"),
        "F.situacao": row.get("F.situacao"),
        "delta.valor": to_number(row.get("delta.valor")),
        "delta.dias": to_number(row.get("delta.dias")),
    }


def load_manual_overrides(path: Path) -> Dict[str, Dict[str, object]]:
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
                status = data.get("status")
                if status in (None, ""):
                    overrides.pop(row_id, None)
                    continue
                normalized = dict(data)
                normalized["row_id"] = row_id
                normalized["status"] = str(status).upper()
                original = normalized.get("original_status")
                if original:
                    normalized["original_status"] = str(original).upper()
                else:
                    normalized.pop("original_status", None)
                overrides[row_id] = normalized
    except Exception:
        return overrides
    return overrides


def apply_override(row: Dict[str, object], override: Dict[str, object]) -> Dict[str, object]:
    status = str(override.get("status") or "").upper()
    if not status:
        return row
    updated = dict(row)
    original = override.get("original_status")
    if original:
        updated["original_status"] = str(original).upper()
    else:
        previous = str(row.get("original_status") or row.get("status") or "").upper()
        if previous:
            updated["original_status"] = previous
        else:
            updated.pop("original_status", None)
    updated["status"] = status
    updated["match.status"] = status
    updated["color"] = STATUS_COLOR.get(status, updated.get("color"))
    motivos = [part for part in str(updated.get("motivos") or "").split(";") if part]
    if MANUAL_OVERRIDE_TAG not in motivos:
        motivos.append(MANUAL_OVERRIDE_TAG)
    updated["motivos"] = ";".join(motivos)
    updated["_manual"] = True
    return updated


def build_rows(
    grid_df: pd.DataFrame,
    sem_fonte_df: pd.DataFrame,
    sem_sucessor_df: pd.DataFrame,
    overrides: Optional[Dict[str, Dict[str, object]]] = None,
) -> List[Dict[str, object]]:
    overrides = overrides or {}
    rows: List[Dict[str, object]] = []
    for idx, row in grid_df.iterrows():
        base = build_base_row(row)
        base["id"] = f"S{base.get('sucessor_idx', idx)}-{base.get('fonte_tipo','?')}-{base.get('fonte_idx','?')}"
        tags: List[str] = []
        if base.get("status"):
            tags.append(str(base["status"]))
        if base.get("fonte_tipo"):
            tags.append(str(base["fonte_tipo"]))
        if row.get("F.cfop"):
            tags.append(f"CFOP:{row.get('F.cfop')}")
        base["tags"] = tags
        override = overrides.get(base["id"])
        if override:
            base = apply_override(base, override)
        rows.append(base)

    for idx, row in sem_fonte_df.iterrows():
        base = {
            "id": f"SF-{idx}",
            "status": "SEM_FONTE",
            "color": STATUS_COLOR["SEM_FONTE"],
            "strategy": None,
            "score": None,
            "motivos": None,
            "sucessor_idx": row.get("S.row_id"),
            "fonte_idx": None,
            "fonte_tipo": None,
            "S.data": row.get("S.data"),
            "S.doc": row.get("S.doc"),
            "S.valor": to_number(row.get("S.valor")),
            "S.part_d": row.get("S.part_d"),
            "S.part_c": row.get("S.part_c"),
            "S.historico": row.get("S.historico"),
            "F.data": None,
            "F.doc": None,
            "F.valor": None,
            "F.cfop": None,
            "F.participante": None,
            "F.situacao": None,
            "delta.valor": None,
            "delta.dias": None,
            "tags": ["SEM_FONTE"],
        }
        override = overrides.get(base["id"])
        if override:
            base = apply_override(base, override)
        rows.append(base)

    for idx, row in sem_sucessor_df.iterrows():
        cfop = row.get("F.cfop")
        tags = ["SEM_SUCESSOR"]
        if cfop:
            tags.append(f"CFOP:{cfop}")
        rows.append(
            {
                "id": f"SS-{idx}",
                "status": "SEM_SUCESSOR",
                "color": STATUS_COLOR["SEM_SUCESSOR"],
                "strategy": None,
                "score": None,
                "motivos": None,
                "sucessor_idx": None,
                "fonte_idx": row.get("F.row_id"),
                "fonte_tipo": row.get("fonte_tipo"),
                "S.data": None,
                "S.doc": None,
                "S.valor": None,
                "S.part_d": None,
                "S.part_c": None,
                "S.historico": None,
                "F.data": row.get("F.data"),
                "F.doc": row.get("F.doc"),
                "F.valor": to_number(row.get("F.valor")),
                "F.cfop": cfop,
                "F.participante": row.get("F.participante"),
                "F.situacao": row.get("F.situacao"),
                "delta.valor": None,
                "delta.dias": None,
                "tags": tags,
            }
        )
        override = overrides.get(rows[-1]["id"])
        if override:
            rows[-1] = apply_override(rows[-1], override)
    return rows


def compute_stats(grid_df: pd.DataFrame, sem_fonte_df: pd.DataFrame, sem_sucessor_df: pd.DataFrame) -> Dict[str, object]:
    status_col = "match.status" if "match.status" in grid_df.columns else "status"
    counts = grid_df[status_col].str.upper().value_counts() if not grid_df.empty else {}
    return {
        "total_matches": int(len(grid_df)),
        "ok": int(counts.get("OK", 0)),
        "alerta": int(counts.get("ALERTA", 0)),
        "divergencia": int(counts.get("DIVERGENCIA", 0)),
        "sem_fonte": int(len(sem_fonte_df)),
        "sem_sucessor": int(len(sem_sucessor_df)),
    }


def load_schema(path: Optional[str]) -> Optional[Dict[str, object]]:
    if not path:
        return None
    schema_path = Path(path)
    if not schema_path.exists():
        return None
    return json.loads(schema_path.read_text(encoding="utf-8"))


def write_jsonl(rows: List[Dict[str, object]], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_meta(schema: Optional[Dict[str, object]], stats: Dict[str, object]) -> Dict[str, object]:
    meta = {
        "version": 1,
        "stats": stats,
        "columns": [],
        "legend": [
            {"status": "OK", "color": STATUS_COLOR["OK"], "label": "Correspond\u00eancia confirmada"},
            {"status": "ALERTA", "color": STATUS_COLOR["ALERTA"], "label": "Precisa de revis\u00e3o"},
            {"status": "DIVERGENCIA", "color": STATUS_COLOR["DIVERGENCIA"], "label": "Conferir detalhes"},
            {"status": "SEM_FONTE", "color": STATUS_COLOR["SEM_FONTE"], "label": "Sem fonte correspondente"},
            {"status": "SEM_SUCESSOR", "color": STATUS_COLOR["SEM_SUCESSOR"], "label": "Sem Sucessor correspondente"},
        ],
        "presets": [
            {"name": "Apenas Diverg\u00eancias", "filters": {"status": ["DIVERGENCIA"]}},
            {"name": "Sem Fonte", "filters": {"status": ["SEM_FONTE"]}},
        ],
    }
    if schema and schema.get("columns"):
        meta["columns"] = schema["columns"]
    else:
        meta["columns"] = [
            {"id": "status", "label": "Status"},
            {"id": "match.score", "label": "Score"},
            {"id": "match.strategy", "label": "Estrat\u00e9gia"},
            {"id": "S.doc", "label": "Documento (S)"},
            {"id": "F.doc", "label": "Documento (F)"},
            {"id": "delta.valor", "label": "Delta Valor"},
            {"id": "delta.dias", "label": "Delta Dias"},
        ]
    return meta


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Builds UI dataset (JSONL + metadata)")
    parser.add_argument("--grid", required=True)
    parser.add_argument("--sem-fonte", required=True)
    parser.add_argument("--sem-sucessor", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--schema")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    grid_df = read_csv(args.grid)
    sem_fonte_df = read_csv(args.sem_fonte)
    sem_sucessor_df = read_csv(args.sem_sucessor)

    overrides_path = Path(args.out_jsonl).parent / "manual_overrides.jsonl"
    overrides = load_manual_overrides(overrides_path)

    rows = build_rows(grid_df, sem_fonte_df, sem_sucessor_df, overrides)
    write_jsonl(rows, Path(args.out_jsonl))

    schema = load_schema(args.schema)
    stats = compute_stats(grid_df, sem_fonte_df, sem_sucessor_df)
    meta = build_meta(schema, stats)
    ensure_dir(Path(args.meta).parent)
    Path(args.meta).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {"rows": len(rows), "meta": args.meta, "jsonl": args.out_jsonl}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

# reconciler.py — 15/28
# Consolidação pós-matching: junta Sucessor × Fonte vencedora,
# calcula deltas/status/cores, monta CSV de grade p/ UI e listas de não casados.
from __future__ import annotations
import json, sys
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore


def pick_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_num(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        s = str(x).strip().replace(".", "").replace(",", ".")
        try:
            return float(s)
        except Exception:
            return None


def to_date(x) -> Optional[pd.Timestamp]:
    if x is None or x == "":
        return None
    try:
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        return None


def abs_diff_days(a: Optional[pd.Timestamp], b: Optional[pd.Timestamp]) -> Optional[int]:
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return None
    return abs((a - b).days)


def status_color(status: str) -> str:
    s = (status or "").upper()
    if s == "OK":
        return "#16a34a"  # verde
    if s == "ALERTA":
        return "#f59e0b"  # amarelo
    if s == "DIVERGENCIA":
        return "#dc2626"  # vermelho
    return "#6b7280"      # cinza


def load_csv(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8")


def build_row(srow: pd.Series, frow: Optional[pd.Series], winner: Optional[Dict[str, Any]], fonte_tipo: Optional[str]) -> Dict[str, Any]:
    # Colunas preferenciais para exibição
    # Sucessor
    S_data = srow.get(pick_first(pd.DataFrame([srow]).astype(str), ["data_iso","data","data_doc","data_emissao","data_lanc"]))
    S_deb = srow.get(pick_first(pd.DataFrame([srow]).astype(str), ["debito_alias","debito_nome","debito","debito_cod"]))
    S_cre = srow.get(pick_first(pd.DataFrame([srow]).astype(str), ["credito_alias","credito_nome","credito","credito_cod"]))
    S_part_d = srow.get(pick_first(pd.DataFrame([srow]).astype(str), ["part_d","participante_d","fornecedor_cod","cliente_cod"]))
    S_part_c = srow.get(pick_first(pd.DataFrame([srow]).astype(str), ["part_c","participante_c"]))
    S_val = srow.get(pick_first(pd.DataFrame([srow]).astype(str), ["valor_num","valor","valor_contabil"]))
    S_doc = srow.get(pick_first(pd.DataFrame([srow]).astype(str), ["doc_num_norm","documento","numero_docto","num_doc"]))
    S_hist = srow.get(pick_first(pd.DataFrame([srow]).astype(str), ["historico","hist","descricao","complemento"]))

    # Fonte (se houver)
    if frow is not None:
        F_data = frow.get(pick_first(pd.DataFrame([frow]).astype(str), ["data_iso","data_emissao","data_entrada","data"]))
        F_val = frow.get(pick_first(pd.DataFrame([frow]).astype(str), ["valor_num","valor_contabil","valor"]))
        F_doc = frow.get(pick_first(pd.DataFrame([frow]).astype(str), ["doc_num_norm","documento","numero_docto","num_doc"]))
        F_part = frow.get(pick_first(pd.DataFrame([frow]).astype(str), ["fornecedor_cod","cliente_cod","participante_cod","participante"]))
        F_cfop = frow.get(pick_first(pd.DataFrame([frow]).astype(str), ["cfop"]))
        F_situ = frow.get(pick_first(pd.DataFrame([frow]).astype(str), ["situacao"]))
    else:
        F_data = F_val = F_doc = F_part = F_cfop = F_situ = None

    # Deltas
    d_val = None
    s_num = to_num(S_val)
    f_num = to_num(F_val) if F_val is not None else None
    if s_num is not None and f_num is not None:
        d_val = round(s_num - f_num, 2)

    d_dias = None
    sd = to_date(S_data)
    fd = to_date(F_data) if F_data is not None else None
    if sd is not None and fd is not None:
        d_dias = abs_diff_days(sd, fd)

    # Winner info
    strategy = winner.get("strategy") if winner else None
    score = int(winner.get("score")) if winner and winner.get("score") is not None else None
    status = winner.get("status") if winner else "SEM_FONTE"
    motivos = winner.get("motivos") if winner else ""

    return {
        # Sucessor
        "sucessor_idx": int(winner.get("sucessor_idx")) if winner and winner.get("sucessor_idx") is not None else None,
        "S.data": S_data, "S.debito": S_deb, "S.credito": S_cre,
        "S.part_d": S_part_d, "S.part_c": S_part_c,
        "S.doc": S_doc, "S.valor": S_val, "S.historico": S_hist,
        # Fonte
        "fonte_tipo": fonte_tipo,
        "F.data": F_data, "F.doc": F_doc, "F.participante": F_part,
        "F.valor": F_val, "F.cfop": F_cfop, "F.situacao": F_situ,
        # Match
        "match.strategy": strategy, "match.score": score, "match.status": status,
        "match.motivos": motivos,
        # Diffs
        "delta.valor": d_val, "delta.dias": d_dias,
        # Cores
        "color": status_color(status),
    }


def run_reconciler(
    sucessor_csv: str,
    entradas_csv: Optional[str],
    saidas_csv: Optional[str],
    servicos_csv: Optional[str],
    matches_csv: str,
    out_grid_csv: str,
    out_sem_fonte_csv: str,
    out_sem_sucessor_csv: str,
) -> Dict[str, Any]:
    assert pd is not None, "pandas requerido"
    df_s = load_csv(sucessor_csv)
    df_ent = load_csv(entradas_csv) if entradas_csv else pd.DataFrame()
    df_sai = load_csv(saidas_csv) if saidas_csv else pd.DataFrame()
    df_srv = load_csv(servicos_csv) if servicos_csv else pd.DataFrame()
    df_m = load_csv(matches_csv)

    fonte_map = {"ENTRADA": df_ent, "SAIDA": df_sai, "SERVICO": df_srv}

    # Linha por linha (winners) → grid
    grid_rows: List[Dict[str, Any]] = []
    used_fonte = set()  # (tipo, idx)

    for _, w in df_m.iterrows():
        try:
            s_idx = int(w.get("sucessor_idx"))
            f_idx = int(w.get("fonte_idx")) if w.get("fonte_idx") not in (None, "") else None
            f_tipo = str(w.get("fonte_tipo") or "")
        except Exception:
            continue

        srow = df_s.iloc[s_idx] if 0 <= s_idx < len(df_s) else None
        fdf = fonte_map.get(f_tipo, pd.DataFrame())
        frow = fdf.iloc[f_idx] if (f_idx is not None and 0 <= f_idx < len(fdf)) else None

        if srow is None:
            continue

        winner = {k: w.get(k) for k in ["sucessor_idx","fonte_idx","fonte_tipo","strategy","score","status","motivos"]}
        row = build_row(srow, frow, winner, f_tipo if frow is not None else None)
        grid_rows.append(row)

        if frow is not None:
            used_fonte.add((f_tipo, f_idx))

    # Marcar SEM_FONTE (linhas do Sucessor sem winner)
    winners_idx = set(int(x) for x in df_m["sucessor_idx"].dropna().astype(int).tolist()) if "sucessor_idx" in df_m.columns else set()
    sem_fonte_rows: List[Dict[str, Any]] = []
    for i in range(len(df_s)):
        if i not in winners_idx:
            srow = df_s.iloc[i]
            row = build_row(srow, None, None, None)
            sem_fonte_rows.append(row)

    # Marcar SEM_SUCESSOR (linhas da fonte não usadas)
    sem_sucessor_rows: List[Dict[str, Any]] = []
    for tipo, df_f in fonte_map.items():
        if df_f.empty:
            continue
        for i in range(len(df_f)):
            if (tipo, i) not in used_fonte:
                # build minimal row with only fonte side
                frow = df_f.iloc[i]
                dummy_win = {"sucessor_idx": None, "strategy": None, "score": None, "status": "SEM_SUCESSOR", "motivos": ""}
                row = build_row(pd.Series(dtype=object), frow, dummy_win, tipo)
                sem_sucessor_rows.append(row)

    # Persistir
    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(out_grid_csv, index=False, encoding="utf-8")

    sem_fonte_df = pd.DataFrame(sem_fonte_rows)
    sem_fonte_df.to_csv(out_sem_fonte_csv, index=False, encoding="utf-8")

    sem_sucessor_df = pd.DataFrame(sem_sucessor_rows)
    sem_sucessor_df.to_csv(out_sem_sucessor_csv, index=False, encoding="utf-8")

    # Métricas
    kpi = {
        "linhas_sucessor": len(df_s),
        "winners": len(grid_df),
        "percent_cobertura": round(100.0 * (len(grid_df) / max(len(df_s), 1)), 2),
        "sem_fonte": len(sem_fonte_df),
        "sem_sucessor": len(sem_sucessor_df),
        "out_grid_csv": out_grid_csv,
        "out_sem_fonte_csv": out_sem_fonte_csv,
        "out_sem_sucessor_csv": out_sem_sucessor_csv,
    }
    return kpi


def main(argv: List[str]) -> int:
    import argparse, json
    p = argparse.ArgumentParser(description="reconciler.py — consolida winners em grade e gera listas de não casados")
    p.add_argument("--sucessor", required=True)
    p.add_argument("--entradas")
    p.add_argument("--saidas")
    p.add_argument("--servicos")
    p.add_argument("--matches", required=True)
    p.add_argument("--out-grid", required=True)
    p.add_argument("--out-sem-fonte", required=True)
    p.add_argument("--out-sem-sucessor", required=True)
    args = p.parse_args(argv)

    res = run_reconciler(
        sucessor_csv=args.sucessor,
        entradas_csv=args.entradas,
        saidas_csv=args.saidas,
        servicos_csv=args.servicos,
        matches_csv=args.matches,
        out_grid_csv=args.out_grid,
        out_sem_fonte_csv=args.out_sem_fonte,
        out_sem_sucessor_csv=args.out_sem_sucessor,
    )
    sys.stdout.write(json.dumps(res, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

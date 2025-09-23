# matcher.py — 14/28
# Motor de conferência (S1–S5): Sucessor × (Entradas | Saídas | Serviços)
# - Gera candidatos por cascata e classifica OK/ALERTA/DIVERGENCIA
# - Usa pesos/regex externos (YAML) ou defaults internos
from pathlib import Path
import json, re, math, sys
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore

# Tentativa opcional de YAML; se indisponível, cai em defaults
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore


# ------------------ Config defaults (se YAMLs não forem carregados) ------------------
DEFAULT_WEIGHTS = {
    "limiares": {"auto_match": 70, "pendente_min": 50, "max_candidates": 6, "janela_default_dias": 3},
    "estrategias": {"S1": 50, "S2": 35, "S3": 25, "S4": 20, "S5": 10, "S6": 0},
    "bonus": {
        "participante_exato": 10,
        "participante_nome_fuzzy_alto": 4,
        "valor_exato_centavos": 6,
        "valor_dentro_tolerancia": 3,
        "data_mesma": 4,
        "data_ate_1_dia": 6,
        "data_ate_3_dias": 3,
        "cfop_contas_coerente": 5,
        "token_forte": 4,
        "mesmo_mes_ref": 2,
        "fonte_prioritaria": 2,
    },
    "penalidades": {
        "participante_conflito": -15,
        "cfop_contas_conflito": -10,
        "documento_cancelado": -25,
        "valor_fora_tolerancia": -40,
        "data_fora_janela": -20,
        "duplicado_mesmo_documento": -8,
        "modelo_inconsistente": -4,
        "situacao_irregular": -6,
    },
    "desempate": {
        "ordem": ["delta_valor_abs", "delta_dias_abs", "prioridade_fonte", "score"],
        "pesos": {"delta_valor_abs": 0.70, "delta_dias_abs": 0.20, "prioridade_fonte": 0.10},
        "prioridade_fonte": {"ENTRADA": 1.0, "SAIDA": 1.0, "SERVICO": 0.9},
    },
}

DEFAULT_REGEX = {
    "TOKENS": {
        "NFE_NUM_SERIE": r"\b(?:NF|NFE)\s*(\d{1,9})\s*[\/\-]\s*(\d{1,4})\b",
        "NFSE_RPS": r"\bRPS\s*(\d{1,10})\b",
        "DCTO": r"\bDCTO\s*(\d{1,15})\b",
        "PIX": r"\b(E2E|TXID)\s*[:\-]?\s*([A-Z0-9]{10,36})\b",
        "CARTAO_ADQ": r"\b(NSU|CIELO|REDE|GETNET|STONE)\s*[:\-]?\s*([A-Z0-9]{4,12})\b",
    }
}


def load_yaml_safe(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    try:
        text = Path(path).read_text(encoding="utf-8")
    except Exception:
        return None
    if yaml is None:
        return None
    try:
        return yaml.safe_load(text)
    except Exception:
        return None


# ------------------ Helpers ------------------
def coalesce(*vals):
    for v in vals:
        if v is not None and v != "":
            return v
    return None


def to_num(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        # tenta remover milhar brasileiro
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


def within_tol(val_a: Optional[float], val_b: Optional[float], tol_abs: float, tol_pct: float) -> Tuple[bool, Optional[float]]:
    if val_a is None or val_b is None:
        return False, None
    delta = val_a - val_b
    if abs(delta) <= tol_abs:
        return True, delta
    base = max(abs(val_a), abs(val_b), 1e-9)
    if abs(delta) <= base * tol_pct:
        return True, delta
    return False, delta


def norm_participante(row: Dict[str, Any], prefer: List[str]) -> Optional[str]:
    for k in prefer:
        v = row.get(k)
        if v is not None and str(v).strip() != "":
            return str(v).strip().upper()
    return None


def pick_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def extract_tokens(text: str, regexes: Dict[str, str]) -> Dict[str, List[str]]:
    found = {}
    if not text:
        return found
    for name, pat in regexes.items():
        try:
            rx = re.compile(pat, flags=re.IGNORECASE)
            matches = rx.findall(text)
            if matches:
                if isinstance(matches[0], tuple):
                    vals = ["|".join([str(m) for m in tup]) for tup in matches]
                else:
                    vals = [str(m) for m in matches]
                found[name] = vals
        except Exception:
            continue
    return found


# ------------------ Core matching ------------------
def gen_candidates(
    df_s: pd.DataFrame,
    df_f: pd.DataFrame,
    fonte_tipo: str,
    pesos: Dict[str, Any],
    regex_cfg: Dict[str, Any],
    janela_dias: int,
    tol_abs: float,
    tol_pct: float,
    cfop_expect: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    """Gera candidatos S1..S5 para uma fonte específica e o Sucessor."""
    if df_s.empty or df_f.empty:
        return []

    # Mapear colunas de interesse
    col_s_data = pick_first(df_s, ["data_iso", "data", "data_doc", "data_emissao", "data_lanc"])
    col_s_val = pick_first(df_s, ["valor_num", "valor", "valor_contabil"])
    col_s_doc = pick_first(df_s, ["doc_num_norm", "numero_docto", "documento_norm", "documento", "num_doc"])
    col_s_hist = pick_first(df_s, ["historico", "hist", "complemento", "descricao"])
    col_s_debito_alias = pick_first(df_s, ["debito_alias", "debito_nome", "debito_conta_alias"])
    col_s_credito_alias = pick_first(df_s, ["credito_alias", "credito_nome", "credito_conta_alias"])
    # participante: considerar vários campos
    s_part_cols = [c for c in ["participante_cod", "participante", "part_d", "part_c", "participante_d", "participante_c", "fornecedor_cod", "cliente_cod"] if c in df_s.columns]

    col_f_data = pick_first(df_f, ["data_iso", "data_emissao", "data_entrada", "data"])
    col_f_val = pick_first(df_f, ["valor_num", "valor_contabil", "valor"])
    col_f_doc = pick_first(df_f, ["doc_num_norm", "documento", "numero_docto", "num_doc"])
    col_f_hist = pick_first(df_f, ["historico", "descricao", "complemento"])
    col_f_cfop = pick_first(df_f, ["cfop"])
    f_part_cols = []
    if fonte_tipo == "ENTRADA":
        f_part_cols = [c for c in ["fornecedor_cod", "fornecedor", "participante_cod", "participante"] if c in df_f.columns]
    elif fonte_tipo == "SAIDA":
        f_part_cols = [c for c in ["cliente_cod", "cliente", "participante_cod", "participante"] if c in df_f.columns]
    else:
        f_part_cols = [c for c in ["participante_cod", "participante", "cliente_cod", "fornecedor_cod"] if c in df_f.columns]

    regex_tok = (regex_cfg or DEFAULT_REGEX).get("TOKENS", DEFAULT_REGEX["TOKENS"])

    # Index por documento quando existir
    f_by_doc = {}
    if col_f_doc:
        for i, r in df_f.iterrows():
            key = str(r.get(col_f_doc) or "").strip()
            if key != "":
                f_by_doc.setdefault(key, []).append(i)

    cands: List[Dict[str, Any]] = []

    for i_s, rs in df_s.iterrows():
        s_doc = str(rs.get(col_s_doc) or "").strip() if col_s_doc else ""
        s_val = to_num(rs.get(col_s_val))
        s_data = to_date(rs.get(col_s_data))
        s_part = norm_participante(rs, s_part_cols)

        s_hist = str(rs.get(col_s_hist) or "")
        s_tokens = extract_tokens(s_hist, regex_tok) if col_s_hist else {}

        # Conjuntos de busca
        idx_candidates: List[int] = []
        if s_doc and s_doc in f_by_doc:
            idx_candidates += f_by_doc[s_doc]
        else:
            # fallback por janela de datas + valor aproximado + participante
            for i_f, rf in df_f.iterrows():
                f_data = to_date(rf.get(col_f_data))
                f_val = to_num(rf.get(col_f_val))
                ok_val, _ = within_tol(s_val, f_val, tol_abs, tol_pct)
                if not ok_val:
                    continue
                dd = abs_diff_days(s_data, f_data)
                if dd is not None and dd <= janela_dias:
                    idx_candidates.append(i_f)

        seen = set()
        for i_f in idx_candidates:
            if i_f in seen:
                continue
            seen.add(i_f)
            rf = df_f.loc[i_f]

            f_doc = str(rf.get(col_f_doc) or "").strip() if col_f_doc else ""
            f_val = to_num(rf.get(col_f_val))
            f_data = to_date(rf.get(col_f_data))
            f_part = norm_participante(rf, f_part_cols)
            f_hist = str(rf.get(col_f_hist) or "")
            f_tokens = extract_tokens(f_hist, regex_tok) if col_f_hist else {}
            f_cfop = str(rf.get(col_f_cfop) or "").strip() if col_f_cfop else ""

            # Estratégia base
            strategy = None
            base = 0
            motivos = []

            # S1: Doc + Part + Valor
            ok_s1 = False
            if s_doc and f_doc and s_doc == f_doc and s_part and f_part and s_part == f_part:
                ok_val, delta_val = within_tol(s_val, f_val, tol_abs, tol_pct)
                if ok_val:
                    ok_s1 = True
                    strategy = "S1"
                    base = pesos["estrategias"].get("S1", DEFAULT_WEIGHTS["estrategias"]["S1"])
                    motivos.append("S1: doc+participante+valor")

            # S2: Doc + Part + Data janela
            if strategy is None and s_doc and f_doc and s_doc == f_doc and s_part and f_part and s_part == f_part:
                dd = abs_diff_days(s_data, f_data)
                if dd is not None and dd <= janela_dias:
                    strategy = "S2"
                    base = pesos["estrategias"].get("S2", DEFAULT_WEIGHTS["estrategias"]["S2"])
                    motivos.append(f"S2: doc+participante+data(±{janela_dias})")

            # S3: Doc + Valor + Data
            if strategy is None and s_doc and f_doc and s_doc == f_doc:
                ok_val, delta_val = within_tol(s_val, f_val, tol_abs, tol_pct)
                dd = abs_diff_days(s_data, f_data)
                if ok_val and (dd is not None and dd <= janela_dias):
                    strategy = "S3"
                    base = pesos["estrategias"].get("S3", DEFAULT_WEIGHTS["estrategias"]["S3"])
                    motivos.append("S3: doc+valor+data")

            # S4: Valor + Part + Data
            if strategy is None and s_part and f_part and s_part == f_part:
                ok_val, delta_val = within_tol(s_val, f_val, tol_abs, tol_pct)
                dd = abs_diff_days(s_data, f_data)
                if ok_val and (dd is not None and dd <= janela_dias):
                    strategy = "S4"
                    base = pesos["estrategias"].get("S4", DEFAULT_WEIGHTS["estrategias"]["S4"])
                    motivos.append("S4: valor+participante+data")

            # S5: Pistas (tokens) — bônus auxiliar
            token_bonus = 0
            if s_tokens and f_tokens:
                common_keys = set(s_tokens.keys()) & set(f_tokens.keys())
                if common_keys:
                    token_bonus = pesos["bonus"].get("token_forte", DEFAULT_WEIGHTS["bonus"]["token_forte"])
                    motivos.append("token_forte")

            # Se nenhuma estratégia acionou, mas tokens existem e valor/data batem, classificar como S5
            if strategy is None and token_bonus > 0:
                ok_val, delta_val = within_tol(s_val, f_val, tol_abs, tol_pct)
                dd = abs_diff_days(s_data, f_data)
                if ok_val and (dd is not None and dd <= janela_dias):
                    strategy = "S5"
                    base = pesos["estrategias"].get("S5", DEFAULT_WEIGHTS["estrategias"]["S5"])
                    motivos.append("S5: tokens+valor+data")

            if strategy is None:
                continue  # candidato fraco demais

            score = base

            # Bônus/penalidades
            if s_part and f_part and s_part == f_part:
                score += pesos["bonus"].get("participante_exato", DEFAULT_WEIGHTS["bonus"]["participante_exato"])
            else:
                score += pesos["penalidades"].get("participante_conflito", DEFAULT_WEIGHTS["penalidades"]["participante_conflito"])

            ok_val, delta_val = within_tol(s_val, f_val, tol_abs, tol_pct)
            if ok_val:
                # valor exato até centavos?
                if abs((s_val or 0) - (f_val or 0)) < 0.005:
                    score += pesos["bonus"].get("valor_exato_centavos", DEFAULT_WEIGHTS["bonus"]["valor_exato_centavos"])
                else:
                    score += pesos["bonus"].get("valor_dentro_tolerancia", DEFAULT_WEIGHTS["bonus"]["valor_dentro_tolerancia"])
            else:
                score += pesos["penalidades"].get("valor_fora_tolerancia", DEFAULT_WEIGHTS["penalidades"]["valor_fora_tolerancia"])

            dd = abs_diff_days(s_data, f_data)
            delta_dias = dd if dd is not None else 999
            if dd is None or dd > janela_dias:
                score += pesos["penalidades"].get("data_fora_janela", DEFAULT_WEIGHTS["penalidades"]["data_fora_janela"])
            else:
                if dd == 0:
                    score += pesos["bonus"].get("data_mesma", DEFAULT_WEIGHTS["bonus"]["data_mesma"])
                elif dd <= 1:
                    score += pesos["bonus"].get("data_ate_1_dia", DEFAULT_WEIGHTS["bonus"]["data_ate_1_dia"])
                elif dd <= 3:
                    score += pesos["bonus"].get("data_ate_3_dias", DEFAULT_WEIGHTS["bonus"]["data_ate_3_dias"])

            score += token_bonus

            # CFOP × contas (se disponível)
            coh_cfop = None
            if cfop_expect is not None and f_cfop:
                exp = cfop_expect[cfop_expect["cfop"] == str(f_cfop)]
                if not exp.empty:
                    # ver se alias de débito/crédito (no Sucessor) batem pelo menos parcialmente
                    deb_alias = str(rs.get(col_s_debito_alias) or "").upper()
                    cre_alias = str(rs.get(col_s_credito_alias) or "").upper()
                    deb_ok, cre_ok = False, False
                    try:
                        deb_sug = str(exp.iloc[0]["debito_sugerido"] or "")
                        cre_sug = str(exp.iloc[0]["credito_sugerido"] or "")
                        deb_ok = any(a.strip() and a.strip().upper() in deb_alias for a in deb_sug.split(";"))
                        cre_ok = any(a.strip() and a.strip().upper() in cre_alias for a in cre_sug.split(";"))
                    except Exception:
                        pass
                    if deb_ok or cre_ok:
                        score += pesos["bonus"].get("cfop_contas_coerente", DEFAULT_WEIGHTS["bonus"]["cfop_contas_coerente"])
                        coh_cfop = True
                    else:
                        score += pesos["penalidades"].get("cfop_contas_conflito", DEFAULT_WEIGHTS["penalidades"]["cfop_contas_conflito"])
                        coh_cfop = False

            cands.append({
                "sucessor_idx": i_s,
                "fonte_tipo": fonte_tipo,
                "fonte_idx": i_f,
                "strategy": strategy,
                "score": int(score),
                "delta_valor": float((s_val or 0) - (f_val or 0)) if (s_val is not None and f_val is not None) else None,
                "delta_dias": int(delta_dias) if delta_dias is not None else None,
                "coerencia_cfop": coh_cfop,
                "motivos": ";".join(motivos),
            })

    return cands


def pick_best(cands: List[Dict[str, Any]], pesos: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Resolve 1:1: escolhe o melhor candidato por linha do Sucessor e evita múltiplos winners para a mesma fonte."""
    if not cands:
        return [], []

    # Ordenar por score desc, depois por delta_valor_abs, delta_dias_abs, prioridade_fonte
    prio = pesos.get("desempate", DEFAULT_WEIGHTS["desempate"]).get("prioridade_fonte", DEFAULT_WEIGHTS["desempate"]["prioridade_fonte"])

    def key_fn(c):
        dv = abs(c.get("delta_valor") or 1e9)
        dd = abs(c.get("delta_dias") or 999)
        pf = {"ENTRADA": prio.get("ENTRADA", 1.0), "SAIDA": prio.get("SAIDA", 1.0), "SERVICO": prio.get("SERVICO", 0.9)}.get(c["fonte_tipo"], 0.8)
        return (-c["score"], dv, dd, -pf)

    sorted_c = sorted(cands, key=key_fn)

    used_f = set()
    best_by_s: Dict[int, Dict[str, Any]] = {}

    for c in sorted_c:
        s = c["sucessor_idx"]
        f_key = (c["fonte_tipo"], c["fonte_idx"])
        if f_key in used_f:
            continue
        if s in best_by_s:
            continue
        best_by_s[s] = c
        used_f.add(f_key)

    winners = list(best_by_s.values())

    # Build losers list (remaining candidates of chosen sucessor lines)
    losers = [c for c in cands if c["sucessor_idx"] in best_by_s and c is not best_by_s[c["sucessor_idx"]]]
    return winners, losers


def classify_rows(winners: List[Dict[str, Any]], limiares: Dict[str, Any]) -> None:
    auto_t = limiares.get("auto_match", DEFAULT_WEIGHTS["limiares"]["auto_match"])
    pend_t = limiares.get("pendente_min", DEFAULT_WEIGHTS["limiares"]["pendente_min"])
    for w in winners:
        if w["score"] >= auto_t:
            w["status"] = "OK"
        elif w["score"] >= pend_t:
            w["status"] = "ALERTA"
        else:
            w["status"] = "DIVERGENCIA"


def run_matcher(
    sucessor_csv: str,
    entradas_csv: Optional[str],
    saidas_csv: Optional[str],
    servicos_csv: Optional[str],
    pesos_yml: Optional[str],
    regex_yml: Optional[str],
    cfop_exp_csv: Optional[str],
    out_candidates_csv: str,
    out_matches_csv: str,
    janela_dias: int = 3,
    tol_abs: float = 0.01,
    tol_pct: float = 0.002,
) -> Dict[str, Any]:
    assert pd is not None, "pandas requerido"
    df_s = pd.read_csv(sucessor_csv, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8")
    df_ent = pd.read_csv(entradas_csv, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8") if entradas_csv else pd.DataFrame()
    df_sai = pd.read_csv(saidas_csv, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8") if saidas_csv else pd.DataFrame()
    df_srv = pd.read_csv(servicos_csv, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8") if servicos_csv else pd.DataFrame()
    cfop_exp = pd.read_csv(cfop_exp_csv, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8") if cfop_exp_csv else None

    pesos = load_yaml_safe(pesos_yml) or DEFAULT_WEIGHTS
    regex_cfg = load_yaml_safe(regex_yml) or DEFAULT_REGEX

    all_cands: List[Dict[str, Any]] = []
    all_cands += gen_candidates(df_s, df_ent, "ENTRADA", pesos, regex_cfg, janela_dias, tol_abs, tol_pct, cfop_exp)
    all_cands += gen_candidates(df_s, df_sai, "SAIDA", pesos, regex_cfg, janela_dias, tol_abs, tol_pct, cfop_exp)
    all_cands += gen_candidates(df_s, df_srv, "SERVICO", pesos, regex_cfg, janela_dias, tol_abs, tol_pct, cfop_exp)

    winners, losers = pick_best(all_cands, pesos)
    classify_rows(winners, pesos.get("limiares", DEFAULT_WEIGHTS["limiares"]))

    # Salvar
    cand_df = pd.DataFrame(all_cands)
    win_df = pd.DataFrame(winners)

    cand_df.to_csv(out_candidates_csv, index=False, encoding="utf-8")
    win_df.to_csv(out_matches_csv, index=False, encoding="utf-8")

    # Métricas rápidas
    ok = int((win_df["status"] == "OK").sum()) if "status" in win_df.columns else 0
    alerta = int((win_df["status"] == "ALERTA").sum()) if "status" in win_df.columns else 0
    diverg = int((win_df["status"] == "DIVERGENCIA").sum()) if "status" in win_df.columns else 0

    return {
        "candidates": len(cand_df),
        "winners": len(win_df),
        "status": {"OK": ok, "ALERTA": alerta, "DIVERGENCIA": diverg},
        "out_candidates_csv": out_candidates_csv,
        "out_matches_csv": out_matches_csv,
    }


# ------------------ CLI ------------------
def main(argv: List[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(description="matcher.py — motor de conferência S1–S5")
    p.add_argument("--sucessor", required=True)
    p.add_argument("--entradas")
    p.add_argument("--saidas")
    p.add_argument("--servicos")
    p.add_argument("--pesos", default="matching_pesos.yml")
    p.add_argument("--regex", default="regex_tokens.yml")
    p.add_argument("--cfop-exp", dest="cfop_exp")
    p.add_argument("--out-candidates", required=True)
    p.add_argument("--out-matches", required=True)
    p.add_argument("--janela-dias", type=int, default=3)
    p.add_argument("--tol-abs", type=float, default=0.01)
    p.add_argument("--tol-pct", type=float, default=0.002)
    args = p.parse_args(argv)

    res = run_matcher(
        sucessor_csv=args.sucessor,
        entradas_csv=args.entradas,
        saidas_csv=args.saidas,
        servicos_csv=args.servicos,
        pesos_yml=args.pesos,
        regex_yml=args.regex,
        cfop_exp_csv=args.cfop_exp,
        out_candidates_csv=args.out_candidates,
        out_matches_csv=args.out_matches,
        janela_dias=args.janela_dias,
        tol_abs=args.tol_abs,
        tol_pct=args.tol_pct,
    )
    sys.stdout.write(json.dumps(res, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

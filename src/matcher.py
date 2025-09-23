from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_upper(text: Optional[str]) -> Optional[str]:
    if text in (None, ""):
        return None
    return str(text).strip().upper()


def parse_float(cell) -> Optional[float]:
    if cell in (None, ""):
        return None
    try:
        return float(cell)
    except Exception:
        try:
            return float(str(cell).replace(" ", "").replace(".", "").replace(",", "."))
        except Exception:
            return None


def parse_date(cell: Optional[str]) -> Optional[datetime]:
    if cell in (None, ""):
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(str(cell), fmt)
        except Exception:
            continue
    return None


def parse_tokens(cell: Optional[str]) -> Set[str]:
    if not cell:
        return set()
    try:
        data = json.loads(cell)
    except Exception:
        return set()
    return {str(item) for item in data if item}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MatchConfig:
    limiares: Dict[str, float]
    tolerancias: Dict[str, Dict[str, float]]
    estrategias: Dict[str, float]
    bonus: Dict[str, float]
    penalidades: Dict[str, float]
    desempate: Dict[str, object]

    @classmethod
    def load(cls, path: Path) -> "MatchConfig":
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(
            limiares=data.get("limiares", {}),
            tolerancias=data.get("tolerancias", {}),
            estrategias=data.get("estrategias", {}),
            bonus=data.get("bonus", {}),
            penalidades=data.get("penalidades", {}),
            desempate=data.get("desempate", {}),
        )

    def auto_match(self) -> float:
        return float(self.limiares.get("auto_match", 70))

    def alerta_min(self) -> float:
        return float(self.limiares.get("pendente_min", 50))

    def max_candidates(self) -> int:
        return int(self.limiares.get("max_candidates", 6))

    def valor_abs_tol(self) -> float:
        return float(self.tolerancias.get("valor", {}).get("abs", 0.01))

    def valor_pct_tol(self) -> float:
        return float(self.tolerancias.get("valor", {}).get("pct", 0.002))

    def janela_padrao(self) -> int:
        return int(self.limiares.get("janela_default_dias", 3))

    def janela_para(self, fonte_tipo: str) -> int:
        por_fonte = self.tolerancias.get("data", {}).get("janela_dias_por_fonte", {})
        try:
            return int(por_fonte.get(fonte_tipo.upper(), self.janela_padrao()))
        except Exception:
            return self.janela_padrao()


def load_cfop_map(path: Optional[str]) -> Dict[str, Dict[str, List[str]]]:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}
    df = pd.read_csv(file_path, dtype=str, keep_default_na=False, encoding="utf-8")
    mapping: Dict[str, Dict[str, List[str]]] = {}
    for _, row in df.iterrows():
        cfop = str(row.get("cfop") or "").strip()
        if not cfop:
            continue
        debito = [item.strip().upper() for item in str(row.get("debito_sugerido", "")).split(";") if item]
        credito = [item.strip().upper() for item in str(row.get("credito_sugerido", "")).split(";") if item]
        mapping[cfop] = {"debito": debito, "credito": credito}
    return mapping


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------


def within_value(val_s: Optional[float], val_f: Optional[float], cfg: MatchConfig) -> Tuple[bool, Optional[float]]:
    if val_s is None or val_f is None:
        return False, None
    delta = val_s - val_f
    limit = max(cfg.valor_abs_tol(), max(abs(val_s), abs(val_f)) * cfg.valor_pct_tol())
    return abs(delta) <= limit + 1e-9, delta


def within_days(date_s: Optional[datetime], date_f: Optional[datetime], window: int) -> Tuple[bool, Optional[int]]:
    if not date_s or not date_f:
        return False, None
    diff = abs((date_s - date_f).days)
    return diff <= window, diff


def cfop_consistent(s_row: pd.Series, f_row: pd.Series, cfop_rules: Dict[str, Dict[str, List[str]]]) -> Optional[bool]:
    cfop = str(f_row.get("cfop") or "").strip()
    if not cfop:
        return None
    expectations = cfop_rules.get(cfop)
    if not expectations:
        return None
    debito = to_upper(s_row.get("debito")) or ""
    credito = to_upper(s_row.get("credito")) or ""
    debito_ok = True
    credito_ok = True
    if expectations.get("debito"):
        debito_ok = any(token in debito for token in expectations["debito"])
    if expectations.get("credito"):
        credito_ok = any(token in credito for token in expectations["credito"])
    return debito_ok and credito_ok


def compute_score(
    strategy: str,
    matches: Dict[str, bool],
    delta_valor: Optional[float],
    delta_dias: Optional[int],
    cfop_state: Optional[bool],
    cfg: MatchConfig,
    fonte_tipo: str,
    same_month: bool,
) -> Tuple[float, List[str]]:
    score = float(cfg.estrategias.get(strategy, 0))
    reasons: List[str] = [strategy]

    if matches.get("participant"):
        score += float(cfg.bonus.get("participante_exato", 0))
        reasons.append("participante")
    elif matches.get("participant_conflict"):
        score += float(cfg.penalidades.get("participante_conflito", 0))
        reasons.append("participante_confuso")

    if matches.get("valor_ok"):
        score += float(cfg.bonus.get("valor_dentro_tolerancia", 0))
        reasons.append("valor_tol")
        if delta_valor is not None and abs(delta_valor) < 0.0005:
            score += float(cfg.bonus.get("valor_exato_centavos", 0))
            reasons.append("valor_exato")
    elif matches.get("valor_fail"):
        score += float(cfg.penalidades.get("valor_fora_tolerancia", 0))
        reasons.append("valor_ruim")

    if delta_dias is not None:
        if delta_dias == 0:
            score += float(cfg.bonus.get("data_mesma", 0))
            reasons.append("data0")
        if delta_dias <= 1:
            score += float(cfg.bonus.get("data_ate_1_dia", 0))
            reasons.append("data1")
        elif delta_dias <= 3:
            score += float(cfg.bonus.get("data_ate_3_dias", 0))
            reasons.append("data3")
        else:
            score += float(cfg.penalidades.get("data_fora_janela", 0))
            reasons.append("data_out")

    if matches.get("tokens"):
        score += float(cfg.bonus.get("token_forte", 0))
        reasons.append("token")

    if cfop_state is True:
        score += float(cfg.bonus.get("cfop_contas_coerente", 0))
        reasons.append("cfop_ok")
    elif cfop_state is False:
        score += float(cfg.penalidades.get("cfop_contas_conflito", 0))
        reasons.append("cfop_bad")

    if same_month:
        score += float(cfg.bonus.get("mesmo_mes_ref", 0))
        reasons.append("mes_ref")

    fonte_bonus = cfg.bonus.get("fonte_prioritaria")
    if fonte_bonus:
        prefs = cfg.desempate.get("prioridade_fonte", {})
        weight = float(prefs.get(fonte_tipo.upper(), 1.0)) if isinstance(prefs, dict) else 1.0
        score += float(fonte_bonus) * weight
        reasons.append(f"prioridade_{fonte_tipo.upper()}")

    return score, reasons


# ---------------------------------------------------------------------------
# Candidate model
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    sucessor_idx: int
    fonte_idx: int
    fonte_tipo: str
    strategy: str
    score: float
    reasons: List[str]
    delta_valor: Optional[float]
    delta_dias: Optional[int]
    s: pd.Series
    f: pd.Series


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_table_generic(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    fpath = Path(path)
    if not fpath.exists():
        return pd.DataFrame()
    if fpath.suffix.lower() == ".parquet":
        return pd.read_parquet(fpath)
    return pd.read_csv(fpath, dtype=str, keep_default_na=False, encoding="utf-8")


def load_fontes(args: argparse.Namespace) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in [args.entradas, args.saidas, args.servicos, getattr(args, "practice", None), getattr(args, "mister", None)]:
        df = load_table_generic(path)
        if not df.empty:
            frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def build_candidates(
    sucessor: pd.DataFrame,
    fontes: pd.DataFrame,
    cfg: MatchConfig,
    cfop_rules: Dict[str, Dict[str, List[str]]],
) -> List[Candidate]:
    if sucessor.empty or fontes.empty:
        return []

    fontes = fontes.reset_index(drop=True).copy()
    sucessor = sucessor.reset_index(drop=True).copy()

    fontes["valor_float"] = fontes["valor"].map(parse_float)
    fontes["data_dt"] = fontes["data_iso"].map(parse_date)
    fontes["tokens_set"] = fontes["tokens"].map(parse_tokens)
    fontes["participante_key"] = fontes["participante_key"].map(to_upper)
    fontes["doc_num"] = fontes["doc_num"].fillna("")

    sucessor["valor_float"] = sucessor["valor"].map(parse_float)
    sucessor["data_dt"] = sucessor["data_iso"].map(parse_date)
    sucessor["tokens_set"] = sucessor["tokens"].map(parse_tokens)
    sucessor["participante_key"] = sucessor["participante_key"].map(to_upper)
    sucessor["doc_num"] = sucessor["doc_num"].fillna("")

    doc_index: Dict[str, Set[int]] = {}
    participant_index: Dict[str, Set[int]] = {}
    token_index: Dict[str, Set[int]] = {}

    for idx, row in fontes.iterrows():
        doc = str(row.get("doc_num") or "").strip()
        if doc:
            doc_index.setdefault(doc, set()).add(idx)
        part = str(row.get("participante_key") or "").strip()
        if part:
            participant_index.setdefault(part, set()).add(idx)
        for token in row.get("tokens_set") or set():
            token_index.setdefault(token, set()).add(idx)

    all_candidates: List[Candidate] = []

    for _, s_row in sucessor.iterrows():
        s_idx = int(s_row.get("row_id", s_row.name + 1))
        s_doc = str(s_row.get("doc_num") or "").strip()
        s_part = str(s_row.get("participante_key") or "").strip()
        s_tokens = s_row.get("tokens_set") or set()
        s_value = s_row.get("valor_float")

        candidate_ids: Set[int] = set()
        if s_doc:
            candidate_ids.update(doc_index.get(s_doc, set()))
        if s_part:
            candidate_ids.update(participant_index.get(s_part, set()))
        for token in s_tokens:
            candidate_ids.update(token_index.get(token, set()))

        if not candidate_ids and s_value is not None:
            limit = max(cfg.valor_abs_tol(), abs(s_value) * cfg.valor_pct_tol())
            mask = fontes["valor_float"].notna()
            mask &= (fontes["valor_float"] - s_value).abs() <= limit + 1e-9
            candidate_ids.update(set(fontes[mask].index))

        if not candidate_ids:
            continue

        local: List[Candidate] = []
        for f_idx in candidate_ids:
            f_row = fontes.loc[f_idx]
            fonte_tipo = str(f_row.get("fonte_tipo") or "?")

            value_ok, delta_valor = within_value(s_value, f_row.get("valor_float"), cfg)
            window = cfg.janela_para(fonte_tipo)
            date_ok, delta_dias = within_days(s_row.get("data_dt"), f_row.get("data_dt"), window)
            tokens_overlap = bool((s_tokens or set()) & (f_row.get("tokens_set") or set()))
            participant_match = bool(s_part and s_part == str(f_row.get("participante_key") or "").strip())
            doc_match = bool(s_doc and s_doc == str(f_row.get("doc_num") or "").strip())

            strategy = None
            if doc_match and participant_match and value_ok:
                strategy = "S1"
            elif doc_match and participant_match and date_ok:
                strategy = "S2"
            elif doc_match and value_ok and date_ok:
                strategy = "S3"
            elif participant_match and value_ok and date_ok:
                strategy = "S4"
            elif tokens_overlap and value_ok and date_ok:
                strategy = "S5"

            if strategy is None:
                continue

            matches = {
                "participant": participant_match,
                "participant_conflict": bool(s_part and f_row.get("participante_key") and not participant_match),
                "valor_ok": value_ok,
                "valor_fail": not value_ok,
                "tokens": tokens_overlap,
            }
            cfop_state = cfop_consistent(s_row, f_row, cfop_rules)
            same_month = bool(s_row.get("mes_ref") and f_row.get("mes_ref") and s_row.get("mes_ref") == f_row.get("mes_ref"))

            score, reasons = compute_score(
                strategy,
                matches,
                delta_valor,
                delta_dias,
                cfop_state,
                cfg,
                fonte_tipo,
                same_month,
            )

            local.append(
                Candidate(
                    sucessor_idx=s_idx,
                    fonte_idx=int(f_row.get("row_id", f_idx + 1)),
                    fonte_tipo=fonte_tipo,
                    strategy=strategy,
                    score=score,
                    reasons=reasons,
                    delta_valor=delta_valor,
                    delta_dias=delta_dias,
                    s=s_row,
                    f=f_row,
                )
            )

        if not local:
            continue
        local.sort(key=lambda c: (-c.score, abs(c.delta_valor or 0.0), abs(c.delta_dias or 0)))
        all_candidates.extend(local[: cfg.max_candidates()])

    return all_candidates


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def classify(score: float, cfg: MatchConfig) -> str:
    if score >= cfg.auto_match():
        return "OK"
    if score >= cfg.alerta_min():
        return "ALERTA"
    return "DIVERGENCIA"


@dataclass
class MatchResult:
    grid: pd.DataFrame
    sem_fonte: pd.DataFrame
    sem_sucessor: pd.DataFrame
    logs: List[Dict[str, object]]


def build_sem_fonte(sucessor: pd.DataFrame, matched: Set[int]) -> pd.DataFrame:
    rows = []
    for _, s in sucessor.iterrows():
        s_idx = int(s.get("row_id", s.name + 1))
        if s_idx in matched:
            continue
        rows.append(
            {
                "S.row_id": s_idx,
                "S.data": s.get("data"),
                "S.doc": s.get("doc"),
                "S.doc_num": s.get("doc_num"),
                "S.valor": s.get("valor"),
                "S.debito": s.get("debito"),
                "S.credito": s.get("credito"),
                "S.part_d": s.get("part_d"),
                "S.part_c": s.get("part_c"),
                "S.historico": s.get("historico"),
            }
        )
    return pd.DataFrame(rows)


def build_sem_sucessor(fontes: pd.DataFrame, matched: Set[Tuple[str, int]]) -> pd.DataFrame:
    rows = []
    for _, f in fontes.iterrows():
        key = (str(f.get("fonte_tipo") or ""), int(f.get("row_id", f.name + 1)))
        if key in matched:
            continue
        rows.append(
            {
                "fonte_tipo": key[0],
                "F.row_id": key[1],
                "F.data": f.get("data"),
                "F.doc": f.get("doc"),
                "F.doc_num": f.get("doc_num"),
                "F.valor": f.get("valor"),
                "F.cfop": f.get("cfop"),
                "F.participante": f.get("participante"),
                "F.situacao": f.get("situacao"),
            }
        )
    return pd.DataFrame(rows)


def pick_best(candidates: List[Candidate], sucessor: pd.DataFrame, fontes: pd.DataFrame, cfg: MatchConfig) -> MatchResult:
    if not candidates:
        return MatchResult(pd.DataFrame(), build_sem_fonte(sucessor, set()), build_sem_sucessor(fontes, set()), [])

    def sort_key(c: Candidate) -> Tuple[float, float, float]:
        return (
            c.score,
            -abs(c.delta_valor or 0.0),
            -abs(c.delta_dias or 0)
        )

    used_s: Dict[int, Candidate] = {}
    used_f: Set[Tuple[str, int]] = set()

    for cand in sorted(candidates, key=sort_key, reverse=True):
        s_idx = cand.sucessor_idx
        f_key = (cand.fonte_tipo, cand.fonte_idx)
        if s_idx in used_s:
            continue
        if f_key in used_f:
            continue
        used_s[s_idx] = cand
        used_f.add(f_key)

    grid_rows: List[Dict[str, object]] = []
    logs: List[Dict[str, object]] = []

    for cand in used_s.values():
        status = classify(cand.score, cfg)
        s = cand.s
        f = cand.f
        grid_rows.append(
            {
                "sucessor_idx": cand.sucessor_idx,
                "fonte_idx": cand.fonte_idx,
                "fonte_tipo": cand.fonte_tipo,
                "status": status,
                "match.status": status,
                "match.strategy": cand.strategy,
                "match.score": round(cand.score, 2),
                "match.motivos": ";".join(cand.reasons),
                "delta.valor": cand.delta_valor,
                "delta.dias": cand.delta_dias,
                "S.data": s.get("data"),
                "S.data_iso": s.get("data_iso"),
                "S.doc": s.get("doc"),
                "S.doc_num": s.get("doc_num"),
                "S.doc_serie": s.get("doc_serie"),
                "S.valor": s.get("valor"),
                "S.debito": s.get("debito"),
                "S.credito": s.get("credito"),
                "S.part_d": s.get("part_d"),
                "S.part_c": s.get("part_c"),
                "S.historico": s.get("historico"),
                "S.participante": s.get("participante_combined"),
                "F.data": f.get("data"),
                "F.data_iso": f.get("data_iso"),
                "F.doc": f.get("doc"),
                "F.doc_num": f.get("doc_num"),
                "F.doc_serie": f.get("doc_serie"),
                "F.valor": f.get("valor"),
                "F.cfop": f.get("cfop"),
                "F.situacao": f.get("situacao"),
                "F.participante": f.get("participante"),
                "F.debito_alias": f.get("debito_alias"),
                "F.credito_alias": f.get("credito_alias"),
            }
        )
        logs.append(
            {
                "sucessor_idx": cand.sucessor_idx,
                "fonte_idx": cand.fonte_idx,
                "fonte_tipo": cand.fonte_tipo,
                "status": status,
                "strategy": cand.strategy,
                "score": cand.score,
                "delta_valor": cand.delta_valor,
                "delta_dias": cand.delta_dias,
                "motivos": cand.reasons,
            }
        )

    sem_fonte = build_sem_fonte(sucessor, set(used_s.keys()))
    sem_sucessor = build_sem_sucessor(fontes, used_f)
    return MatchResult(pd.DataFrame(grid_rows), sem_fonte, sem_sucessor, logs)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def save_match_outputs(result: MatchResult, out_dir: Path) -> Dict[str, str]:
    ensure_dir(out_dir)
    grid_path = out_dir / "reconc_grid.csv"
    result.grid.to_csv(grid_path, index=False, encoding="utf-8")

    sem_fonte_path = out_dir / "reconc_sem_fonte.csv"
    result.sem_fonte.to_csv(sem_fonte_path, index=False, encoding="utf-8")

    sem_sucessor_path = out_dir / "reconc_sem_sucessor.csv"
    result.sem_sucessor.to_csv(sem_sucessor_path, index=False, encoding="utf-8")

    return {
        "grid": str(grid_path),
        "sem_fonte": str(sem_fonte_path),
        "sem_sucessor": str(sem_sucessor_path),
    }


def write_match_log(path: Path, logs: List[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in logs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matcher S1..S5 for reconciliation")
    parser.add_argument("--sucessor", required=True)
    parser.add_argument("--entradas")
    parser.add_argument("--saidas")
    parser.add_argument("--servicos")
    parser.add_argument("--cfg-pesos", default="cfg/matching_pesos.yml")
    parser.add_argument("--cfop-map", default="cfg/cfop_expectativas.csv")
    parser.add_argument("--cfg-tokens")
    parser.add_argument("--fornecedores")
    parser.add_argument("--plano-contas")
    parser.add_argument("--practice")
    parser.add_argument("--mister")
    parser.add_argument("--out", required=True)
    parser.add_argument("--log", default="out/match/match_log.jsonl")
    return parser.parse_args(argv)


def run_matcher(args: argparse.Namespace) -> Dict[str, object]:
    cfg = MatchConfig.load(Path(args.cfg_pesos))
    sucessor_df = load_table_generic(args.sucessor)
    fontes_df = load_fontes(args)
    cfop_rules = load_cfop_map(args.cfop_map)

    candidates = build_candidates(sucessor_df, fontes_df, cfg, cfop_rules)
    result = pick_best(candidates, sucessor_df, fontes_df, cfg)
    outputs = save_match_outputs(result, Path(args.out))
    write_match_log(Path(args.log), result.logs)

    return {
        "matches": int(len(result.grid)),
        "sem_fonte": int(len(result.sem_fonte)),
        "sem_sucessor": int(len(result.sem_sucessor)),
        **outputs,
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    summary = run_matcher(args)
    sys.stdout.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())




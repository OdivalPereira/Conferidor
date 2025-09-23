# normalizer.py — 13/28
# Aplica profiles JSON (profile_*.json) a DataFrames:
# - renomeações, datas (BR→ISO), números (vírgula), documento (num/série),
# - participante (upper/sem acento), CFOP, situação,
# - campos derivados simples e validações,
# - mapeamento para tabela de staging.
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json, re, sys, unicodedata
from datetime import datetime

try:
    import pandas as pd  # preferido
except Exception:
    pd = None  # type: ignore

try:
    import polars as pl  # opcional
except Exception:
    pl = None  # type: ignore

__all__ = ["load_profile", "Normalizer", "digits", "strip_accents"]

# ---------------- Utils ----------------
def strip_accents(s: str) -> str:
    if s is None:
        return s
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn")

def to_upper(s: Any) -> Any:
    return None if s is None else str(s).upper()

def collapse_spaces(s: Optional[str]) -> Optional[str]:
    if s is None: return s
    return re.sub(r"\s+", " ", str(s)).strip()

def digits(s: Any, expect_len: Optional[int] = None) -> Optional[str]:
    if s is None: return None
    ds = re.sub(r"\D", "", str(s))
    if expect_len is not None and len(ds) != expect_len:
        return None
    return ds or None

def lstrip_zeros(s: Any) -> Optional[str]:
    if s is None: return None
    out = str(s).lstrip("0")
    return out if out != "" else "0"

def parse_date_any(s: Any, fmts: List[str]) -> Optional[str]:
    if s is None: return None
    txt = str(s).strip()
    if txt == "": return None
    for fmt in fmts:
        try:
            dt = datetime.strptime(txt, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    return None

def parse_number_br(
    s: Any,
    remove_thousands: bool = True,
    decimal_comma: bool = True,
    fallback_decimal_point: bool = True,
    allow_parentheses_negative: bool = True,
    round_places: Optional[int] = 2,
    percent_like: bool = False,
) -> Optional[float]:
    if s is None: return None
    txt = str(s).strip()
    if txt == "": return None
    neg = False
    if allow_parentheses_negative and txt.startswith("(") and txt.endswith(")"):
        neg = True; txt = txt[1:-1]
    if remove_thousands:
        txt = txt.replace(".", "").replace(" ", "")
    if decimal_comma and "," in txt:
        txt = txt.replace(",", ".")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", txt)
    if not m: return None
    val = float(m.group(0))
    if percent_like and "%" in str(s):
        val = val / 100.0
    if neg: val = -val
    if round_places is not None: val = round(val, round_places)
    return val

def coalesce(*values):
    for v in values:
        if v is not None and v != "":
            return v
    return None

def substr(s: Optional[str], start1: int, length: int) -> Optional[str]:
    if s is None: return None
    i0 = max(start1 - 1, 0)
    return s[i0 : i0 + length]

def safe_bool(val) -> bool:
    return str(val).strip().lower() in ("1","true","t","yes","y","sim")

# ------------- Profile Loader -------------
def load_profile(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

# ------------- Normalizer -------------
class Normalizer:
    """Aplica um profile_* a pandas (padrão) ou polars, produzindo df normalizado e df de staging."""
    def __init__(self, backend: str = "pandas"):
        if backend not in ("pandas","polars","auto"):
            raise ValueError("backend must be 'pandas', 'polars' or 'auto'")
        if backend == "auto":
            backend = "pandas" if pd is not None else "polars"
        self.backend = backend
        if self.backend == "pandas" and pd is None:
            raise RuntimeError("pandas é requerido para 'pandas'")
        if self.backend == "polars" and pl is None:
            raise RuntimeError("polars é requerido para 'polars'")

    # ---- helpers ----
    def _df_from_any(self, data):
        if self.backend == "pandas":
            if isinstance(data, pd.DataFrame): return data.copy()
            raise TypeError("Esperado pandas.DataFrame")
        else:
            if isinstance(data, pl.DataFrame): return data.clone()
            raise TypeError("Esperado polars.DataFrame")

    def _ensure_columns(self, df, cols: List[str]):
        for c in cols:
            if c not in df.columns:
                if self.backend == "pandas":
                    df[c] = None
                else:
                    df = df.with_columns(pl.lit(None).alias(c))
        return df

    def _apply_renames(self, cols: List[str], profile: Dict[str, Any]) -> Tuple[List[str], Dict[str,str], List[str]]:
        rules = profile.get("rename_rules", [])
        compiled = [(re.compile(r["pattern"]), r["to"]) for r in rules]
        new_cols, mapping = [], {}
        warnings = []
        for c in cols:
            c_new = c
            for rx, dest in compiled:
                if rx.search(c):
                    c_new = dest; break
            new_cols.append(c_new); mapping[c] = c_new
        if profile.get("profiling", {}).get("warn_on_unknown_columns", False):
            known = set(profile.get("canonical_columns", []))
            for c in new_cols:
                if c not in known and c not in ("data_iso","valor_num","doc_num_norm","doc_serie_norm"):
                    warnings.append(f"Coluna '{c}' não é canônica no profile '{profile.get('profile_name')}'.")
        return new_cols, mapping, warnings

    # ---- parse blocks ----
    def _apply_parse_blocks(self, df, profile: Dict[str, Any]):
        rules = profile.get("parse_rules", {})

        # Dates
        date_list = []
        if rules.get("date"): date_list.append(rules["date"])
        date_list += rules.get("dates", [])
        for dr in date_list:
            inf, outf = dr.get("input_field"), dr.get("output_field")
            fmts = dr.get("formats", [])
            if self.backend == "pandas":
                df[outf] = df[inf].apply(lambda x: parse_date_any(x, fmts))
            else:
                df = df.with_columns(pl.col(inf).map_elements(lambda x: parse_date_any(x, fmts)).alias(outf))

        # Numbers
        num_list = []
        if rules.get("number"): num_list.append(rules["number"])
        num_list += rules.get("numbers", [])
        for nr in num_list:
            inf, outf = nr.get("input_field"), nr.get("output_field")
            opts = {
                "remove_thousands": nr.get("remove_thousands", True),
                "decimal_comma": nr.get("decimal_comma", True),
                "fallback_decimal_point": nr.get("fallback_decimal_point", True),
                "allow_parentheses_negative": nr.get("allow_parentheses_negative", True),
                "round_places": nr.get("round_places", 2),
                "percent_like": nr.get("percent_like", False),
            }
            if self.backend == "pandas":
                df[outf] = df[inf].apply(lambda x: parse_number_br(x, **opts))
            else:
                df = df.with_columns(pl.col(inf).map_elements(lambda x: parse_number_br(x, **opts)).alias(outf))

        # Document
        if "document" in rules:
            dr = rules["document"]
            inf, outs = dr.get("input_field"), dr.get("outputs", {})
            strip_tokens = set(dr.get("strip_tokens", []))
            uppercase = safe_bool(dr.get("uppercase", True))

            if self.backend == "pandas":
                base = df[inf].astype(str)
                if uppercase: base = base.str.upper()
                for tok in strip_tokens:
                    base = base.str.replace(rf"\b{re.escape(tok)}\b", "", regex=True)
                base = base.str.strip()
                if outs.get("doc_num_norm", {}).get("regex_keep_digits", False):
                    df["doc_num_norm"] = base.apply(lambda x: digits(x))
                ser_pat = outs.get("doc_serie_norm", {}).get("regex_series")
                if ser_pat:
                    r = re.compile(ser_pat)
                    df["doc_serie_norm"] = base.apply(lambda x: (lambda m: lstrip_zeros(m.group(1)) if m else None)(r.search(str(x))))
            else:
                base = pl.col(inf).cast(pl.Utf8)
                if uppercase: base = base.str.to_uppercase()
                for tok in strip_tokens:
                    base = base.str.replace(rf"\b{re.escape(tok)}\b", "", literal=False)
                base = base.str.strip()
                df = df.with_columns(base.alias("_doc_base"))
                df = df.with_columns(pl.col("_doc_base").map_elements(lambda x: digits(x)).alias("doc_num_norm"))
                ser_pat = outs.get("doc_serie_norm", {}).get("regex_series")
                if ser_pat:
                    r = re.compile(ser_pat)
                    df = df.with_columns(
                        pl.col("_doc_base").map_elements(lambda x: (lambda m: lstrip_zeros(m.group(1)) if m else None)(r.search(str(x)))).alias("doc_serie_norm")
                    )
                df = df.drop("_doc_base")

        # Participant (generic)
        if "participant" in rules:
            pr = rules["participant"]
            fields = pr.get("fields", [])
            norm = pr.get("normalize", {})
            do_upper = safe_bool(norm.get("upper", True))
            do_strip = safe_bool(norm.get("strip", True))
            do_rm_acc = safe_bool(norm.get("remove_accents", True))

            def _norm_str(x):
                if x is None: return None
                s = str(x)
                if do_strip: s = s.strip()
                if do_rm_acc: s = strip_accents(s)
                if do_upper: s = s.upper()
                s = collapse_spaces(s)
                return s

            if self.backend == "pandas":
                for f in fields:
                    if f in df.columns: df[f] = df[f].apply(_norm_str)
            else:
                for f in fields:
                    if f in df.columns: df = df.with_columns(pl.col(f).map_elements(_norm_str).alias(f))

        # Text cleanup
        if "text_cleanup" in rules:
            tr = rules["text_cleanup"]
            fields = tr.get("fields", [])
            do_upper = safe_bool(tr.get("upper", True))
            collapse = safe_bool(tr.get("collapse_spaces", True))
            trim_punc = safe_bool(tr.get("trim_punctuation", True))

            def _cleanup(x):
                if x is None: return None
                s = str(x)
                if do_upper: s = s.upper()
                if collapse: s = collapse_spaces(s)
                if trim_punc: s = re.sub(r"[;,:]+$", "", s)
                return s

            if self.backend == "pandas":
                for f in fields:
                    if f in df.columns: df[f] = df[f].apply(_cleanup)
            else:
                for f in fields:
                    if f in df.columns: df = df.with_columns(pl.col(f).map_elements(_cleanup).alias(f))

        # CFOP
        if "cfop" in rules:
            cr = rules["cfop"]
            inf = cr.get("input_field")
            keep_digits_only = safe_bool(cr.get("keep_digits_only", True))
            pad_left_to_4 = safe_bool(cr.get("pad_left_to_4", True))

            def _cfop(x):
                if x is None: return None
                s = str(x).strip()
                if keep_digits_only: s = digits(s) or ""
                if pad_left_to_4 and s != "": s = s.zfill(4)
                return s or None

            if self.backend == "pandas":
                if inf in df.columns: df["cfop"] = df[inf].apply(_cfop)
            else:
                if inf in df.columns: df = df.with_columns(pl.col(inf).map_elements(_cfop).alias("cfop"))

        # Situação
        if "situacao" in rules:
            sr = rules["situacao"]
            inf = sr.get("input_field")
            mapping = sr.get("normalize_map", {})
            uppercase = safe_bool(sr.get("uppercase", True))

            def _sit(x):
                if x is None: return None
                s = str(x)
                if uppercase: s = s.upper()
                s_clean = s.strip()
                return mapping.get(s_clean.lower(), s if uppercase else s_clean)

            if self.backend == "pandas":
                if inf in df.columns: df["situacao"] = df[inf].apply(_sit)
            else:
                if inf in df.columns: df = df.with_columns(pl.col(inf).map_elements(_sit).alias("situacao"))

        # Derived
        for de in rules.get("derived", []):
            expr = de.get("expr", ""); out = de.get("as")
            if not out: continue

            def _eval_row(row: Dict[str, Any]):
                def _get(name): return row.get(name)
                m = re.match(r"coalesce\(([^,]+),\s*([^)]+)\)", expr, flags=re.I)
                if m:
                    a, b = m.group(1).strip(), m.group(2).strip()
                    return coalesce(_get(a), _get(b))
                m2 = re.match(r"substr\((.+),\s*(\d+),\s*(\d+)\)", expr, flags=re.I)
                if m2:
                    inner, start, length = m2.group(1).strip(), int(m2.group(2)), int(m2.group(3))
                    if inner.lower().startswith("coalesce("):
                        m3 = re.match(r"coalesce\(([^,]+),\s*([^)]+)\)", inner, flags=re.I)
                        if m3:
                            a, b = m3.group(1).strip(), m3.group(2).strip()
                            val = coalesce(_get(a), _get(b))
                            return substr(val, start, length)
                    else:
                        return substr(_get(inner), start, length)
                return None

            if self.backend == "pandas":
                df[out] = df.apply(lambda r: _eval_row(r.to_dict()), axis=1)
            else:
                df = df.with_columns(pl.struct(df.columns).map_elements(lambda r: _eval_row(r)).alias(out))

        return df

    # ---- validators ----
    def _apply_validators(self, df, profile: Dict[str, Any]) -> List[str]:
        problems = []
        v = profile.get("validators", {})
        for col in v.get("required_columns", []):
            if col not in df.columns:
                problems.append(f"Coluna obrigatória ausente: '{col}'")

        for k in ("at_least_one_participant","at_least_one_date"):
            cols = v.get(k, [])
            if cols:
                present = [c for c in cols if c in df.columns]
                if not present:
                    problems.append(f"Nenhuma das colunas {cols} encontrada para '{k}'")

        def check_rule(expr: str, row: Dict[str, Any]) -> bool:
            def _basic(cond: str) -> bool:
                cond = cond.strip()
                m = re.match(r"length\((\w+)\)\s*>=\s*(\d+)", cond, flags=re.I)
                if m:
                    fld, n = m.group(1), int(m.group(2))
                    val = row.get(fld)
                    return len(str(val)) >= n if val is not None else False
                m = re.match(r"(\w+)\s+IS\s+NULL", cond, flags=re.I)
                if m: return row.get(m.group(1)) in (None, "")
                m = re.match(r"(\w+)\s+IS\s+NOT\s+NULL", cond, flags=re.I)
                if m: 
                    v = row.get(m.group(1)); return v is not None and v != ""
                m = re.match(r"(\w+)\s*(>=|<=|=|>|<)\s*([0-9.]+)", cond, flags=re.I)
                if m:
                    fld, op, val = m.group(1), m.group(2), float(m.group(3))
                    v = row.get(fld)
                    if v is None: return False
                    try: vf = float(v)
                    except: return False
                    return (vf >= val if op==">=" else
                            vf <= val if op=="<=" else
                            vf > val if op==">" else
                            vf < val if op=="<" else
                            abs(vf - val) < 1e-9)
                return True
            parts_or = re.split(r"\s+OR\s+", expr, flags=re.I)
            res_or = False
            for pr in parts_or:
                parts_and = re.split(r"\s+AND\s+", pr, flags=re.I)
                res_and = True
                for pa in parts_and:
                    res_and = res_and and _basic(pa)
                    if not res_and: break
                res_or = res_or or res_and
                if res_or: break
            return res_or

        sample = df.head(200) if self.backend == "pandas" else df.head(200)
        for rule in v.get("rules", []):
            name, expr = rule.get("name","regra"), rule.get("expr","")
            if self.backend == "pandas":
                bad = []
                for idx, r in sample.iterrows():
                    if not check_rule(expr, r.to_dict()):
                        bad.append(idx)
                if bad:
                    problems.append(f"Validação '{name}' falhou em {len(bad)} linhas (amostra).")
            else:
                def _row_bad(rdict): return not check_rule(expr, rdict)
                bad = sample.with_columns(pl.struct(sample.columns).map_elements(lambda r: _row_bad(r)).alias("_bad"))
                nbad = int(bad["_bad"].sum()) if "_bad" in bad.columns else 0
                if nbad:
                    problems.append(f"Validação '{name}' falhou em {nbad} linhas (amostra).")
        return problems

    # ---- staging mapping ----
    def _map_to_staging(self, df, profile: Dict[str, Any]):
        mapping = profile.get("output_mapping_to_staging", {})
        if not mapping: return None, None
        table = mapping.get("table")
        cols_map = mapping.get("columns", {})

        def _eval_expr(expr: str, row: Dict[str, Any]):
            expr = expr.strip()
            if expr.startswith("'") and expr.endswith("'") and len(expr) >= 2:
                return expr[1:-1]
            m = re.match(r"coalesce\(([^,]+),\s*([^)]+)\)", expr, flags=re.I)
            if m:
                a, b = m.group(1).strip(), m.group(2).strip()
                return coalesce(row.get(a), row.get(b))
            return row.get(expr)

        if self.backend == "pandas":
            out_rows = []
            for _, r in df.iterrows():
                rdict = r.to_dict()
                row_out = {dest: _eval_expr(src_expr, rdict) for dest, src_expr in cols_map.items()}
                out_rows.append(row_out)
            out_df = pd.DataFrame(out_rows)
        else:
            def _build_row(r):
                rdict = r
                return {dest: _eval_expr(src_expr, rdict) for dest, src_expr in cols_map.items()}
            out_df = df.with_columns(pl.struct(df.columns).map_elements(_build_row).alias("_tmp")).select("_tmp").unnest("_tmp")
        return table, out_df

    # ---- API pública ----
    def normalize(self, df_any, profile: Dict[str, Any]):
        df = self._df_from_any(df_any)
        new_cols, mapping, warns = self._apply_renames(list(df.columns), profile)
        if self.backend == "pandas":
            df.columns = new_cols
        else:
            df = df.rename({old:new for old,new in mapping.items()})
        df = self._ensure_columns(df, profile.get("canonical_columns", []))
        df = self._apply_parse_blocks(df, profile)
        problems = self._apply_validators(df, profile)
        table, df_staging = self._map_to_staging(df, profile)
        return {"df_norm": df, "staging": (table, df_staging), "warnings": warns, "problems": problems}

# ------------- CLI -------------
CLI_HELP = """
Uso:
  python normalizer.py --csv <input.csv> --profile <profile.json> --out-norm <norm.csv> --out-staging <staging.csv> [--backend pandas|polars|auto]
"""

def _read_csv_any(path: str, backend: str):
    if backend == "pandas":
        if pd is None: raise RuntimeError("pandas indisponível")
        return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8")
    else:
        if pl is None: raise RuntimeError("polars indisponível")
        return pl.read_csv(path, infer_schema_length=5000)

def _write_csv_any(df, path: str, backend: str):
    if backend == "pandas":
        df.to_csv(path, index=False, encoding="utf-8")
    else:
        df.write_csv(path)

def main(argv: List[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Normalizer (profiles → staging)")
    p.add_argument("--csv", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--out-norm", required=True)
    p.add_argument("--out-staging", required=True)
    p.add_argument("--backend", default="pandas", choices=["pandas","polars","auto"])
    args = p.parse_args(argv)

    prof = load_profile(args.profile)
    normalizer = Normalizer(backend=args.backend)
    df = _read_csv_any(args.csv, "pandas" if normalizer.backend=="pandas" else "polars")
    result = normalizer.normalize(df, prof)

    df_norm = result["df_norm"]
    table, df_staging = result["staging"]
    _write_csv_any(df_norm, args.out_norm, "pandas" if normalizer.backend=="pandas" else "polars")
    if df_staging is not None:
        _write_csv_any(df_staging, args.out_staging, "pandas" if normalizer.backend=="pandas" else "polars")
    else:
        Path(args.out_staging).write_text("", encoding="utf-8")

    sys.stdout.write(f"Normalized rows: {len(df_norm)}\n")
    sys.stdout.write(f"Staging table: {table}\n")
    if result["warnings"]:
        sys.stdout.write("Warnings:\n- " + "\n- ".join(result["warnings"]) + "\n")
    if result["problems"]:
        sys.stdout.write("Problems:\n- " + "\n- ".join(result["problems"]) + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

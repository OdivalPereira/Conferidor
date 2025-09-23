# loader.py — 12/28
# CSV Loader robusto (MVP local)
# - Detecta encoding e delimitador
# - Lê como texto (sem coerção), preserva cabeçalhos
# - Valida cabeçalhos contra profile_*.json (opcional)
# - Exporta CSV padronizado (UTF-8 + vírgula)
# - (Opcional) Ingestão em SQLite (colunas TEXT)
from __future__ import annotations
import csv as _csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore

import sqlite3

ENCODING_CANDIDATES = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
DELIMITER_CANDIDATES = [";", ",", "\t", "|"]

def sniff_encoding(path: str, candidates: List[str] = ENCODING_CANDIDATES) -> str:
    for enc in candidates:
        try:
            with open(path, "r", encoding=enc) as f:
                f.read(4096)
            return enc
        except Exception:
            continue
    return "utf-8"

def sniff_delimiter(path: str, encoding: str) -> Tuple[str, str, bool]:
    with open(path, "r", encoding=encoding, errors="replace") as f:
        sample = f.read(65536)
    try:
        dialect = _csv.Sniffer().sniff(sample, delimiters=";,|\t")
        has_header = _csv.Sniffer().has_header(sample)
        return dialect.delimiter, (dialect.quotechar or '"'), has_header
    except Exception:
        # Heurística simples
        best_delim = max(DELIMITER_CANDIDATES, key=lambda d: sample.count(d))
        return best_delim, '"', True

def read_csv_smart(path: str,
                   encoding: Optional[str] = None,
                   delimiter: Optional[str] = None,
                   quotechar: Optional[str] = None,
                   has_header: Optional[bool] = None):
    assert pd is not None, "pandas é necessário para o loader"
    enc = encoding or sniff_encoding(path)
    if delimiter is None:
        delim, quote, header = sniff_delimiter(path, enc)
    else:
        delim, quote, header = delimiter, (quotechar or '"'), True if has_header is None else has_header
    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        encoding=enc,
        sep=delim,
        quotechar=quote,
        header=0 if header else None,
        engine="python",
        on_bad_lines="skip",
    )
    return df, {"encoding": enc, "delimiter": delim, "quotechar": quote, "has_header": header}

def load_profile(profile_path: Optional[str]) -> Optional[Dict]:
    if not profile_path:
        return None
    return json.loads(Path(profile_path).read_text(encoding="utf-8"))

def canonicalize_headers(cols: List[str]) -> List[str]:
    def _norm(c: str) -> str:
        c = str(c).strip()
        c = " ".join(c.split())
        return c
    return [_norm(c) for c in cols]

def validate_against_profile(df, profile: Dict) -> Dict[str, List[str]]:
    canon = set(profile.get("canonical_columns", []))
    present = set(df.columns)
    missing = [c for c in canon if c not in present]
    extras = [c for c in df.columns if c not in canon]
    return {"missing": missing, "extras": extras}

def write_csv_utf8(df, out_path: str) -> None:
    df.to_csv(out_path, index=False, encoding="utf-8")

def ensure_sqlite_table(conn: sqlite3.Connection, table: str, columns: List[str], replace: bool = False):
    cur = conn.cursor()
    if replace:
        cur.execute(f'DROP TABLE IF EXISTS "{table}"')
    cols_sql = ", ".join([f'"{c}" TEXT' for c in columns])
    cur.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({cols_sql})')
    conn.commit()

def insert_sqlite_bulk(conn: sqlite3.Connection, table: str, df) -> int:
    cur = conn.cursor()
    cols = list(df.columns)
    placeholders = ", ".join(["?"] * len(cols))
    col_list = ", ".join([f'"{c}"' for c in cols])
    rows = []
    for _, row in df.iterrows():
        rows.append(tuple(None if (v == "" or str(v).lower() == "nan") else str(v) for v in row.tolist()))
    cur.executemany(f'INSERT INTO "{table}" ({col_list}) VALUES ({placeholders})', rows)
    conn.commit()
    return len(rows)

def main(argv: List[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Loader — leitura robusta de CSV (UTF-8 + separador canônico)")
    p.add_argument("--in", dest="inp", required=True, help="Caminho do CSV de entrada")
    p.add_argument("--profile", dest="profile", help="(Opcional) profile_*.json para validar cabeçalhos")
    p.add_argument("--out-csv", dest="out_csv", required=True, help="Saída CSV padronizada (UTF-8, separador ',')")
    p.add_argument("--encoding", dest="encoding")
    p.add_argument("--delimiter", dest="delimiter")
    p.add_argument("--quotechar", dest="quotechar")
    p.add_argument("--no-header", dest="no_header", action="store_true")
    p.add_argument("--db-sqlite", dest="db_sqlite", help="(Opcional) caminho .db para inserir dados")
    p.add_argument("--table", dest="table", help="(Opcional) nome da tabela destino")
    p.add_argument("--replace", dest="replace", action="store_true", help="DROP + CREATE tabela antes de inserir")
    args = p.parse_args(argv)

    if pd is None:
        sys.stderr.write("ERRO: pandas não disponível. Instale pandas.\n")
        return 2

    df, meta = read_csv_smart(
        args.inp,
        encoding=args.encoding,
        delimiter=args.delimiter,
        quotechar=args.quotechar,
        has_header=False if args.no_header else None,
    )

    df.columns = canonicalize_headers(list(df.columns))

    profiling_report = {}
    prof = load_profile(args.profile) if args.profile else None
    if prof:
        profiling_report = validate_against_profile(df, prof)

    write_csv_utf8(df, args.out_csv)

    if args.db_sqlite and args.table:
        conn = sqlite3.connect(args.db_sqlite)
        try:
            ensure_sqlite_table(conn, args.table, list(df.columns), replace=args.replace)
            n = insert_sqlite_bulk(conn, args.table, df)
        finally:
            conn.close()
    else:
        n = len(df)

    sys.stdout.write(json.dumps({
        "input": args.inp,
        "detected": meta,
        "rows": int(n),
        "out_csv": args.out_csv,
        "profile_check": profiling_report
    }, ensure_ascii=False, indent=2) + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

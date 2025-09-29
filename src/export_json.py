from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def read_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path, dtype=str, keep_default_na=False, encoding="utf-8")


def dataframe_to_records(df: pd.DataFrame) -> List[Dict[str, object]]:
    if df.empty:
        return []
    return df.to_dict(orient="records")


def run(grid_csv: str, out_path: str, *, indent: int = 2, ensure_ascii: bool = False) -> Dict[str, object]:
    grid_df = read_csv(grid_csv)
    records = dataframe_to_records(grid_df)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=ensure_ascii, indent=indent)

    return {
        "out": str(out_file.resolve()),
        "rows": len(records),
        "columns": list(grid_df.columns),
    }


def parse_args(argv: None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export reconciliation grid as JSON")
    parser.add_argument("--grid", required=True, help="Caminho para o CSV da grade de reconciliação")
    parser.add_argument("--out", required=True, help="Caminho do arquivo JSON de saída")
    parser.add_argument("--indent", type=int, default=2, help="Nível de indentação para o JSON (default: 2)")
    parser.add_argument(
        "--ensure-ascii",
        action="store_true",
        help="Força escape ASCII no JSON (desabilitado por padrão)",
    )
    return parser.parse_args(argv)


def main(argv: None = None) -> int:
    args = parse_args(argv)
    result = run(args.grid, args.out, indent=args.indent, ensure_ascii=args.ensure_ascii)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

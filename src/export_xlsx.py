from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd

STATUS_SHEETS = {
    "OK": "OK",
    "ALERTA": "Alertas",
    "DIVERGENCIA": "Divergencias",
}


def read_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path, dtype=str, keep_default_na=False, encoding="utf-8")


def style_sheet(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame, freeze: bool = True) -> None:
    worksheet = writer.sheets[sheet_name]
    if freeze:
        worksheet.freeze_panes(1, 0)
    header_format = writer.book.add_format({"bold": True, "bg_color": "#111827", "font_color": "#f9fafb"})
    for col_idx, column in enumerate(df.columns):
        worksheet.write(0, col_idx, column, header_format)
        lengths = df[column].astype(str).replace({"nan":"","None":""}).str.len()
        valid_max = lengths[~lengths.isna()].max() if not lengths.empty else 12
        try:
            max_len = int(max(12, valid_max or 12))
        except (TypeError, ValueError):
            max_len = 12
        worksheet.set_column(col_idx, col_idx, min(max_len + 2, 60))


def build_sheets(grid_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    sheets: Dict[str, pd.DataFrame] = {}
    status_col = "match.status" if "match.status" in grid_df.columns else "status"
    for status, sheet_name in STATUS_SHEETS.items():
        if status_col in grid_df.columns:
            subset = grid_df[grid_df[status_col].str.upper() == status].copy()
        else:
            subset = pd.DataFrame()
        sheets[sheet_name] = subset
    return sheets


def run(grid_csv: str, sem_fonte_csv: str, sem_sucessor_csv: str, out_path: str) -> Dict[str, object]:
    grid_df = read_csv(grid_csv)
    sem_fonte_df = read_csv(sem_fonte_csv)
    sem_sucessor_df = read_csv(sem_sucessor_csv)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_file, engine="xlsxwriter") as writer:
        sheets = build_sheets(grid_df)
        for sheet_name, df in sheets.items():
            if df.empty:
                df = pd.DataFrame(columns=grid_df.columns)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            style_sheet(writer, sheet_name, df)

        sem_fonte_df.to_excel(writer, sheet_name="Sem Fonte", index=False)
        style_sheet(writer, "Sem Fonte", sem_fonte_df, freeze=False)

        sem_sucessor_df.to_excel(writer, sheet_name="Sem Sucessor", index=False)
        style_sheet(writer, "Sem Sucessor", sem_sucessor_df, freeze=False)

    return {
        "out": str(out_file),
        "ok": int(len(sheets["OK"])),
        "alertas": int(len(sheets["Alertas"])),
        "divergencias": int(len(sheets["Divergencias"])),
        "sem_fonte": int(len(sem_fonte_df)),
        "sem_sucessor": int(len(sem_sucessor_df)),
    }


def parse_args(argv: None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export reconciliation Excel report")
    parser.add_argument("--grid", required=True)
    parser.add_argument("--sem-fonte", required=True)
    parser.add_argument("--sem-sucessor", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: None = None) -> int:
    args = parse_args(argv)
    result = run(args.grid, args.sem_fonte, args.sem_sucessor, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

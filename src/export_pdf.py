from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    REPORTLAB_AVAILABLE = True
except Exception:  # pragma: no cover
    REPORTLAB_AVAILABLE = False


def read_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path, dtype=str, keep_default_na=False, encoding="utf-8")


def status_counts(grid_df: pd.DataFrame) -> Dict[str, int]:
    status_col = "match.status" if "match.status" in grid_df.columns else "status"
    vc = grid_df[status_col].str.upper().value_counts() if not grid_df.empty else {}
    return {
        "OK": int(vc.get("OK", 0)),
        "ALERTA": int(vc.get("ALERTA", 0)),
        "DIVERGENCIA": int(vc.get("DIVERGENCIA", 0)),
    }


def build_pdf(out_path: Path, grid_df: pd.DataFrame, cliente: Optional[str], periodo: Optional[str]) -> None:
    doc = SimpleDocTemplate(str(out_path), pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    title = "Relatorio de Conferencia"
    if cliente:
        title += f" - {cliente}"
    elements.append(Paragraph(title, styles["Title"]))

    subtitle = f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    if periodo:
        subtitle += f" | Periodo: {periodo}"
    elements.append(Paragraph(subtitle, styles["Normal"]))
    elements.append(Spacer(1, 12))

    counts = status_counts(grid_df)
    data = [
        ["Status", "Quantidade"],
        ["OK", counts["OK"]],
        ["Alerta", counts["ALERTA"]],
        ["Divergencia", counts["DIVERGENCIA"]],
    ]
    table = Table(data, colWidths=[200, 120])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.gray),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f3f4f6")),
            ]
        )
    )
    elements.append(table)

    doc.build(elements)


def build_html(out_path: Path, grid_df: pd.DataFrame, cliente: Optional[str], periodo: Optional[str]) -> None:
    counts = status_counts(grid_df)
    html = f"""
    <html><head><meta charset='utf-8'><title>Relatorio de Conferencia</title>
    <style>
      body {{ font-family: system-ui, sans-serif; padding: 32px; background: #f9fafb; color: #1f2937; }}
      table {{ border-collapse: collapse; margin-top: 20px; min-width: 300px; }}
      th, td {{ border: 1px solid #d1d5db; padding: 8px 12px; text-align: center; }}
      th {{ background: #111827; color: #f9fafb; }}
    </style></head><body>
      <h1>Relatorio de Conferencia</h1>
      <p>Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
    """
    if cliente:
        html += f"<p>Cliente: {cliente}</p>"
    if periodo:
        html += f"<p>Periodo: {periodo}</p>"

    html += f"""
      <table>
        <tr><th>Status</th><th>Quantidade</th></tr>
        <tr><td>OK</td><td>{counts['OK']}</td></tr>
        <tr><td>Alerta</td><td>{counts['ALERTA']}</td></tr>
        <tr><td>Divergencia</td><td>{counts['DIVERGENCIA']}</td></tr>
      </table>
    </body></html>
    """
    out_path.write_text(html, encoding="utf-8")


def run(grid_csv: str, out_path: str, cliente: Optional[str] = None, periodo: Optional[str] = None) -> Dict[str, object]:
    grid_df = read_csv(grid_csv)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if REPORTLAB_AVAILABLE:
        build_pdf(out_file, grid_df, cliente, periodo)
        return {
            "pdf": str(out_file),
            "stats": status_counts(grid_df),
        }
    else:
        html_out = out_file.with_suffix(".html")
        build_html(html_out, grid_df, cliente, periodo)
        return {
            "html": str(html_out),
            "stats": status_counts(grid_df),
        }


def parse_args(argv: None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export reconciliation PDF report (with HTML fallback)")
    parser.add_argument("--grid", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cliente")
    parser.add_argument("--periodo")
    return parser.parse_args(argv)


def main(argv: None = None) -> int:
    args = parse_args(argv)
    result = run(args.grid, args.out, args.cliente, args.periodo)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())



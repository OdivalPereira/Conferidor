# export_pdf.py — 18/28
# Gera um PDF com sumário executivo:
#  - Cabeçalho (cliente, período, data/hora)
#  - KPIs (contagens por status)
#  - Gráfico 1: barras por status
#  - Gráfico 2: top 10 motivos
#  - Tabela: top 50 divergências por |Δ Valor|
# Se ReportLab indisponível, gera HTML equivalente (fallback).
from __future__ import annotations
import argparse, json, os, io
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore

# Charts: matplotlib (sem seaborn, uma figura por gráfico, sem cores específicas)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # type: ignore

# PDF
try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_LEFT
except Exception:
    SimpleDocTemplate = None  # type: ignore

def _read_csv(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8")

def _to_float(x) -> Optional[float]:
    if x in (None, ""):
        return None
    try:
        return float(x)
    except Exception:
        s = str(x).strip().replace(".", "").replace(",", ".")
        try:
            return float(s)
        except Exception:
            return None

def _prep_grid(df: pd.DataFrame) -> pd.DataFrame:
    # normaliza tipos úteis
    for col in ["S.valor", "F.valor", "delta.valor", "match.score"]:
        if col in df.columns:
            df[col] = df[col].map(_to_float)
    return df

def _kpis(df: pd.DataFrame) -> Dict[str, int]:
    col = "match.status" if "match.status" in df.columns else "status"
    vals = df[col].fillna("NA").tolist() if col in df.columns else []
    def c(x): return sum(1 for v in vals if v == x)
    return {"OK": c("OK"), "ALERTA": c("ALERTA"), "DIVERGENCIA": c("DIVERGENCIA"), "SEM_FONTE": c("SEM_FONTE"), "SEM_SUCESSOR": c("SEM_SUCESSOR"), "TOTAL": len(vals)}

def _status_counts(df: pd.DataFrame) -> Dict[str, int]:
    counts = _kpis(df)
    return {k: counts[k] for k in ["OK","ALERTA","DIVERGENCIA","SEM_FONTE","SEM_SUCESSOR"]}

def _top_motivos(df: pd.DataFrame, n=10) -> List[tuple]:
    # conta tokens por ';' em match.motivos
    if "match.motivos" not in df.columns:
        return []
    from collections import Counter
    c = Counter()
    for m in df["match.motivos"].fillna("").tolist():
        for tok in [t.strip() for t in str(m).split(";") if t.strip()]:
            c[tok] += 1
    return c.most_common(n)

def _save_status_chart(counts: Dict[str,int], out_path: str) -> Optional[str]:
    if plt is None:
        return None
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=150)
    ax.bar(labels, values)  # sem cores definidas
    ax.set_title("Distribuição por Status")
    ax.set_xlabel("Status")
    ax.set_ylabel("Qtde")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

def _save_motivos_chart(items: List[tuple], out_path: str) -> Optional[str]:
    if plt is None or not items:
        return None
    labels = [k for k,_ in items]
    values = [v for _,v in items]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.barh(labels, values)  # sem cores definidas
    ax.set_title("Top 10 motivos (ocorrências)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

def _select_top_divs(df: pd.DataFrame, top_n=50) -> pd.DataFrame:
    df2 = df.copy()
    col = "match.status" if "match.status" in df2.columns else "status"
    if "delta.valor" not in df2.columns or col not in df2.columns:
        return pd.DataFrame()
    df2["_abs_delta"] = df2["delta.valor"].map(lambda x: abs(_to_float(x) or 0.0))
    df2 = df2[df2[col] == "DIVERGENCIA"]
    if df2.empty:
        return df2
    df2 = df2.sort_values(by="_abs_delta", ascending=False).head(top_n)
    keep = []
    for c in ["S.doc","F.doc","S.data","F.data","fonte_tipo","S.valor","F.valor","delta.valor","match.strategy","match.score","match.motivos"]:
        if c in df2.columns: keep.append(c)
    return df2[keep].copy()

def _fmt_money(x) -> str:
    if x is None: return ""
    try:
        return f"{float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(x)

def _build_pdf(
    grid_csv: str,
    out_pdf: str,
    cliente: Optional[str],
    periodo: Optional[str],
) -> Dict[str, Any]:
    assert pd is not None, "pandas requerido"
    df = _read_csv(grid_csv)
    df = _prep_grid(df)
    kpis = _kpis(df)
    counts = _status_counts(df)
    motivos = _top_motivos(df, 10)
    topdiv = _select_top_divs(df, 50)

    charts: List[str] = []
    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    c1 = _save_status_chart(counts, os.path.splitext(out_pdf)[0] + "_chart_status.png")
    if c1: charts.append(c1)
    c2 = _save_motivos_chart(motivos, os.path.splitext(out_pdf)[0] + "_chart_motivos.png")
    if c2: charts.append(c2)

    if SimpleDocTemplate is None:
        # Fallback HTML
        html_path = os.path.splitext(out_pdf)[0] + ".html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("<html><head><meta charset='utf-8'><title>Relatório</title></head><body>")
            f.write(f"<h2>Relatório de Conferência</h2>")
            f.write(f"<p><b>Cliente:</b> {cliente or ''} &nbsp;&nbsp; <b>Período:</b> {periodo or ''} &nbsp;&nbsp; <b>Gerado:</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>")
            f.write("<h3>KPI</h3><ul>")
            for k in ["TOTAL","OK","ALERTA","DIVERGENCIA","SEM_FONTE","SEM_SUCESSOR"]:
                f.write(f"<li>{k}: {kpis.get(k,0)}</li>")
            f.write("</ul>")
            for c in charts:
                f.write(f"<img src='{os.path.basename(c)}' style='max-width:720px; display:block; margin:12px 0;'/>")
            f.write("<h3>Top 50 divergências</h3>")
            f.write("<table border='1' cellpadding='4' cellspacing='0'>")
            if not topdiv.empty:
                f.write("<tr>")
                for col in topdiv.columns:
                    f.write(f"<th>{col}</th>")
                f.write("</tr>")
                for _, r in topdiv.iterrows():
                    f.write("<tr>")
                    for col in topdiv.columns:
                        val = r.get(col)
                        if col in ("S.valor","F.valor","delta.valor"):
                            val = _fmt_money(val)
                        f.write(f"<td>{val if val is not None else ''}</td>")
                    f.write("</tr>")
            f.write("</table></body></html>")
        return {"ok": True, "pdf": None, "html": html_path, "charts": charts, "kpis": kpis}

    # PDF com ReportLab
    doc = SimpleDocTemplate(out_pdf, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    style_h = styles["Heading2"]; style_h.alignment = TA_LEFT
    style_p = styles["BodyText"]

    story: List[Any] = []
    story.append(Paragraph("Relatório de Conferência", style_h))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Cliente:</b> {cliente or ''} &nbsp;&nbsp; <b>Período:</b> {periodo or ''} &nbsp;&nbsp; <b>Gerado:</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}", style_p))
    story.append(Spacer(1, 10))

    # KPIs
    kpi_data = [["Métrica", "Valor"],
                ["TOTAL", kpis.get("TOTAL",0)],
                ["OK", kpis.get("OK",0)],
                ["ALERTA", kpis.get("ALERTA",0)],
                ["DIVERGÊNCIA", kpis.get("DIVERGENCIA",0)],
                ["SEM FONTE", kpis.get("SEM_FONTE",0)],
                ["SEM SUCESSOR", kpis.get("SEM_SUCESSOR",0)]]
    t = Table(kpi_data, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.black),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.lightgrey]),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # Charts
    for c in charts:
        try:
            story.append(Image(c, width=480, height=280))
            story.append(Spacer(1, 6))
        except Exception:
            pass

    # Tabela de divergências
    story.append(Paragraph("Top 50 divergências por |Δ Valor|", style_h))
    story.append(Spacer(1, 6))
    if not topdiv.empty:
        # cabeçalho
        head = list(topdiv.columns)
        rows = [head]
        for _, r in topdiv.iterrows():
            row = []
            for col in head:
                val = r.get(col)
                if col in ("S.valor","F.valor","delta.valor"):
                    val = _fmt_money(val)
                row.append("" if val is None else str(val))
            rows.append(row)

        # criar tabela com largura flexível
        table = Table(rows, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.black),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("ALIGN", (0,0), (-1,-1), "LEFT"),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ]))
        story.append(table)
    else:
        story.append(Paragraph("Nenhuma divergência identificada no período.", style_p))

    doc.build(story)
    return {"ok": True, "pdf": out_pdf, "html": None, "charts": charts, "kpis": kpis}

def run_export_pdf(grid_csv: str, out_pdf: str, cliente: Optional[str], periodo: Optional[str]) -> Dict[str, Any]:
    return _build_pdf(grid_csv, out_pdf, cliente, periodo)

def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="export_pdf.py — Gera sumário executivo em PDF a partir da grid")
    p.add_argument("--grid", required=True, help="CSV da grade (reconciler --out-grid)")
    p.add_argument("--out", required=True, help="Arquivo PDF de saída")
    p.add_argument("--cliente", help="Nome do cliente para o cabeçalho")
    p.add_argument("--periodo", help="Período (ex.: 08/2025)")
    args = p.parse_args(argv)

    res = run_export_pdf(args.grid, args.out, args.cliente, args.periodo)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))

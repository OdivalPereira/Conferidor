import shutil
import json
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from loader import main as loader_main
from normalizer import main as normalizer_main
from matcher import main as matcher_main
from issues_engine import main as issues_main
from ui_dataset_builder import main as dataset_main
from export_xlsx import run as export_xlsx_run
from export_pdf import run as export_pdf_run
from run_pipeline import resolve_inputs

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
CFG = ROOT / "cfg"


def copy_fixture(tmp_dir: Path, name: str, target_name: Optional[str] = None) -> Path:
    target_name = target_name or name
    target = tmp_dir / target_name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / name, target)
    return target


def prepare_pipeline(tmp_path: Path):
    dados_dir = tmp_path / "dados"
    staging_dir = tmp_path / "out" / "staging"
    normalized_dir = tmp_path / "out" / "normalized"
    match_dir = tmp_path / "out" / "match"

    inputs = [
        copy_fixture(dados_dir, "sucessor.csv"),
        copy_fixture(dados_dir, "suprema_entradas.csv"),
        copy_fixture(dados_dir, "suprema_saidas.csv"),
        copy_fixture(dados_dir, "suprema_servicos.csv"),
        copy_fixture(dados_dir, "fornecedores.csv"),
        copy_fixture(dados_dir, "plano_contas.csv"),
    ]

    loader_args = [
        "--inputs",
        *[str(path) for path in inputs],
        "--profiles",
        str(CFG / "profiles_map.yml"),
        "--staging",
        str(staging_dir),
        "--log",
        str(tmp_path / "loader.jsonl"),
    ]
    assert loader_main(loader_args) == 0

    normalizer_args = [
        "--staging",
        str(staging_dir),
        "--out",
        str(normalized_dir),
        "--profiles",
        str(CFG / "profiles_map.yml"),
        "--tokens",
        str(CFG / "regex_tokens.yml"),
        "--log",
        str(tmp_path / "normalizer.jsonl"),
    ]
    assert normalizer_main(normalizer_args) == 0

    matcher_args = [
        "--sucessor",
        str(normalized_dir / "sucessor.parquet"),
        "--entradas",
        str(normalized_dir / "entradas.parquet"),
        "--saidas",
        str(normalized_dir / "saidas.parquet"),
        "--servicos",
        str(normalized_dir / "servicos.parquet"),
        "--cfg-pesos",
        str(CFG / "matching_pesos.yml"),
        "--cfop-map",
        str(CFG / "cfop_expectativas.csv"),
        "--out",
        str(match_dir),
        "--log",
        str(match_dir / "match_log.jsonl"),
    ]
    assert matcher_main(matcher_args) == 0

    issues_args = [
        "--grid",
        str(match_dir / "reconc_grid.csv"),
        "--rules",
        str(CFG / "issues_rules.yml"),
        "--out-issues",
        str(match_dir / "issues.jsonl"),
        "--out-grid",
        str(match_dir / "reconc_grid_issues.csv"),
    ]
    assert issues_main(issues_args) == 0

    dataset_args = [
        "--grid",
        str(match_dir / "reconc_grid_issues.csv"),
        "--sem-fonte",
        str(match_dir / "reconc_sem_fonte.csv"),
        "--sem-sucessor",
        str(match_dir / "reconc_sem_sucessor.csv"),
        "--out-jsonl",
        str(tmp_path / "out" / "ui_grid.jsonl"),
        "--meta",
        str(tmp_path / "out" / "ui_meta.json"),
        "--schema",
        str(CFG / "ui_schema.json"),
    ]
    assert dataset_main(dataset_args) == 0

    return match_dir, tmp_path / "out" / "ui_grid.jsonl", tmp_path / "out" / "ui_meta.json"


def test_full_pipeline(tmp_path):
    match_dir, grid_jsonl, meta_json = prepare_pipeline(tmp_path)

    grid_csv = match_dir / "reconc_grid.csv"
    assert grid_csv.exists()
    df = pd.read_csv(grid_csv, dtype=str)
    assert "match.score" in df.columns
    if not df.empty:
        assert df["match.status"].str.upper().isin(["OK", "ALERTA", "DIVERGENCIA"]).all()
    else:
        sem_fonte = match_dir / "reconc_sem_fonte.csv"
        sem_sucessor = match_dir / "reconc_sem_sucessor.csv"
        assert sem_fonte.exists() or sem_sucessor.exists()

    log_file = match_dir / "match_log.jsonl"
    assert log_file.exists()
    log_content = log_file.read_text(encoding="utf-8").strip()
    if log_content:
        json.loads(log_content.splitlines()[0])

    assert grid_jsonl.exists()
    first_line = grid_jsonl.read_text(encoding="utf-8").splitlines()[0]
    sample = json.loads(first_line)
    assert "status" in sample

    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    stats = meta.get("stats", {})
    assert stats.get("total_matches") == len(df)
    assert {"sem_fonte", "sem_sucessor"}.issubset(stats.keys())

def test_exports(tmp_path):
    grid_csv = tmp_path / "grid.csv"
    grid_csv.write_text("match.status,score,S.valor\nOK,75,100.0\n", encoding="utf-8")
    sem_fonte_csv = tmp_path / "sem_fonte.csv"
    sem_fonte_csv.write_text("S.row_id,S.doc\n1,NA\n", encoding="utf-8")
    sem_sucessor_csv = tmp_path / "sem_sucessor.csv"
    sem_sucessor_csv.write_text("fonte_tipo,F.doc\nENTRADA,123\n", encoding="utf-8")

    out_xlsx = tmp_path / "relatorio.xlsx"
    export_xlsx_run(str(grid_csv), str(sem_fonte_csv), str(sem_sucessor_csv), str(out_xlsx))
    assert out_xlsx.exists()

    out_pdf = tmp_path / "relatorio.pdf"
    result_pdf = export_pdf_run(str(grid_csv), str(out_pdf), cliente="Teste", periodo="2025")
    pdf_path = result_pdf.get("pdf")
    html_path = result_pdf.get("html") or result_pdf.get("download_html")
    assert (pdf_path and Path(pdf_path).exists()) or (html_path and Path(html_path).exists())



def test_autodetect_inputs(tmp_path):
    dados_dir = tmp_path / "dados"
    dados_dir.mkdir(parents=True, exist_ok=True)

    nome_sucessor = "CONSULTA DE LAN" + "Ç" + "AMENTOS DA EMPRESA 534 - OXIGENIO MODELO INDUSTRIA E COMERCIO DE GASES LTDA.csv"
    nome_entradas = "CONSULTA MOVIMENTO DE ENTRADAS - 534 - OXIGENIO MODELO INDUSTRIA E COMERCIO DE GASES LTDA.csv"
    nome_saidas = "CONSULTA MOVIMENTO DE SA" + "Í" + "DAS - 534 - OXIGENIO MODELO INDUSTRIA E COMERCIO DE GASES LTDA.csv"
    nome_servicos = "CONSULTA MOVIMENTO DE SERVI" + "Ç" + "OS - 534 - OXIGENIO MODELO INDUSTRIA E COMERCIO DE GASES LTDA.csv"
    nome_fornecedores = "CONSULTA DE FORNECEDORES.csv"
    nome_plano = "CONSULTA DO PLANO DE CONTAS 1 - DOURALEX - 9014 PARTICIPANTE.csv"

    copy_fixture(dados_dir, "sucessor.csv", nome_sucessor)
    copy_fixture(dados_dir, "suprema_entradas.csv", nome_entradas)
    copy_fixture(dados_dir, "suprema_saidas.csv", nome_saidas)
    copy_fixture(dados_dir, "suprema_servicos.csv", nome_servicos)
    copy_fixture(dados_dir, "fornecedores.csv", nome_fornecedores)
    copy_fixture(dados_dir, "plano_contas.csv", nome_plano)

    args = argparse.Namespace(
        sucessor=None,
        entradas=None,
        saidas=None,
        servicos=None,
        fornecedores=None,
        plano=None,
    )
    inputs = resolve_inputs(args, dados_dir)
    assert inputs["sucessor"].name == nome_sucessor
    assert inputs["entradas"].name == nome_entradas
    assert inputs["saidas"].name == nome_saidas
    assert inputs["servicos"].name == nome_servicos
    assert inputs["fornecedores"].name == nome_fornecedores
    assert inputs["plano"].name == nome_plano

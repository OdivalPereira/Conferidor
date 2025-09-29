import shutil
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loader import main as loader_main
from normalizer import main as normalizer_main
from matcher import main as matcher_main
from issues_engine import main as issues_main
from ui_dataset_builder import main as dataset_main
from export_xlsx import run as export_xlsx_run
from export_pdf import run as export_pdf_run
from run_pipeline import resolve_inputs, choose_normalized_table, main as run_pipeline_main

FIXTURES = ROOT / "tests" / "fixtures"
CFG = ROOT / "cfg"


def copy_fixture(tmp_dir: Path, name: str, target_name: Optional[str] = None) -> Path:
    target_name = target_name or name
    target = tmp_dir / target_name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / name, target)
    return target


def run_pipeline_with_inputs(tmp_path: Path, inputs: List[Path], *, force_csv: bool = False):
    dados_dir = tmp_path / "dados"
    staging_dir = tmp_path / "out" / "staging"
    normalized_dir = tmp_path / "out" / "normalized"
    match_dir = tmp_path / "out" / "match"

    for path in inputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            raise FileNotFoundError(path)

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

    if force_csv:
        for name in ["sucessor", "entradas", "saidas", "servicos"]:
            parquet_path = normalized_dir / f"{name}.parquet"
            if parquet_path.exists():
                parquet_path.unlink()

    matcher_inputs: Dict[str, Path] = {
        "sucessor": choose_normalized_table(normalized_dir, "sucessor"),
        "entradas": choose_normalized_table(normalized_dir, "entradas"),
        "saidas": choose_normalized_table(normalized_dir, "saidas"),
        "servicos": choose_normalized_table(normalized_dir, "servicos"),
    }

    matcher_args = [
        "--sucessor",
        str(matcher_inputs["sucessor"]),
        "--entradas",
        str(matcher_inputs["entradas"]),
        "--saidas",
        str(matcher_inputs["saidas"]),
        "--servicos",
        str(matcher_inputs["servicos"]),
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

    return match_dir, tmp_path / "out" / "ui_grid.jsonl", tmp_path / "out" / "ui_meta.json", matcher_inputs


def prepare_pipeline(tmp_path: Path, *, force_csv: bool = False):
    dados_dir = tmp_path / "dados"
    inputs = [
        copy_fixture(dados_dir, "sucessor.csv"),
        copy_fixture(dados_dir, "suprema_entradas.csv"),
        copy_fixture(dados_dir, "suprema_saidas.csv"),
        copy_fixture(dados_dir, "suprema_servicos.csv"),
        copy_fixture(dados_dir, "fornecedores.csv"),
        copy_fixture(dados_dir, "plano_contas.csv"),
    ]
    return run_pipeline_with_inputs(tmp_path, inputs, force_csv=force_csv)


def test_full_pipeline(tmp_path):
    match_dir, grid_jsonl, meta_json, _ = prepare_pipeline(tmp_path)

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


def test_pipeline_uses_csv_when_parquet_missing(tmp_path):
    match_dir, _, _, matcher_inputs = prepare_pipeline(tmp_path, force_csv=True)

    normalized_dir = tmp_path / "out" / "normalized"
    for name, path in matcher_inputs.items():
        assert path.suffix == ".csv"
        assert path.exists()
        assert not (normalized_dir / f"{name}.parquet").exists()

    grid_csv = match_dir / "reconc_grid.csv"
    assert grid_csv.exists()


def test_pipeline_handles_messy_inputs(tmp_path):
    dados_dir = tmp_path / "dados"

    sucessor_df = pd.DataFrame(
        [
            {
                "Transação": "1",
                "Débito": "1.1.1",
                "Crédito": "2.2.2",
                "Nº Docto": "NF 123",
                "Valor": "R$ 1.234,56",
                "Hist": "Venda mês janeiro",
                "Participante D": "João Comércio Ltda",
                "Participante C": "",
                "Data": "15-01-2024",
            }
        ]
    )
    entradas_df = pd.DataFrame(
        [
            {
                "Data Emissão": "2024/01/15",
                "Documento": "NF 123",
                "Nome Fornecedor": "  Joao  Comercio LTDA  ",
                "Valor Contábil": "1 234,56",
                "CFOP": "5102",
                "Modelo": "55",
                "Situação do Documento": "Autorizado",
                "Chave NF-e/CT-e/NFC-e/BP-e": "",
            }
        ]
    )
    saidas_df = pd.DataFrame(
        [
            {
                "Data Emissão": "15/01/2024",
                "Documento": "900001",
                "Nome Cliente": "Cliente Exemplo",
                "Valor Contábil": "0,00",
                "CFOP": "6108",
                "Modelo": "55",
                "Situação do Documento": "Cancelado",
            }
        ]
    )
    servicos_df = pd.DataFrame(
        [
            {
                "Data Emissão": "2024-01-15",
                "Nr. Doc.": "S-01",
                "Cliente": "Cliente Serviço",
                "Valor Contábil": "100,00",
                "Nat. Operação": "Serviço",
                "Situação": "Normal",
            }
        ]
    )

    def _write(df: pd.DataFrame, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, sep=";", index=False, encoding="latin1")
        return path

    sucessor_path = _write(sucessor_df, dados_dir / "sucessor.csv")
    entradas_path = _write(entradas_df, dados_dir / "suprema_entradas.csv")
    saidas_path = _write(saidas_df, dados_dir / "suprema_saidas.csv")
    servicos_path = _write(servicos_df, dados_dir / "suprema_servicos.csv")
    fornecedores_path = copy_fixture(dados_dir, "fornecedores.csv")
    plano_contas_path = copy_fixture(dados_dir, "plano_contas.csv")

    match_dir, _, _, _ = run_pipeline_with_inputs(
        tmp_path,
        [
            sucessor_path,
            entradas_path,
            saidas_path,
            servicos_path,
            fornecedores_path,
            plano_contas_path,
        ],
    )

    normalized_dir = tmp_path / "out" / "normalized"
    sucessor_normalized = pd.read_csv(normalized_dir / "sucessor.csv")
    entradas_normalized = pd.read_csv(normalized_dir / "entradas.csv")

    assert sucessor_normalized.loc[0, "valor"] == pytest.approx(1234.56, rel=1e-5)
    assert entradas_normalized.loc[0, "valor"] == pytest.approx(1234.56, rel=1e-5)

    assert sucessor_normalized.loc[0, "data_iso"] == "2024-01-15"
    assert entradas_normalized.loc[0, "data_iso"] == "2024-01-15"

    assert sucessor_normalized.loc[0, "participante_key"] == "JOAO COMERCIO LTDA"
    assert entradas_normalized.loc[0, "participante_key"] == "JOAO COMERCIO LTDA"

    grid_df = pd.read_csv(match_dir / "reconc_grid.csv")
    target_rows = grid_df.loc[grid_df["S.doc_num"].astype(str) == "123"]
    assert not target_rows.empty
    assert (target_rows["match.status"].str.upper() == "OK").all()
    assert (target_rows["F.participante"].str.strip() == "Joao Comercio LTDA").all()
    assert target_rows["match.score"].astype(float).ge(70).all()
    assert target_rows["delta.valor"].astype(float).abs().le(0.01).all()
    assert target_rows["delta.dias"].astype(float).abs().le(0.0).all()


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


def test_run_pipeline_incremental_skips_steps(tmp_path):
    dados_dir = tmp_path / "dados"
    for name in [
        "sucessor.csv",
        "suprema_entradas.csv",
        "suprema_saidas.csv",
        "suprema_servicos.csv",
        "fornecedores.csv",
        "plano_contas.csv",
    ]:
        copy_fixture(dados_dir, name)

    out_dir = tmp_path / "out"
    log_dir = tmp_path / "logs"
    argv = [
        "--dados-dir",
        str(dados_dir),
        "--cfg-dir",
        str(CFG),
        "--out-dir",
        str(out_dir),
        "--log-dir",
        str(log_dir),
        "--skip-ui",
    ]

    assert run_pipeline_main(argv) == 0

    log_path = log_dir / "pipeline.jsonl"
    assert log_path.exists()

    assert run_pipeline_main(argv) == 0

    content = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    skip_entries = [entry for entry in content if entry.get("event") == "skip"]
    assert {entry.get("step") for entry in skip_entries} == {"loader", "normalizer", "matcher", "issues_engine"}
    assert all(entry.get("status") == "skipped" for entry in skip_entries)

    cache_path = log_dir / "pipeline_cache.json"
    assert cache_path.exists()
    cache_doc = json.loads(cache_path.read_text(encoding="utf-8"))
    assert set(cache_doc.get("steps", {}).keys()).issuperset({"loader", "normalizer", "matcher", "issues_engine"})



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

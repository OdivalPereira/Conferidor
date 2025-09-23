import shutil
import json
from pathlib import Path

import pandas as pd

from loader import main as loader_main
from normalizer import main as normalizer_main
from matcher import main as matcher_main
from issues_engine import main as issues_main
from ui_dataset_builder import main as dataset_main

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
CFG = ROOT / "cfg"


def copy_fixture(tmp_dir: Path, name: str) -> Path:
    target = tmp_dir / name
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
    assert not df.empty
    assert "match.score" in df.columns
    assert df["match.status"].str.upper().isin(["OK", "ALERTA", "DIVERGENCIA"]).all()

    log_file = match_dir / "match_log.jsonl"
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert lines and json.loads(lines[0])

    assert grid_jsonl.exists()
    first_line = grid_jsonl.read_text(encoding="utf-8").splitlines()[0]
    sample = json.loads(first_line)
    assert "status" in sample

    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    assert meta["stats"]["total_matches"] >= 1

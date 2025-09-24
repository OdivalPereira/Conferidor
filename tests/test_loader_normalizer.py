import shutil
from pathlib import Path
from typing import Optional

from loader import main as loader_main
from normalizer import main as normalizer_main

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
CFG = ROOT / "cfg" / "profiles_map.yml"
TOKENS = ROOT / "cfg" / "regex_tokens.yml"


def copy_fixture(tmp_dir: Path, name: str, target_name: Optional[str] = None) -> Path:
    target_name = target_name or name
    target = tmp_dir / target_name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / name, target)
    return target


def test_loader_and_normalizer(tmp_path):
    dados_dir = tmp_path / "dados"
    staging_dir = tmp_path / "out" / "staging"
    normalized_dir = tmp_path / "out" / "normalized"

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
        str(CFG),
        "--staging",
        str(staging_dir),
        "--log",
        str(tmp_path / "loader.jsonl"),
    ]
    assert loader_main(loader_args) == 0

    normalized_args = [
        "--staging",
        str(staging_dir),
        "--out",
        str(normalized_dir),
        "--profiles",
        str(CFG),
        "--tokens",
        str(TOKENS),
        "--log",
        str(tmp_path / "normalizer.jsonl"),
    ]
    assert normalizer_main(normalized_args) == 0

    sucessor_csv = normalized_dir / "sucessor.csv"
    assert sucessor_csv.exists()
    content = sucessor_csv.read_text(encoding="utf-8")
    assert "doc_num" in content

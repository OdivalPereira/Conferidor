from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Optional, Set

# Ensure src/ is importable
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loader import main as loader_main
from normalizer import main as normalizer_main
from matcher import main as matcher_main
from issues_engine import main as issues_main
from ui_dataset_builder import main as ui_dataset_main

DEFAULT_INPUT_FILENAMES: Dict[str, str] = {
    "sucessor": "sucessor.csv",
    "entradas": "suprema_entradas.csv",
    "saidas": "suprema_saidas.csv",
    "servicos": "suprema_servicos.csv",
    "fornecedores": "fornecedores.csv",
    "plano": "plano_contas.csv",
}



AUTO_INPUT_PATTERNS: Dict[str, list[list[str]]] = {
    "sucessor": [["lancamento"], ["lancamentos"]],
    "entradas": [["movimento", "entrada"], ["entrada"]],
    "saidas": [["movimento", "saida"], ["saidas"], ["saida"]],
    "servicos": [["movimento", "servico"], ["servicos"], ["servico"]],
    "fornecedores": [["fornecedor"], ["fornecedores"]],
    "plano": [["plano", "contas"], ["plano", "conta"]],
}


def _normalise_filename(name: str) -> str:
    simplified = unicodedata.normalize("NFKD", name)
    without_accents = "".join(ch for ch in simplified if not unicodedata.combining(ch))
    return without_accents.lower()


def _discover_input_file(dados_dir: Path, key: str, used: Set[Path]) -> Optional[Path]:
    patterns = AUTO_INPUT_PATTERNS.get(key, [])
    if not patterns:
        return None
    candidates: list[tuple[int, int, str, Path]] = []
    for path in dados_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != ".csv":
            continue
        resolved = path.resolve()
        if resolved in used:
            continue
        normalised = _normalise_filename(path.name)
        best_match = 0
        for keywords in patterns:
            if all(keyword in normalised for keyword in keywords):
                best_match = max(best_match, len(keywords))
        if best_match:
            candidates.append((best_match, len(path.name), path.name.lower(), resolved))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return candidates[0][3]


def choose_normalized_table(normalized_dir: Path, name: str) -> Path:
    """Return the preferred normalized table file for *name*.

    The matcher can consume either Parquet or CSV outputs. Prefer Parquet when
    available, otherwise fall back to the CSV produced by the normalizer. When
    neither file exists yet, keep the Parquet path so downstream checks can
    surface the missing artefact.
    """

    parquet_path = normalized_dir / f"{name}.parquet"
    if parquet_path.exists():
        return parquet_path

    csv_path = normalized_dir / f"{name}.csv"
    if csv_path.exists():
        return csv_path

    return parquet_path
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full reconciliation pipeline (loader → normalizer → matcher → issues → UI dataset)."
    )
    parser.add_argument("--dados-dir", type=Path, default=Path("dados"), help="Directory containing the raw CSV inputs.")
    parser.add_argument("--cfg-dir", type=Path, default=Path("cfg"), help="Directory containing configuration files.")
    parser.add_argument("--out-dir", type=Path, default=Path("out"), help="Directory to write pipeline outputs.")
    parser.add_argument(
        "--profiles", type=Path, default=None, help="Override path to profiles_map.yml (defaults to <cfg-dir>/profiles_map.yml)."
    )
    parser.add_argument(
        "--matching-config",
        type=Path,
        default=None,
        help="Override path to matching_pesos.yml (defaults to <cfg-dir>/matching_pesos.yml).",
    )
    parser.add_argument(
        "--tokens",
        type=Path,
        default=None,
        help="Override path to regex_tokens.yml (defaults to <cfg-dir>/regex_tokens.yml).",
    )
    parser.add_argument(
        "--cfop-map",
        type=Path,
        default=None,
        help="Override path to cfop_expectativas.csv (defaults to <cfg-dir>/cfop_expectativas.csv).",
    )
    parser.add_argument(
        "--issues-rules",
        type=Path,
        default=None,
        help="Override path to issues_rules.yml (defaults to <cfg-dir>/issues_rules.yml).",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=None,
        help="Override path to ui_schema.json (defaults to <cfg-dir>/ui_schema.json).",
    )
    parser.add_argument(
        "--skip-ui",
        action="store_true",
        help="Skip the ui_dataset_builder step (still runs loader, normalizer, matcher and issues).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory to store step logs (defaults to <out-dir>).",
    )

    for key in DEFAULT_INPUT_FILENAMES:
        parser.add_argument(
            f"--{key}",
            type=Path,
            default=None,
            help=f"Path to {key} CSV (defaults to <dados-dir>/{DEFAULT_INPUT_FILENAMES[key]}).",
        )

    return parser


def resolve_inputs(args: argparse.Namespace, dados_dir: Path) -> Dict[str, Path]:
    inputs: Dict[str, Path] = {}
    used: Set[Path] = set()
    for key, filename in DEFAULT_INPUT_FILENAMES.items():
        override: Optional[Path] = getattr(args, key)
        base_path = (dados_dir / filename).resolve()
        chosen: Path
        if override is not None:
            chosen = override.resolve()
        elif base_path.exists():
            chosen = base_path
        else:
            discovered = _discover_input_file(dados_dir, key, used)
            if discovered is not None:
                chosen = discovered
                sys.stdout.write(f"[pipeline] Auto-detected {key}: {chosen.name}\\n")
            else:
                chosen = base_path
        inputs[key] = chosen
        if chosen.exists():
            used.add(chosen.resolve())
    return inputs


def require_files(description: str, files: Sequence[Path]) -> None:
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        sys.stderr.write(f"[pipeline] Missing {description}: {json.dumps(missing, ensure_ascii=False)}\n")
        sys.exit(2)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def call_step(name: str, func, args: Iterable[str]) -> None:
    sys.stdout.write(f"[pipeline] Running {name}...\n")
    exit_code = func(list(args))
    if exit_code != 0:
        sys.stderr.write(f"[pipeline] Step '{name}' failed with exit code {exit_code}.\n")
        sys.exit(exit_code)
    sys.stdout.write(f"[pipeline] Step '{name}' completed.\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dados_dir = args.dados_dir.resolve()
    cfg_dir = args.cfg_dir.resolve()
    out_dir = args.out_dir.resolve()
    log_dir = (args.log_dir or out_dir).resolve()

    ensure_dir(out_dir)
    ensure_dir(log_dir)

    profiles_path = (args.profiles or cfg_dir / "profiles_map.yml").resolve()
    matching_path = (args.matching_config or cfg_dir / "matching_pesos.yml").resolve()
    tokens_path = (args.tokens or cfg_dir / "regex_tokens.yml").resolve()
    cfop_path = (args.cfop_map or cfg_dir / "cfop_expectativas.csv").resolve()
    issues_rules_path = (args.issues_rules or cfg_dir / "issues_rules.yml").resolve()
    schema_path = (args.schema or cfg_dir / "ui_schema.json").resolve()

    input_paths = resolve_inputs(args, dados_dir)
    require_files("input CSVs", list(input_paths.values()))
    require_files(
        "configuration files",
        [profiles_path, matching_path, tokens_path, cfop_path, issues_rules_path],
    )
    if not args.skip_ui:
        require_files("UI schema", [schema_path])

    staging_dir = out_dir / "staging"
    normalized_dir = out_dir / "normalized"
    match_dir = out_dir / "match"

    ensure_dir(staging_dir)
    ensure_dir(normalized_dir)
    ensure_dir(match_dir)

    loader_args: List[str] = [
        "--inputs",
        *[str(path) for path in (
            input_paths["sucessor"],
            input_paths["entradas"],
            input_paths["saidas"],
            input_paths["servicos"],
            input_paths["fornecedores"],
            input_paths["plano"],
        )],
        "--profiles",
        str(profiles_path),
        "--staging",
        str(staging_dir),
        "--log",
        str(log_dir / "loader.jsonl"),
    ]
    call_step("loader", loader_main, loader_args)

    normalizer_args: List[str] = [
        "--staging",
        str(staging_dir),
        "--out",
        str(normalized_dir),
        "--profiles",
        str(profiles_path),
        "--tokens",
        str(tokens_path),
        "--log",
        str(log_dir / "normalizer.jsonl"),
    ]
    call_step("normalizer", normalizer_main, normalizer_args)

    matcher_args: List[str] = [
        "--sucessor",
        str(choose_normalized_table(normalized_dir, "sucessor")),
        "--entradas",
        str(choose_normalized_table(normalized_dir, "entradas")),
        "--saidas",
        str(choose_normalized_table(normalized_dir, "saidas")),
        "--servicos",
        str(choose_normalized_table(normalized_dir, "servicos")),
        "--cfg-pesos",
        str(matching_path),
        "--cfg-tokens",
        str(tokens_path),
        "--cfop-map",
        str(cfop_path),
        "--out",
        str(match_dir),
        "--log",
        str(match_dir / "match_log.jsonl"),
    ]
    call_step("matcher", matcher_main, matcher_args)

    issues_args: List[str] = [
        "--grid",
        str(match_dir / "reconc_grid.csv"),
        "--rules",
        str(issues_rules_path),
        "--out-issues",
        str(match_dir / "issues.jsonl"),
        "--out-grid",
        str(match_dir / "reconc_grid_issues.csv"),
    ]
    call_step("issues_engine", issues_main, issues_args)

    if not args.skip_ui:
        ui_args: List[str] = [
            "--grid",
            str(match_dir / "reconc_grid_issues.csv"),
            "--sem-fonte",
            str(match_dir / "reconc_sem_fonte.csv"),
            "--sem-sucessor",
            str(match_dir / "reconc_sem_sucessor.csv"),
            "--out-jsonl",
            str(out_dir / "ui_grid.jsonl"),
            "--meta",
            str(out_dir / "ui_meta.json"),
            "--schema",
            str(schema_path),
        ]
        call_step("ui_dataset_builder", ui_dataset_main, ui_args)

    sys.stdout.write("[pipeline] All steps completed successfully.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())



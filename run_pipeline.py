from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set

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


class PipelineCancelled(Exception):
    """Signal that the pipeline should abort due to an external request."""


_CANCEL_CHECK: Optional[Callable[[], None]] = None


def set_cancel_check(callback: Optional[Callable[[], None]]) -> None:
    """Register a callable that raises :class:`PipelineCancelled` when triggered.

    The callable will be invoked before and after each pipeline step, allowing an
    external controller (such as the FastAPI UI server) to signal cancellation
    between steps without modifying the CLI surface of the pipeline runner.
    """

    global _CANCEL_CHECK
    _CANCEL_CHECK = callback


def _ensure_not_cancelled() -> None:
    if _CANCEL_CHECK is not None:
        _CANCEL_CHECK()

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


@dataclass
class StepMetrics:
    rows_processed: Optional[int] = None
    inconsistencies: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {}
        if self.rows_processed is not None:
            record["rows_processed"] = self.rows_processed
        record["inconsistencies"] = self.inconsistencies
        if self.details:
            record["details"] = self.details
        return record


@dataclass
class StepContext:
    name: str
    args: List[str]
    log_path: Optional[Path]
    staging_dir: Path
    normalized_dir: Path
    match_dir: Path
    out_dir: Path
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepDefinition:
    name: str
    func: Callable[[List[str]], int]
    args: List[str]
    log_path: Optional[Path] = None
    metrics_collector: Optional[Callable[[StepContext], StepMetrics]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive path
                raise ValueError(f"Invalid JSON on line {line_no} of {path.name}: {exc}") from exc


def count_csv_rows(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        total = sum(1 for _ in handle)
    if total <= 0:
        return 0
    return max(total - 1, 0)


def count_jsonl_rows(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def loader_metrics(context: StepContext) -> StepMetrics:
    metrics = StepMetrics(rows_processed=0, details={"files_processed": 0})
    path = context.log_path
    if path is None or not path.exists():
        metrics.rows_processed = None
        metrics.details.pop("files_processed", None)
        metrics.inconsistencies.append("Arquivo de log da etapa ausente.")
        return metrics

    rows_total = 0
    files = 0
    inconsistencies: List[str] = []
    for record in iter_jsonl(path):
        files += 1
        rows_total += int(record.get("rows") or 0)
        status = str(record.get("status") or "").lower()
        if status and status not in {"written", "dry-run"}:
            source = Path(str(record.get("input") or "")).name or "(desconhecido)"
            inconsistencies.append(f"{source}: status={status}")
        details = record.get("details") or {}
        reason = details.get("reason")
        if reason and status != "written":
            source = Path(str(record.get("input") or "")).name or "(desconhecido)"
            inconsistencies.append(f"{source}: {reason}")

    metrics.rows_processed = rows_total
    metrics.details["files_processed"] = files
    metrics.inconsistencies = inconsistencies
    return metrics


def normalizer_metrics(context: StepContext) -> StepMetrics:
    metrics = StepMetrics(rows_processed=0, details={"datasets": {}})
    path = context.log_path
    if path is None or not path.exists():
        metrics.rows_processed = None
        metrics.details.pop("datasets", None)
        metrics.inconsistencies.append("Arquivo de log da etapa ausente.")
        return metrics

    rows_total = 0
    datasets: Dict[str, int] = {}
    inconsistencies: List[str] = []
    for record in iter_jsonl(path):
        dataset = str(record.get("dataset") or "desconhecido")
        rows = int(record.get("rows") or 0)
        rows_total += rows
        datasets[dataset] = rows
        if not record.get("csv", False):
            inconsistencies.append(f"{dataset}: CSV não gerado")
        if not record.get("parquet", False):
            inconsistencies.append(f"{dataset}: Parquet não gerado")

    metrics.rows_processed = rows_total
    metrics.details["datasets"] = datasets
    metrics.inconsistencies = inconsistencies
    return metrics


def matcher_metrics(context: StepContext) -> StepMetrics:
    grid_path = Path(context.extra.get("grid", context.match_dir / "reconc_grid.csv"))
    sem_fonte_path = Path(context.extra.get("sem_fonte", context.match_dir / "reconc_sem_fonte.csv"))
    sem_sucessor_path = Path(context.extra.get("sem_sucessor", context.match_dir / "reconc_sem_sucessor.csv"))
    metrics = StepMetrics(details={
        "grid": str(grid_path),
        "sem_fonte": str(sem_fonte_path),
        "sem_sucessor": str(sem_sucessor_path),
    })

    rows = count_csv_rows(grid_path)
    if rows is None:
        metrics.inconsistencies.append("Grid principal não encontrada após o matcher.")
    else:
        metrics.rows_processed = rows

    log_path = context.log_path
    if log_path is not None and log_path.exists():
        metrics.details["log_entries"] = count_jsonl_rows(log_path)
    else:
        metrics.inconsistencies.append("match_log.jsonl ausente.")

    sem_fonte_rows = count_csv_rows(sem_fonte_path)
    if sem_fonte_rows is None:
        metrics.inconsistencies.append("Arquivo reconc_sem_fonte.csv não encontrado.")
    elif sem_fonte_rows > 0:
        metrics.inconsistencies.append(f"Linhas sem fonte: {sem_fonte_rows}")
        metrics.details["sem_fonte_rows"] = sem_fonte_rows
    else:
        metrics.details["sem_fonte_rows"] = 0

    sem_sucessor_rows = count_csv_rows(sem_sucessor_path)
    if sem_sucessor_rows is None:
        metrics.inconsistencies.append("Arquivo reconc_sem_sucessor.csv não encontrado.")
    elif sem_sucessor_rows > 0:
        metrics.inconsistencies.append(f"Linhas sem Sucessor: {sem_sucessor_rows}")
        metrics.details["sem_sucessor_rows"] = sem_sucessor_rows
    else:
        metrics.details["sem_sucessor_rows"] = 0

    return metrics


def issues_metrics(context: StepContext) -> StepMetrics:
    issues_path = Path(context.extra.get("issues"))
    grid_out_path = Path(context.extra.get("grid_out"))
    metrics = StepMetrics(details={
        "issues": str(issues_path),
        "grid_out": str(grid_out_path),
    })

    rows = count_csv_rows(grid_out_path)
    if rows is None:
        metrics.inconsistencies.append("Grid anotada não encontrada após issues_engine.")
    else:
        metrics.rows_processed = rows

    issues_count = count_jsonl_rows(issues_path)
    if issues_count is None:
        metrics.inconsistencies.append("Arquivo issues.jsonl não encontrado.")
    else:
        metrics.details["issues_emitted"] = issues_count
        if issues_count > 0:
            metrics.inconsistencies.append(f"Issues emitidas: {issues_count}")

    return metrics


def ui_metrics(context: StepContext) -> StepMetrics:
    jsonl_path = context.log_path
    meta_path = Path(context.extra.get("meta"))
    metrics = StepMetrics(details={"meta": str(meta_path)})

    if jsonl_path is None or not jsonl_path.exists():
        metrics.rows_processed = None
        metrics.inconsistencies.append("Dataset JSONL da UI não foi gerado.")
    else:
        count = count_jsonl_rows(jsonl_path) or 0
        metrics.rows_processed = count
        metrics.details["jsonl"] = str(jsonl_path)

    if not meta_path.exists():
        metrics.inconsistencies.append("Arquivo ui_meta.json não encontrado.")
    else:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            metrics.inconsistencies.append(f"ui_meta.json inválido: {exc}")
        else:
            metrics.details["meta_stats_keys"] = sorted((meta.get("stats") or {}).keys())

    return metrics


def call_step(step: StepDefinition, pipeline_log: Path, *, staging_dir: Path, normalized_dir: Path, match_dir: Path, out_dir: Path) -> None:
    _ensure_not_cancelled()
    args = list(step.args)
    sys.stdout.write(f"[pipeline] Running {step.name}...\n")

    start_time = datetime.now(timezone.utc)
    start_perf = time.perf_counter()
    append_jsonl(
        pipeline_log,
        {
            "step": step.name,
            "event": "start",
            "timestamp": start_time.isoformat(),
            "args": args,
        },
    )

    exit_code = step.func(args)
    end_perf = time.perf_counter()
    end_time = datetime.now(timezone.utc)
    duration = end_perf - start_perf

    if exit_code != 0:
        append_jsonl(
            pipeline_log,
            {
                "step": step.name,
                "event": "end",
                "timestamp": end_time.isoformat(),
                "duration_seconds": duration,
                "status": "failed",
                "exit_code": exit_code,
            },
        )
        sys.stderr.write(f"[pipeline] Step '{step.name}' failed with exit code {exit_code}.\n")
        sys.exit(exit_code)

    context = StepContext(
        name=step.name,
        args=args,
        log_path=step.log_path,
        staging_dir=staging_dir,
        normalized_dir=normalized_dir,
        match_dir=match_dir,
        out_dir=out_dir,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=duration,
        extra=step.extra,
    )

    metrics = StepMetrics()
    if step.metrics_collector is not None:
        try:
            metrics = step.metrics_collector(context)
        except Exception as exc:  # pragma: no cover - defensive path
            metrics = StepMetrics()
            metrics.inconsistencies.append(f"Falha ao coletar métricas: {exc}")

    append_jsonl(
        pipeline_log,
        {
            "step": step.name,
            "event": "end",
            "timestamp": end_time.isoformat(),
            "duration_seconds": duration,
            "status": "ok",
            **metrics.as_record(),
        },
    )

    rows_text = "?" if metrics.rows_processed is None else str(metrics.rows_processed)
    inconsistencies_count = len(metrics.inconsistencies)
    sys.stdout.write(
        f"[pipeline] Step '{step.name}' completed in {duration:.2f}s | linhas={rows_text} | inconsistências={inconsistencies_count}.\n"
    )
    if metrics.inconsistencies:
        for item in metrics.inconsistencies:
            sys.stdout.write(f"[pipeline]   - {item}\n")

    _ensure_not_cancelled()


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

    pipeline_log_path = log_dir / "pipeline.jsonl"
    if pipeline_log_path.exists():
        pipeline_log_path.unlink()

    steps: List[StepDefinition] = []

    loader_log = log_dir / "loader.jsonl"
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
        str(loader_log),
    ]
    steps.append(
        StepDefinition(
            name="loader",
            func=loader_main,
            args=loader_args,
            log_path=loader_log,
            metrics_collector=loader_metrics,
            extra={
                "inputs": [
                    str(input_paths["sucessor"]),
                    str(input_paths["entradas"]),
                    str(input_paths["saidas"]),
                    str(input_paths["servicos"]),
                    str(input_paths["fornecedores"]),
                    str(input_paths["plano"]),
                ]
            },
        )
    )

    normalizer_log = log_dir / "normalizer.jsonl"
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
        str(normalizer_log),
    ]
    steps.append(
        StepDefinition(
            name="normalizer",
            func=normalizer_main,
            args=normalizer_args,
            log_path=normalizer_log,
            metrics_collector=normalizer_metrics,
        )
    )

    matcher_log = match_dir / "match_log.jsonl"
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
        str(matcher_log),
    ]
    steps.append(
        StepDefinition(
            name="matcher",
            func=matcher_main,
            args=matcher_args,
            log_path=matcher_log,
            metrics_collector=matcher_metrics,
            extra={
                "grid": match_dir / "reconc_grid.csv",
                "sem_fonte": match_dir / "reconc_sem_fonte.csv",
                "sem_sucessor": match_dir / "reconc_sem_sucessor.csv",
            },
        )
    )

    issues_out = match_dir / "issues.jsonl"
    issues_grid_out = match_dir / "reconc_grid_issues.csv"
    issues_args: List[str] = [
        "--grid",
        str(match_dir / "reconc_grid.csv"),
        "--rules",
        str(issues_rules_path),
        "--out-issues",
        str(issues_out),
        "--out-grid",
        str(issues_grid_out),
    ]
    steps.append(
        StepDefinition(
            name="issues_engine",
            func=issues_main,
            args=issues_args,
            metrics_collector=issues_metrics,
            extra={
                "issues": issues_out,
                "grid_out": issues_grid_out,
            },
        )
    )

    if not args.skip_ui:
        ui_jsonl = out_dir / "ui_grid.jsonl"
        ui_meta = out_dir / "ui_meta.json"
        ui_args: List[str] = [
            "--grid",
            str(issues_grid_out),
            "--sem-fonte",
            str(match_dir / "reconc_sem_fonte.csv"),
            "--sem-sucessor",
            str(match_dir / "reconc_sem_sucessor.csv"),
            "--out-jsonl",
            str(ui_jsonl),
            "--meta",
            str(ui_meta),
            "--schema",
            str(schema_path),
        ]
        steps.append(
            StepDefinition(
                name="ui_dataset_builder",
                func=ui_dataset_main,
                args=ui_args,
                log_path=ui_jsonl,
                metrics_collector=ui_metrics,
                extra={"meta": ui_meta},
            )
        )

    try:
        for step in steps:
            call_step(
                step,
                pipeline_log_path,
                staging_dir=staging_dir,
                normalized_dir=normalized_dir,
                match_dir=match_dir,
                out_dir=out_dir,
            )
    except PipelineCancelled:
        append_jsonl(
            pipeline_log_path,
            {
                "event": "pipeline",
                "status": "cancelled",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        sys.stderr.write("[pipeline] Execution cancelled before completion.\n")
        return 3

    append_jsonl(
        pipeline_log_path,
        {
            "event": "pipeline",
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    sys.stdout.write("[pipeline] All steps completed successfully.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())



from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------

@dataclass
class ProfileDetection:
    required_any: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ProfileDetection":
        required_any = [str(item) for item in data.get("required_any", [])]
        return cls(required_any=required_any)


@dataclass
class ProfileConfig:
    profile_id: str
    source: str
    detect: ProfileDetection
    csv_opts: Dict[str, str]
    column_map: Dict[str, List[str]]
    fixed: Dict[str, str]

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ProfileConfig":
        profile_id = str(data.get("id"))
        source = str(data.get("source"))
        detect = ProfileDetection.from_dict(data.get("detect", {}))
        csv_opts = {k: str(v) for k, v in (data.get("csv") or {}).items()}
        column_map = {
            str(target): [str(pattern) for pattern in patterns or []]
            for target, patterns in (data.get("map") or {}).items()
        }
        fixed = {str(k): str(v) for k, v in (data.get("fixed") or {}).items()}
        return cls(
            profile_id=profile_id,
            source=source,
            detect=detect,
            csv_opts=csv_opts,
            column_map=column_map,
            fixed=fixed,
        )


@dataclass
class ProfilesMap:
    defaults: Dict[str, str]
    profiles: List[ProfileConfig]

    @classmethod
    def load(cls, path: Path) -> "ProfilesMap":
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        defaults = {str(k): str(v) for k, v in (data.get("defaults") or {}).items()}
        profiles_raw = data.get("profiles") or []
        profiles = [ProfileConfig.from_dict(item) for item in profiles_raw]
        if not profiles:
            raise ValueError("profiles_map.yml contains no profile entries")
        return cls(defaults=defaults, profiles=profiles)


# ---------------------------------------------------------------------------
# Loader implementation
# ---------------------------------------------------------------------------

def _normalise_header(header: str) -> str:
    return re.sub(r"\s+", " ", header.strip())


def _build_header_list(path: Path, delimiter: str, encoding: str) -> List[str]:
    with path.open("r", encoding=encoding, errors="ignore", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            return []
    return [_normalise_header(col) for col in header]


def _score_profile(profile: ProfileConfig, headers: Iterable[str]) -> Tuple[int, List[str]]:
    matches: List[str] = []
    hits = 0
    total = max(len(profile.detect.required_any), 1)
    for pattern in profile.detect.required_any:
        rx = re.compile(pattern, re.IGNORECASE)
        for column in headers:
            if rx.search(column):
                matches.append(column)
                hits += 1
                break
    score = hits / total
    return score, matches


def detect_profile(path: Path, profiles_map: ProfilesMap) -> Tuple[Optional[ProfileConfig], Dict[str, object]]:
    delimiter = profiles_map.defaults.get("delimiter", ";")
    encoding = profiles_map.defaults.get("encoding", "utf-8")
    headers = _build_header_list(path, delimiter=delimiter, encoding=encoding)
    if not headers:
        return None, {"headers": [], "reason": "empty file"}

    best: Optional[ProfileConfig] = None
    best_score = -(10 ** 9)
    best_matches: List[str] = []
    filename = path.name.lower()
    hints = {
        "sucessor": "SUCESSOR",
        "entradas": "SUPREMA_ENTRADA",
        "saidas": "SUPREMA_SAIDA",
        "servico": "SUPREMA_SERVICO",
        "fornecedor": "FORNECEDORES",
        "plano": "PLANO_CONTAS",
    }

    for profile in profiles_map.profiles:
        delim = profile.csv_opts.get("delimiter", delimiter)
        enc = profile.csv_opts.get("encoding", encoding)
        header_list = _build_header_list(path, delimiter=delim, encoding=enc)
        if not header_list:
            continue
        score, matches = _score_profile(profile, header_list)
        for hint, source in hints.items():
            if hint in filename:
                if profile.source == source:
                    score += 0.2
                else:
                    score -= 0.1
        if score > best_score:
            best_score = score
            best = profile
            best_matches = matches

    details = {
        "headers": headers,
        "score": best_score,
        "matches": best_matches,
    }
    return best, details


def _read_csv(path: Path, profile: ProfileConfig, defaults: Dict[str, str]) -> pd.DataFrame:
    delimiter = profile.csv_opts.get("delimiter") or defaults.get("delimiter") or ";"
    encoding = profile.csv_opts.get("encoding") or defaults.get("encoding") or "utf-8"
    decimal = profile.csv_opts.get("decimal") or defaults.get("decimal") or ","
    try:
        df = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            encoding=encoding,
            sep=delimiter,
            engine="python",
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            encoding="latin1",
            sep=delimiter,
            engine="python",
        )
    df.columns = [_normalise_header(col) for col in df.columns]
    if decimal != ".":
        # keep raw values as text; decimal handling happens in normaliser
        pass
    return df


def _map_columns(df: pd.DataFrame, profile: ProfileConfig) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    result = pd.DataFrame()
    mapping: Dict[str, Optional[str]] = {}
    used_columns: set[str] = set()
    for target, patterns in profile.column_map.items():
        matched_column: Optional[str] = None
        for pattern in patterns:
            rx = re.compile(pattern, re.IGNORECASE)
            for column in df.columns:
                if column in used_columns:
                    continue
                if rx.search(column):
                    matched_column = column
                    break
            if matched_column:
                break
        if matched_column:
            result[target] = df[matched_column]
            used_columns.add(matched_column)
        else:
            result[target] = ""
        mapping[target] = matched_column
    for key, value in profile.fixed.items():
        result[key] = value
        mapping[key] = f"<fixed:{value}>"
    result["profile_id"] = profile.profile_id
    result["source"] = profile.source
    return result, mapping


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8")


def write_log(log_path: Path, record: Dict[str, object]) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


OUTPUT_NAME_MAP: Dict[str, str] = {
    "SUCESSOR": "sucessor.csv",
    "SUPREMA_ENTRADA": "suprema_entradas.csv",
    "SUPREMA_SAIDA": "suprema_saidas.csv",
    "SUPREMA_SERVICO": "suprema_servicos.csv",
    "PRACTICE": "practice.csv",
    "MISTER_CONTADOR": "mister_contador.csv",
    "FORNECEDORES": "fornecedores.csv",
    "PLANO_CONTAS": "plano_contas.csv",
    "LEGACY_BANK": "legacy_bank.csv",
}


def process_file(
    csv_path: Path,
    profiles_map: ProfilesMap,
    staging_dir: Path,
    log_path: Path,
    dry_run: bool = False,
) -> Dict[str, object]:
    profile, details = detect_profile(csv_path, profiles_map)
    record: Dict[str, object] = {
        "input": str(csv_path),
        "detected_profile": profile.profile_id if profile else None,
        "source": profile.source if profile else None,
        "details": details,
        "rows": 0,
        "out": None,
    }
    if profile is None:
        record["status"] = "skipped"
        write_log(log_path, record)
        return record

    df_raw = _read_csv(csv_path, profile, profiles_map.defaults)
    mapped_df, mapping = _map_columns(df_raw, profile)
    record["mapping"] = mapping
    record["rows"] = int(len(mapped_df))

    output_name = OUTPUT_NAME_MAP.get(profile.source, f"{profile.source.lower()}.csv")
    output_path = staging_dir / output_name
    record["out"] = str(output_path)
    if not dry_run:
        write_csv(mapped_df, output_path)
        record["status"] = "written"
    else:
        record["status"] = "dry-run"

    write_log(log_path, record)
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CSV loader that normalises headers using profiles_map.yml")
    parser.add_argument("--inputs", nargs="+", required=True, help="List of CSV files under dados/")
    parser.add_argument("--profiles", required=True, help="Path to profiles_map.yml")
    parser.add_argument("--staging", required=True, help="Directory for staging outputs")
    parser.add_argument("--log", default="out/logs/loader.jsonl", help="Path to append JSONL logs")
    parser.add_argument("--dry-run", action="store_true", help="Enable detection only, no files are written")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    staging_dir = Path(args.staging)
    profiles_path = Path(args.profiles)
    log_path = Path(args.log)

    if not profiles_path.exists():
        sys.stderr.write(f"profiles file not found: {profiles_path}\n")
        return 2

    profiles_map = ProfilesMap.load(profiles_path)
    results = []
    for item in args.inputs:
        csv_path = Path(item)
        if not csv_path.exists():
            record = {
                "input": str(csv_path),
                "status": "missing",
            }
            write_log(log_path, record)
            results.append(record)
            continue
        record = process_file(csv_path, profiles_map, staging_dir, log_path, dry_run=args.dry_run)
        results.append(record)

    summary = {
        "processed": len(results),
        "written": sum(1 for r in results if r.get("status") == "written"),
        "skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "dry_run": sum(1 for r in results if r.get("status") == "dry-run"),
        "missing": sum(1 for r in results if r.get("status") == "missing"),
    }
    sys.stdout.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())




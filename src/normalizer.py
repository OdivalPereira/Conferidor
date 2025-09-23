from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def strip_accents(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    import unicodedata

    normalized = unicodedata.normalize("NFD", str(text))
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalise_space(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    new_value = re.sub(r"\s+", " ", str(text)).strip()
    return new_value or None


def safe_upper(text: Optional[str]) -> Optional[str]:
    value = normalise_space(text)
    return value.upper() if value else None


def abs_or_none(value: Optional[float]) -> Optional[float]:
    return abs(value) if value is not None else None


def parse_decimal(text: Optional[str], decimal: str = ",") -> Optional[float]:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    raw = raw.replace(" ", "")
    if decimal == ",":
        raw = raw.replace(".", "").replace(",", ".")
    else:
        raw = raw.replace(",", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", raw)
    if not match:
        return None
    try:
        value = float(match.group(0))
    except ValueError:
        return None
    if "(" in raw and ")" in raw:
        value = -abs(value)
    return value


def parse_date(text: Optional[str], formats: Iterable[str]) -> Optional[datetime]:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    try:
        from dateutil import parser as dateutil_parser

        return dateutil_parser.parse(raw, dayfirst=True)
    except Exception:
        return None


def to_iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.strftime("%Y-%m-%d") if dt else None


def to_month(dt: Optional[datetime]) -> Optional[str]:
    return dt.strftime("%Y-%m") if dt else None


def extract_first(pattern: re.Pattern, text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    match = pattern.search(str(text))
    if not match:
        return None
    groups = match.groups()
    if groups:
        for item in groups:
            if item:
                return re.sub(r"\D", "", str(item)) or None
    return re.sub(r"\D", "", match.group(0)) or None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, payload: Dict[str, object]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_table(df: pd.DataFrame, out_dir: Path, name: str) -> Tuple[str, Optional[str]]:
    ensure_dir(out_dir)
    parquet_path = out_dir / f"{name}.parquet"
    csv_path = out_dir / f"{name}.csv"

    try:
        df.to_parquet(parquet_path, index=False)
        written_parquet = str(parquet_path)
    except Exception:
        written_parquet = None

    df.to_csv(csv_path, index=False, encoding="utf-8")
    return str(csv_path), written_parquet


def column(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series([None] * len(df), index=df.index, dtype=object)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NormalizerConfig:
    date_formats: List[str]
    doc_num_regex: re.Pattern
    doc_serie_regex: re.Pattern
    tokens_regex: List[Tuple[str, re.Pattern]]

    @classmethod
    def load(
        cls,
        profiles_path: Path,
        tokens_path: Optional[Path],
    ) -> "NormalizerConfig":
        doc = yaml.safe_load(profiles_path.read_text(encoding="utf-8")) or {}
        defaults = doc.get("defaults") or {}
        date_formats = [str(fmt) for fmt in defaults.get("date_formats", ["%d/%m/%Y"])]
        doc_num_pattern = re.compile(str(defaults.get("doc_num_regex", r"\b(\d{1,12})\b")), re.IGNORECASE)
        doc_serie_pattern = re.compile(str(defaults.get("doc_serie_regex", r"\b(\d{1,4})\b")), re.IGNORECASE)

        tokens_regex: List[Tuple[str, re.Pattern]] = []
        if tokens_path and tokens_path.exists():
            tokens_doc = yaml.safe_load(tokens_path.read_text(encoding="utf-8")) or {}
            for name, pattern in (tokens_doc.get("TOKENS") or {}).items():
                tokens_regex.append((str(name), re.compile(str(pattern), re.IGNORECASE)))

        return cls(
            date_formats=date_formats,
            doc_num_regex=doc_num_pattern,
            doc_serie_regex=doc_serie_pattern,
            tokens_regex=tokens_regex,
        )


# ---------------------------------------------------------------------------
# Normalisers per dataset
# ---------------------------------------------------------------------------


class DatasetNormaliser:
    def __init__(self, config: NormalizerConfig):
        self.config = config

    def _extract_tokens(self, *texts: Optional[str]) -> str:
        values: List[str] = []
        for text in texts:
            if not text:
                continue
            for token_name, pattern in self.config.tokens_regex:
                for match in pattern.finditer(str(text)):
                    value = match.group(1) if match.groups() else match.group(0)
                    if value:
                        values.append(f"{token_name}:{value}")
        return json.dumps(values, ensure_ascii=False)

    def normalise_sucessor(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["row_id"] = range(1, len(df) + 1)

        parts = df.reindex(columns=["part_d", "part_c"], fill_value="")
        def _combine_parts(row):
            values = [str(x).strip() for x in [row.get("part_d"), row.get("part_c")] if x not in (None, "", "nan", "NaN")]
            text = " ".join(values) if values else None
            return normalise_space(text)

        df["participante_combined"] = parts.apply(_combine_parts, axis=1)
        df["participante_key"] = df["participante_combined"].map(lambda x: safe_upper(strip_accents(x)))

        valor_raw = column(df, "valor")
        df["valor"] = valor_raw.map(lambda v: parse_decimal(v))
        df["valor_abs"] = df["valor"].map(abs_or_none)

        data_raw = column(df, "data")
        df["data_dt"] = data_raw.map(lambda v: parse_date(v, self.config.date_formats))
        df["data_iso"] = df["data_dt"].map(to_iso)
        df["mes_ref"] = df["data_dt"].map(to_month)

        doc_raw = column(df, "doc")
        historico_raw = column(df, "historico")

        df["doc_num"] = pd.Series(
            [
                extract_first(self.config.doc_num_regex, doc_raw.iloc[i] or historico_raw.iloc[i])
                for i in range(len(df))
            ],
            index=df.index,
        )
        df["doc_serie"] = pd.Series(
            [
                extract_first(self.config.doc_serie_regex, doc_raw.iloc[i] or historico_raw.iloc[i])
                for i in range(len(df))
            ],
            index=df.index,
        )

        df["tokens"] = pd.Series(
            [
                self._extract_tokens(doc_raw.iloc[i], historico_raw.iloc[i], df["participante_combined"].iloc[i])
                for i in range(len(df))
            ],
            index=df.index,
        )

        keep = [
            "row_id",
            "profile_id",
            "source",
            "data",
            "data_iso",
            "mes_ref",
            "valor",
            "valor_abs",
            "doc",
            "doc_num",
            "doc_serie",
            "historico",
            "tokens",
            "debito",
            "credito",
            "part_d",
            "part_c",
            "participante_combined",
            "participante_key",
            "transacao_id",
        ]
        for col in keep:
            if col not in df.columns:
                df[col] = None
        return df[keep]

    def normalise_fonte(self, df: pd.DataFrame, fonte_tipo: str) -> pd.DataFrame:
        df = df.copy()
        df["row_id"] = range(1, len(df) + 1)
        df["fonte_tipo"] = fonte_tipo

        participante_raw = column(df, "participante")
        df["participante"] = participante_raw.map(normalise_space)
        df["participante_key"] = df["participante"].map(lambda x: safe_upper(strip_accents(x)))

        valor_raw = column(df, "valor")
        df["valor"] = valor_raw.map(lambda v: parse_decimal(v))
        df["valor_abs"] = df["valor"].map(abs_or_none)

        data_raw = column(df, "data")
        df["data_dt"] = data_raw.map(lambda v: parse_date(v, self.config.date_formats))
        df["data_iso"] = df["data_dt"].map(to_iso)
        df["mes_ref"] = df["data_dt"].map(to_month)

        doc_raw = column(df, "doc")
        historico_raw = column(df, "historico")

        df["doc_num"] = pd.Series(
            [extract_first(self.config.doc_num_regex, doc_raw.iloc[i] or historico_raw.iloc[i]) for i in range(len(df))],
            index=df.index,
        )
        df["doc_serie"] = pd.Series(
            [extract_first(self.config.doc_serie_regex, doc_raw.iloc[i] or historico_raw.iloc[i]) for i in range(len(df))],
            index=df.index,
        )

        df["tokens"] = pd.Series(
            [self._extract_tokens(doc_raw.iloc[i], historico_raw.iloc[i], df["participante"].iloc[i]) for i in range(len(df))],
            index=df.index,
        )

        keep = [
            "row_id",
            "profile_id",
            "source",
            "fonte_tipo",
            "data",
            "data_iso",
            "mes_ref",
            "valor",
            "valor_abs",
            "doc",
            "doc_num",
            "doc_serie",
            "participante",
            "participante_key",
            "cfop",
            "modelo",
            "especie",
            "situacao",
            "debito_alias",
            "credito_alias",
            "condicao",
            "chave_xml",
            "tokens",
        ]
        for col in keep:
            if col not in df.columns:
                df[col] = None
        return df[keep]

    def normalise_fornecedores(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["codigo"] = column(df, "codigo")
        df["nome"] = column(df, "nome").map(normalise_space)
        df["cnpj"] = column(df, "cnpj").map(lambda v: re.sub(r"\D", "", str(v)) if v else None)
        df["nome_key"] = df["nome"].map(lambda x: safe_upper(strip_accents(x)))
        return df[["codigo", "nome", "nome_key", "cnpj"]]

    def normalise_plano(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["codigo"] = column(df, "codigo")
        df["alias"] = column(df, "alias").map(normalise_space)
        df["nome"] = column(df, "nome").map(normalise_space)
        df["natureza"] = column(df, "natureza").map(normalise_space)
        df["alias_key"] = df["alias"].map(lambda x: safe_upper(strip_accents(x)))
        return df[["codigo", "alias", "alias_key", "nome", "natureza"]]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

STAGING_TO_KIND: Dict[str, str] = {
    "sucessor.csv": "sucessor",
    "suprema_entradas.csv": "entradas",
    "suprema_saidas.csv": "saidas",
    "suprema_servicos.csv": "servicos",
    "practice.csv": "practice",
    "mister_contador.csv": "mister",
    "fornecedores.csv": "fornecedores",
    "plano_contas.csv": "plano",
}


_KIND_TO_FONTE = {
    "entradas": "ENTRADA",
    "saidas": "SAIDA",
    "servicos": "SERVICO",
    "practice": "PRACTICE",
    "mister": "MISTER",
}


def run_normaliser(args: argparse.Namespace) -> Dict[str, object]:
    staging_dir = Path(args.staging)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    config = NormalizerConfig.load(Path(args.profiles), Path(args.tokens) if args.tokens else None)
    normaliser = DatasetNormaliser(config)

    summary: Dict[str, Dict[str, object]] = {}

    for file_name, kind in STAGING_TO_KIND.items():
        csv_path = staging_dir / file_name
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8")
        if kind == "sucessor":
            out_df = normaliser.normalise_sucessor(df)
        elif kind in _KIND_TO_FONTE:
            tipo = _KIND_TO_FONTE[kind]
            out_df = normaliser.normalise_fonte(df, tipo)
        elif kind == "fornecedores":
            out_df = normaliser.normalise_fornecedores(df)
        elif kind == "plano":
            out_df = normaliser.normalise_plano(df)
        else:
            out_df = df.copy()

        csv_written, parquet_written = write_table(out_df, out_dir, kind)
        record = {
            "rows": int(len(out_df)),
            "csv": csv_written,
            "parquet": parquet_written,
        }
        summary[kind] = record
        write_jsonl(Path(args.log), {"dataset": kind, **record})

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalises staging CSV files into typed datasets")
    parser.add_argument("--staging", required=True, help="Directory with loader outputs")
    parser.add_argument("--out", required=True, help="Directory to write normalised tables")
    parser.add_argument("--profiles", required=True, help="Path to profiles_map.yml")
    parser.add_argument("--tokens", default="cfg/regex_tokens.yml", help="Regex tokens YAML")
    parser.add_argument("--log", default="out/logs/normaliser.jsonl", help="Log file path")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    summary = run_normaliser(args)
    sys.stdout.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())



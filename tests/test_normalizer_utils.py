import pandas as pd
from pathlib import Path

from normalizer import DatasetNormaliser, NormalizerConfig, parse_decimal, extract_first

CFG_PATH = Path(__file__).resolve().parents[1] / "cfg" / "profiles_map.yml"
TOKENS_PATH = Path(__file__).resolve().parents[1] / "cfg" / "regex_tokens.yml"


def test_parse_decimal_br_format():
    assert parse_decimal("1.234,56") == 1234.56
    assert parse_decimal("(200,00)") == -200.0


def test_extract_first_doc_regex():
    config = NormalizerConfig.load(CFG_PATH, TOKENS_PATH)
    match = extract_first(config.doc_num_regex, "NF 12345/1")
    assert match == "12345"


def test_normalise_sucessor_builds_tokens():
    config = NormalizerConfig.load(CFG_PATH, TOKENS_PATH)
    normaliser = DatasetNormaliser(config)
    df = pd.DataFrame(
        {
            "data": ["01/08/2025"],
            "doc": ["NF 12345/1"],
            "valor": ["1.000,00"],
            "part_d": ["Fornecedor ABC"],
            "part_c": ["Banco XYZ"],
            "historico": ["Compra NF 12345"],
            "debito": ["1.1.1"],
            "credito": ["2.2.2"],
        }
    )
    result = normaliser.normalise_sucessor(df)
    assert result.loc[0, "doc_num"] == "12345"
    assert result.loc[0, "valor"] == 1000.0
    tokens = result.loc[0, "tokens"]
    assert "NFE_NUM_SERIE:12345" in tokens

import json
from pathlib import Path

import pandas as pd

from matcher import MatchConfig, build_candidates


def make_config(tmp_path: Path) -> MatchConfig:
    config_path = tmp_path / "matching.yml"
    config_path.write_text(
        """
limiares:
  auto_match: 70
  pendente_min: 50
  janela_default_dias: 3
  max_candidates: 5
tolerancias:
  valor:
    abs: 0.01
    pct: 0.001
  data:
    janela_dias_por_fonte:
      ENTRADA: 3
estrategias:
  S1: 40
  S2: 35
  S3: 30
  S4: 25
  S5: 20
bonus:
  valor_dentro_tolerancia: 20
  valor_exato_centavos: 5
  data_mesma: 5
  token_forte: 10
  participante_exato: 5
penalidades:
  valor_fora_tolerancia: -10
  data_fora_janela: -5
desempate:
  prioridade_fonte:
    ENTRADA: 1.0
        """.strip(),
        encoding="utf-8",
    )
    return MatchConfig.load(config_path)


def make_sucessor_row(**overrides) -> dict:
    base = {
        "row_id": 1,
        "valor": "100.00",
        "data_iso": "2025-01-01",
        "tokens": json.dumps(["TOK:BASE"]),
        "participante_key": "CLIENTE A",
        "doc_num": "NF123",
        "mes_ref": "2025-01",
        "debito": "1.1.1",
        "credito": "2.2.2",
    }
    base.update(overrides)
    return base


def make_fonte_row(**overrides) -> dict:
    base = {
        "row_id": 10,
        "fonte_tipo": "ENTRADA",
        "valor": "100.00",
        "data_iso": "2025-01-01",
        "tokens": json.dumps(["TOK:BASE"]),
        "participante_key": "CLIENTE A",
        "doc_num": "NF123",
        "mes_ref": "2025-01",
        "cfop": "5102",
        "situacao": "AUT",
        "participante": "Cliente A",
    }
    base.update(overrides)
    return base


def build_single_candidate(sucessor_row: dict, fonte_row: dict, cfg: MatchConfig):
    sucessor_df = pd.DataFrame([sucessor_row])
    fontes_df = pd.DataFrame([fonte_row])
    return build_candidates(sucessor_df, fontes_df, cfg, {})


def test_strategy_s1_when_doc_participant_and_value_match(tmp_path):
    cfg = make_config(tmp_path)
    candidates = build_single_candidate(make_sucessor_row(), make_fonte_row(), cfg)
    assert any(c.strategy == "S1" for c in candidates)


def test_strategy_s2_when_doc_and_participant_with_date(tmp_path):
    cfg = make_config(tmp_path)
    sucessor = make_sucessor_row()
    fonte = make_fonte_row(valor="150.00")  # outside tolerance => value_ok False
    candidates = build_single_candidate(sucessor, fonte, cfg)
    assert any(c.strategy == "S2" for c in candidates)
    assert all(c.strategy != "S1" for c in candidates)


def test_strategy_s3_when_doc_and_value_and_date_match(tmp_path):
    cfg = make_config(tmp_path)
    sucessor = make_sucessor_row(participante_key="CLIENTE X")
    fonte = make_fonte_row(participante_key="CLIENTE Y")
    candidates = build_single_candidate(sucessor, fonte, cfg)
    assert any(c.strategy == "S3" for c in candidates)


def test_strategy_s4_when_participant_value_and_date_match(tmp_path):
    cfg = make_config(tmp_path)
    sucessor = make_sucessor_row(doc_num="NF-999")
    fonte = make_fonte_row(doc_num="NF-888")
    candidates = build_single_candidate(sucessor, fonte, cfg)
    assert any(c.strategy == "S4" for c in candidates)


def test_strategy_s5_when_tokens_value_and_date_match(tmp_path):
    cfg = make_config(tmp_path)
    token = json.dumps(["TOKEN:SHARED"])
    sucessor = make_sucessor_row(doc_num="", participante_key="CLIENTE Z", tokens=token)
    fonte = make_fonte_row(doc_num="", participante_key="OUTRO", tokens=token)
    candidates = build_single_candidate(sucessor, fonte, cfg)
    assert any(c.strategy == "S5" for c in candidates)


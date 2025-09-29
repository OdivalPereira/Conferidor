from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_pipeline import ConfigurationError, validate_configurations

CFG_DIR = ROOT / "cfg"


def write_copy(src: Path, dst: Path) -> None:
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def test_validate_configurations_accepts_defaults(tmp_path: Path) -> None:
    matching = tmp_path / "matching.yml"
    rules = tmp_path / "rules.yml"
    write_copy(CFG_DIR / "matching_pesos.yml", matching)
    write_copy(CFG_DIR / "issues_rules.yml", rules)

    validate_configurations(matching, rules)


def test_validate_configurations_invalid_matching(tmp_path: Path) -> None:
    matching = tmp_path / "matching.yml"
    rules = tmp_path / "rules.yml"
    write_copy(CFG_DIR / "issues_rules.yml", rules)

    matching.write_text(
        """
estrategias:
  S1: cem
  S2: 40
limiares:
  auto_match: 70
  pendente_min: 50
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError) as excinfo:
        validate_configurations(matching, rules)

    assert "matching_pesos.yml" in str(excinfo.value)


def test_validate_configurations_invalid_rules(tmp_path: Path) -> None:
    matching = tmp_path / "matching.yml"
    rules = tmp_path / "rules.yml"
    write_copy(CFG_DIR / "matching_pesos.yml", matching)

    rules.write_text(
        """
rules:
  - id: R001
    emit:
      message: { value: 123 }
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError) as excinfo:
        validate_configurations(matching, rules)

    assert "issues_rules.yml" in str(excinfo.value)

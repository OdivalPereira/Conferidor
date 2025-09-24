import csv
import json
from pathlib import Path

from issues_engine import run_issues


def test_issues_engine_emits_issue_and_updates_status(tmp_path):
    grid_path = tmp_path / "grid.jsonl"
    grid_row = {
        "id": "row-1",
        "status": "OK",
        "delta.valor": "12.50",
        "S.valor": "100.00",
        "F.valor": "87.50",
    }
    grid_path.write_text(json.dumps(grid_row, ensure_ascii=False) + "\n", encoding="utf-8")

    rules_path = tmp_path / "rules.yml"
    rules_path.write_text(
        """
rules:
  - id: delta_maior_que_5
    when:
      all:
        - field: "delta.valor"
          op: abs_gt
          value: 5
    emit:
      code: DELTA
      mark_status: ALERTA
      message: "Diferença {delta.valor}"
        """.strip(),
        encoding="utf-8",
    )

    out_issues = tmp_path / "issues.jsonl"
    out_grid = tmp_path / "grid_out.csv"

    result = run_issues(str(grid_path), str(rules_path), str(out_issues), str(out_grid))
    assert result["rows"] == 1

    issues_lines = out_issues.read_text(encoding="utf-8").strip().splitlines()
    assert len(issues_lines) == 1
    issue = json.loads(issues_lines[0])
    assert issue["code"] == "DELTA"
    assert "12.50" in issue["message"]

    with out_grid.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert rows[0]["status"] == "ALERTA"
    assert rows[0]["issues"] == "DELTA"
    assert "Diferença" in rows[0]["issues_msgs"]


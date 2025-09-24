from loader import ProfilesMap, detect_profile, process_file


def test_loader_detects_profile_and_maps_columns(tmp_path):
    profiles_yml = tmp_path / "profiles_map.yml"
    profiles_yml.write_text(
        """
version: 1
defaults:
  delimiter: ";"
  encoding: "utf-8"
profiles:
  - id: simples
    source: SUCESSOR
    detect:
      required_any:
        - '(?i)^data$'
        - '(?i)^valor$'
    csv:
      delimiter: ";"
      decimal: ","
      encoding: "utf-8"
    map:
      data:
        - '(?i)^data$'
      valor:
        - '(?i)^valor$'
      doc:
        - '(?i)^documento$'
    fixed:
      fonte_tipo: SUCESSOR
        """.strip(),
        encoding="utf-8",
    )

    csv_path = tmp_path / "custom_sucessor.csv"
    csv_path.write_text("Data;Valor;Documento\n01/08/2025;1000,00;NF123\n", encoding="utf-8")

    profiles_map = ProfilesMap.load(profiles_yml)
    profile, details = detect_profile(csv_path, profiles_map)
    assert profile is not None
    assert profile.profile_id == "simples"
    assert profile.source == "SUCESSOR"
    assert details["matches"], "expected header matches to be tracked"

    staging_dir = tmp_path / "staging"
    log_path = tmp_path / "loader.jsonl"
    record = process_file(csv_path, profiles_map, staging_dir, log_path, dry_run=True)

    assert record["detected_profile"] == "simples"
    assert record["rows"] == 1
    assert record["mapping"]["data"] == "Data"
    assert record["mapping"]["doc"] == "Documento"
    assert record["mapping"]["fonte_tipo"] == "<fixed:SUCESSOR>"

    output_path = staging_dir / "sucessor.csv"
    assert not output_path.exists(), "dry-run mode should not create output files"


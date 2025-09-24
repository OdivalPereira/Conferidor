Project tree for Conferidor

|-- cfg
|   -- cfop_expectativas.csv
|   -- issues_rules.yml
|   -- matching_pesos.yml
|   -- profiles_map.yml
|   -- regex_tokens.yml
|   -- ui_schema.json
|-- dados
|   -- (CSV de entrada por cliente/per?odo)
|-- out
|   -- (gerado pelo pipeline)
|-- schemas
|   -- schema_conferencia_postgres.sql
|   -- schema_conferencia_sqlite.sql
|-- src
|   -- export_pdf.py
|   -- export_xlsx.py
|   -- issues_engine.py
|   -- loader.py
|   -- matcher.py
|   -- normalizer.py
|   -- ui_app.html
|   -- ui_dataset_builder.py
|   -- ui_server.py
|-- tests
|   -- conftest.py
|   -- fixtures/
|      -- fornecedores.csv
|      -- plano_contas.csv
|      -- sucessor.csv
|      -- suprema_entradas.csv
|      -- suprema_saidas.csv
|      -- suprema_servicos.csv
|   -- test_issues_engine.py
|   -- test_loader_normalizer.py
|   -- test_loader_profiles.py
|   -- test_matcher_strategies.py
|   -- test_normalizer_utils.py
|   -- test_pipeline.py
|-- run_pipeline.py
|-- README_implantacao.md
|-- requirements.txt
|-- agente.md
|-- CHANGELOG.md
|-- TREE.md
-- (outros diret?rios ocultos/.devcontainer/.vscode)

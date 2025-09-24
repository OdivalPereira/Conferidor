Project tree for Conferidor

|-- .devcontainer
|   -- devcontainer.json
|-- .vscode
|   |-- launch.json
|   -- tasks.json
|-- cfg
|   |-- cfop_expectativas.csv
|   |-- issues_rules.yml
|   |-- matching_pesos.yml
|   |-- profiles_map.yml
|   |-- regex_tokens.yml
|   -- ui_schema.json
|-- dados
|-- out
|-- schemas
|   |-- schema_conferencia_postgres.sql
|   -- schema_conferencia_sqlite.sql
|-- src
|   |-- export_pdf.py
|   |-- export_xlsx.py
|   |-- issues_engine.py
|   |-- loader.py
|   |-- matcher.py
|   |-- normalizer.py
|   |-- ui_app.html
|   |-- ui_dataset_builder.py
|   -- ui_server.py
|-- tests
|   |-- conftest.py
|   |-- test_loader_normalizer.py
|   |-- test_normalizer_utils.py
|   |-- test_pipeline.py
|   -- fixtures
|       |-- fornecedores.csv
|       |-- plano_contas.csv
|       |-- sucessor.csv
|       |-- suprema_entradas.csv
|       |-- suprema_saidas.csv
|       -- suprema_servicos.csv
|-- agente.md
|-- README_implantacao.md
-- TREE.md

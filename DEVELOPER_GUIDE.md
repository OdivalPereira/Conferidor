# Guia de Desenvolvimento do Conferidor

Este documento descreve o fluxo de trabalho do projeto, os componentes principais e os procedimentos recomendados para instalar, executar, testar e contribuir com o Conferidor.

## Visão geral do produto

O Conferidor automatiza a conferência de lançamentos contábeis comparando a base "Sucessor" com múltiplas fontes fiscais (Suprema Entradas, Saídas, Serviços etc.). O pipeline lê arquivos CSV, normaliza os dados, executa estratégias de matching, aplica regras de issues e gera tanto conjuntos de dados para uma interface web quanto relatórios finais (XLSX/PDF).

A pasta `src/` concentra os scripts responsáveis por cada etapa e a API/FastAPI que serve os dados processados. Configurações vivem em `cfg/` e os insumos de clientes em `dados/` (um cliente por período). Saídas são gravadas em `out/`.

## Fluxo do pipeline

1. **Ingestão (`src/loader.py`)** – Lê CSVs de entrada utilizando perfis declarados em `cfg/profiles_map.yml` e grava arquivos intermediários em `out/staging/`.
2. **Normalização (`src/normalizer.py`)** – Ajusta tipos, formatos de valores, datas e tokens, produzindo arquivos Parquet em `out/normalized/`.
3. **Matching (`src/matcher.py`)** – Executa estratégias S1…S5, aplica pesos (`cfg/matching_pesos.yml`), tokens (`cfg/regex_tokens.yml`) e validações de CFOP (`cfg/cfop_expectativas.csv`), gerando resultados em `out/match/`.
4. **Regras de issues (`src/issues_engine.py`)** – Reclassifica combinações com base em `cfg/issues_rules.yml`, adicionando status e motivos às linhas.
5. **Dataset da UI (`src/ui_dataset_builder.py`)** – Produz `out/ui_grid.jsonl` e `out/ui_meta.json`, consumidos pela interface web.
6. **Serviços de UI e exports** – `src/ui_server.py` sobe uma API FastAPI (por padrão usando `DATA_DIR=out`). `src/export_xlsx.py` e `src/export_pdf.py` geram relatórios finais.

## Dependências

### Python

- Python 3.11+.
- Dependências listadas em `requirements.txt` (FastAPI, Uvicorn, Pandas, Polars, DuckDB, PyYAML, Python-Dateutil, XlsxWriter, ReportLab, Matplotlib, python-multipart).

### JavaScript

- Node.js 20 LTS (ou superior compatível).
- Dependências de desenvolvimento definidas em `package.json` para linting com ESLint + Prettier.

## Configuração do ambiente local

1. **Preparar ambiente Python**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -U pip
   pip install -r requirements.txt
   ```
2. **Instalar ferramentas de JavaScript (opcional, apenas para lint)**
   ```bash
   npm install
   ```
3. **Organizar insumos**
   - Coloque CSVs do cliente em `dados/`.
   - Revise/ajuste perfis e configurações em `cfg/` conforme necessário.
   - Garanta que `out/` exista (será populada automaticamente pelos scripts).

## Executando o pipeline localmente

Você pode executar os scripts de forma sequencial diretamente pelo Python:

```bash
source .venv/bin/activate
python src/loader.py --inputs dados/sucessor.csv dados/suprema_entradas.csv dados/suprema_saidas.csv dados/suprema_servicos.csv dados/fornecedores.csv dados/plano_contas.csv --profiles cfg/profiles_map.yml --staging out/staging
python src/normalizer.py --staging out/staging --out out/normalized --profiles cfg/profiles_map.yml
python src/matcher.py --sucessor out/normalized/sucessor.parquet --entradas out/normalized/entradas.parquet --saidas out/normalized/saidas.parquet --servicos out/normalized/servicos.parquet --fornecedores out/normalized/fornecedores.parquet --plano-contas out/normalized/plano_contas.parquet --cfg-pesos cfg/matching_pesos.yml --cfg-tokens cfg/regex_tokens.yml --cfop-map cfg/cfop_expectativas.csv --out out/match
python src/issues_engine.py --match-dir out/match --rules cfg/issues_rules.yml --out out/match
python src/ui_dataset_builder.py --match-dir out/match --schema cfg/ui_schema.json --out-dir out
```

Para explorar os resultados na interface web, execute o servidor FastAPI:

```bash
DATA_DIR=out uvicorn src.ui_server:app --reload
```

Abra `http://127.0.0.1:8000` para acessar os endpoints (ex.: `/api/grid`) e utilize `src/ui_app.html` para uma visualização local.

### Relatórios

- Excel: `python src/export_xlsx.py --match-dir out/match --out out/relatorio_conferencia.xlsx`
- PDF: `python src/export_pdf.py --match-dir out/match --out out/relatorio_conferencia.pdf`

## Executando testes e verificações

1. **Testes Python**
   ```bash
   source .venv/bin/activate
   pytest
   ```
2. **Lint JavaScript** (opcional, caso modifique arquivos `.js`)
   ```bash
   npm run lint
   ```

Os testes utilizam `pytest.ini` para adicionar `src/` ao `PYTHONPATH` e suprimir warnings de depreciação.

## Boas práticas de contribuição

1. **Branching** – Trabalhe em branches separados com nomes descritivos (`feature/...`, `fix/...`).
2. **Padrões de código**
   - Siga a estrutura modular existente dos scripts e mantenha os dados dentro de `out/`.
   - Preserve comentários e explique alterações relevantes em arquivos de configuração.
   - Evite try/except envolvendo apenas imports.
3. **Commits** – Faça commits pequenos e autoexplicativos. Inclua contexto sobre ajustes em pesos, perfis ou regras.
4. **Testes obrigatórios** – Execute `pytest` e `npm run lint` (se aplicável) antes de abrir PR.
5. **Documentação e changelog** – Atualize `CHANGELOG.md` quando alterar regras do pipeline ou comportamento funcional relevante.
6. **Revisão** – Certifique-se de que o pipeline completo roda (`loader` → `normalizer` → `matcher` → `issues` → `ui_dataset_builder`) e que os relatórios são gerados sem erros antes de solicitar revisão.

Seguindo estas orientações, novos contribuidores terão um roteiro claro para preparar o ambiente, entender o fluxo principal e entregar mudanças consistentes.

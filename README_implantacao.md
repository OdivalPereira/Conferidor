# App de Conferência — Guia de Implantação (28/28)

> Stack: **Python 3.11+**, CSVs locais, **FastAPI** (backend), **UI HTML** single‑file, exports **XLSX/PDF**. Ambiente 100% offline opcional.

---

## 1) Pré‑requisitos
- Python **3.11+** (3.12 OK).
- `pip install` dos pacotes (use um venv):
  ```bash
  pip install fastapi uvicorn pydantic pandas polars duckdb pyyaml python-dateutil xlsxwriter reportlab matplotlib
  ```
  > `reportlab` é opcional (gera PDF nativo). Sem ele, o export gera **HTML** equivalente.
### Ambiente virtual (.venv)
- Crie o ambiente com `python -m venv .venv`.
- Ative no Windows PowerShell com `./.venv/Scripts/Activate.ps1`, no CMD com `./.venv/Scripts/activate.bat` e no Linux/macOS com `source .venv/bin/activate`.
- Depois de ativar, execute `pip install -r requirements.txt`.
- As tasks do VS Code e os scripts do projeto assumem `.venv/bin/python` (Unix) ou `.venv/Scripts/python.exe` (Windows); ajuste se usar outro nome de venv.


---

## 2) Estrutura sugerida de pastas
```
projeto-conferencia/
├─ cfg/                                    # configuração por cliente/período
│  ├─ cfop_expectativas.csv
│  ├─ issues_rules.yml
│  ├─ matching_pesos.yml
│  ├─ profiles_map.yml
│  ├─ regex_tokens.yml
│  └─ ui_schema.json
├─ dados/                                  # coloque aqui os CSVs do cliente/período
├─ out/                                    # saídas geradas (pipeline)
├─ schemas/
│  ├─ schema_conferencia_postgres.sql
│  └─ schema_conferencia_sqlite.sql
├─ src/                                    # scripts do projeto
│  ├─ export_pdf.py
│  ├─ export_xlsx.py
│  ├─ issues_engine.py
│  ├─ loader.py
│  ├─ matcher.py
│  ├─ normalizer.py
│  ├─ ui_app.html
│  ├─ ui_dataset_builder.py
│  └─ ui_server.py
├─ tests/
│  ├─ conftest.py
│  ├─ fixtures/
│  ├─ test_issues_engine.py
│  ├─ test_loader_normalizer.py
│  ├─ test_loader_profiles.py
│  ├─ test_matcher_strategies.py
│  ├─ test_normalizer_utils.py
│  └─ test_pipeline.py
├─ run_pipeline.py
├─ README_implantacao.md
└─ requirements.txt
```
## 3) Preparar os CSVs (escopo aceito)
- Um **cliente** por vez e **um período** (ex.: 08/2025).
- Upload **manual** para `dados/`.
- Cabeçalhos podem variar: o **profiles_map.yml** (em `cfg/`) mapeia colunas → campos padronizados.

> Dica: abra os CSVs e verifique separador (`;`), decimal (`,`), **encoding UTF‑8**. Ajuste em `profiles_map.yml` se necessário.

---

## 4) Pipeline ponta‑a‑ponta

> Nota: o fluxo legado `src/reconciler.py` foi descontinuado; use `matcher.py -> issues_engine.py -> ui_dataset_builder.py` para gerar `reconc_grid.csv` e derivados.


### 4.0 Pipeline automático
```bash
python run_pipeline.py
```
> Use parâmetros como `--dados-dir`, `--out-dir` ou `--skip-ui` para personalizar o fluxo.

> O script detecta automaticamente arquivos como `CONSULTA MOVIMENTO DE ENTRADAS ...` e `CONSULTA DE LANÇAMENTOS ...` dentro de `dados/`; use `--entradas`/`--sucessor` etc. apenas se quiser forçar um caminho específico.

#### Notas para Windows/PowerShell
- Se um comando usar `export VAR=valor`, execute `Set-Item -Path Env:VAR -Value valor`.
- Substitua `rm -rf caminho` por `Remove-Item caminho -Recurse -Force`.
- Use `Select-String -Pattern texto arquivo` no lugar de `grep texto arquivo`.

### 4.1 Carregar e validar (loader → normalizer)
```bash
cd projeto-conferencia
python src/loader.py   --inputs dados/sucessor.csv dados/suprema_entradas.csv dados/suprema_saidas.csv dados/suprema_servicos.csv dados/fornecedores.csv dados/plano_contas.csv   --profiles cfg/profiles_map.yml   --staging out/staging
# Normalização (datas, valores, doc_num/serie, tokens)
python src/normalizer.py   --staging out/staging   --out out/normalized   --profiles cfg/profiles_map.yml
```
Saídas esperadas em `out/normalized/`: `sucessor.parquet`, `entradas.parquet`, `saidas.parquet`, `servicos.parquet`, `fornecedores.parquet`, `plano_contas.parquet` (ou CSVs/JSONL conforme implementação).

### 4.2 Matching (S1…S5 + validação CFOP)
```bash
python src/matcher.py   --sucessor out/normalized/sucessor.parquet   --entradas out/normalized/entradas.parquet   --saidas out/normalized/saidas.parquet   --servicos out/normalized/servicos.parquet   --fornecedores out/normalized/fornecedores.parquet   --plano-contas out/normalized/plano_contas.parquet   --cfg-pesos cfg/matching_pesos.yml   --cfg-tokens cfg/regex_tokens.yml   --cfop-map cfg/cfop_expectativas.csv   --out out/match
```
Saídas esperadas em `out/match/`: `reconc_grid.csv`, `reconc_sem_fonte.csv`, `reconc_sem_sucessor.csv`, `match_log.jsonl`.

### 4.3 Regras de Issues (classificação final)
```bash
python src/issues_engine.py   --grid out/match/reconc_grid.csv   --rules cfg/issues_rules.yml   --out-issues out/match/issues.jsonl   --out-grid out/match/reconc_grid_issues.csv
```
Resultado: a grid anotada já pode ajustar `status` (ALERTA/DIVERGÊNCIA) conforme regras.

### 4.4 Dataset para UI
```bash
python src/ui_dataset_builder.py   --grid out/match/reconc_grid_issues.csv   --meta out/ui_meta.json   --schema cfg/ui_schema.json   --out-jsonl out/ui_grid.jsonl
```
Gera:
- `out/ui_grid.jsonl` — linhas para a grade (com `status`, `score`, `S.*`, `F.*`, `delta.*`, `tags`, `motivos`).
- `out/ui_meta.json` — KPIs por status, presets, legendas.  
> O `ui_schema.json` é lido diretamente de `cfg/`.

### 4.5 Servidor local + UI
Em um terminal:
```bash
export DATA_DIR=out
uvicorn src.ui_server:app --reload --port 8000
```
Abra `src/ui_app.html` no navegador (duplo‑clique).  
- O HTML detecta automaticamente a variavel global `window.UI_API_BASE` (defina-a no console do navegador ou em um script antes de abrir o arquivo) ou o parametro `?api=http://host:porta` na URL; por padrao usa `http://localhost:8000`.
- Clique em uma linha da grid para abrir o painel lateral e usar os botoes de "Marcar OK" ou "Marcar Divergencia"; o botao "Reverter" volta para o status original.

### 4.6 Exportação
**Excel**:
```bash
python src/export_xlsx.py   --grid out/match/reconc_grid.csv   --sem-fonte out/match/reconc_sem_fonte.csv   --sem-sucessor out/match/reconc_sem_sucessor.csv   --out out/relatorio_conferencia.xlsx
```
**PDF / HTML**:
```bash
python src/export_pdf.py   --grid out/match/reconc_grid.csv   --out out/relatorio_conferencia.pdf   --cliente "Nome do Cliente"   --periodo "08/2025"
```

---

## 5) Arquivos de configuração (o que ajustar)
- `cfg/matching_pesos.yml` — pesos, bônus/penalidades, tolerâncias e janelas por fonte (auto‑match ≥ 70).
- `cfg/regex_tokens.yml` — regex para pistas de histórico (NFE x/y, PIX E2E/TXID, NSU Cielo/Rede, DCTO, boleto).
- `cfg/cfop_expectativas.csv` — expectativa de contas por CFOP (bônus/penalidade na coerência).
- `cfg/ui_schema.json` — colunas, cores, filtros e atalhos da grid.
- `cfg/profiles_map.yml` — mapeia cabeçalhos variados de CSV para campos padronizados.
- `cfg/issues_rules.yml` — regras pós‑matching (valor fora da tolerância, cancelados, devoluções, ST, etc.).

---

## 6) Qualidade e auditoria
- **Logs** `match_log.jsonl` e `issues.jsonl` com `row_id`, estratégia S1…S5, campos usados, deltas e decisão.
- **Reprodutibilidade**: mesmo input → mesmo output. Versione `cfg/` por **cliente/período**.
- **Precisão recomendada**: `precision_auto-match ≥ 99%` (valide amostras).

---

## 7) Performance
- Use **Polars** ou **DuckDB** no `matcher`/`normalizer` para 50k+ linhas.
- Crie *blocking keys* (doc/valor/participante/mês) para listar candidatos em O(log N).
- Prefira JSONL para streams (`ui_grid.jsonl`).

---

## 8) Troubleshooting (objetivo, sem floreio)
- **CSV não carrega** → confira `delimiter/decimal/encoding` em `profiles_map.yml` e cabeçalhos exigidos por perfil.
- **Pouco match** → relaxe janelas em `matching_pesos.yml` e habilite S4/S5 (tokens). Verifique `doc_num_regex` em `profiles_map.yml`.
- **CFOP incoerente** → ajuste `cfop_expectativas.csv` e aliases de conta na origem.
- **Export PDF falhou** → sem `reportlab`: gere HTML equivalente (script já faz fallback).
- **UI vazia** → `ui_server.py` precisa ver `out/ui_grid.jsonl`. Verifique `DATA_DIR=out` e `/api/health`.

---

## 9) Roadmap pós‑MVP
- Multiempresa e histórico longitudinal (banco persistente).
- Editor visual de regras (issues) e matriz CFOP.
- Integração leve com rede/drive para coleta automática (mantendo CSV como fonte da verdade).

---

## 10) Licença e autoria
Projeto desenhado para **open‑source** voltado a escritórios contábeis. Adapte e publique conforme sua política.

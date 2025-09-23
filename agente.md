# agente.md — Manual de Uso por Agentes de IA (VS Code + GPT‑5/Codex‑style)

> Este repositório contém um **pipeline local** de conferência (CSV → normalização → matching → issues → dataset UI → relatórios).  
> Este manual ensina **agentes de IA que operam no VS Code** (ex.: GPT‑5/Codex‑style) a trabalhar dentro do projeto com segurança, previsibilidade e auditoria.

---

## 1) Papel do agente e objetivos
- **Meta**: ajudar a automatizar a conferência entre **Sucessor** (contábil) e **fontes** (Suprema Entradas/Saídas/Serviços, Practice, Mister Contador), classificando cada linha em **OK / ALERTA / DIVERGÊNCIA** e emitindo relatórios (XLSX/PDF).
- **Escopo** (MVP): **um cliente por período**, ingestão **CSV** via upload manual, execução **local/offline** quando necessário.
- **Como o agente atua**: edita arquivos do projeto, cria/ajusta `tasks.json` e `launch.json`, executa scripts Python via **VS Code Tasks**, roda servidor da UI e monta exports.

**Regra de ouro**: *tudo que o agente fizer deve ser reprodutível via Tasks/Launch e registrado em logs `.jsonl`.*

---

## 2) Setup de ambiente (automatizável pelo agente)
1. **Python 3.11+** e venv:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -U pip
   pip install fastapi uvicorn pydantic pandas polars duckdb pyyaml python-dateutil xlsxwriter reportlab matplotlib
   ```
2. **Estrutura de pastas** (esperada):
   ```text
   dados/    # CSVs do cliente/período
   cfg/      # configurações (YAML/CSV/JSON)
   out/      # saídas geradas
   src/      # código-fonte (scripts Python + UI)
   .vscode/  # tasks/launch/settings para VS Code
   ```
3. **Workspace Trust**: habilitar confiança para executar tarefas e debug. (VS Code exige trust para rodar Tasks/Debug.)

> Nota: Tasks VS Code rodam processos, provêm automação e ficam em `.vscode/tasks.json`. Launch (debug) usa `.vscode/launch.json`.**

---

## 3) Scripts e “ferramentas” que o agente pode invocar
Todos em `src/`:
- `loader.py` — carrega CSVs → `out/staging` (aplica profiles de cabeçalhos).
- `normalizer.py` — limpa tipos/valores/datas/tokens → `out/normalized/`.
- `matcher.py` — estratégias S1…S5 + validação CFOP → `out/match/`.
- `issues_engine.py` — aplica `issues_rules.yml` e ajusta status → grid anotada.
- `ui_dataset_builder.py` — gera `out/ui_grid.jsonl`, `out/ui_meta.json` (UI).
- `ui_server.py` — FastAPI para servir a UI/exports (usa `DATA_DIR=out`).
- `export_xlsx.py` / `export_pdf.py` — relatórios finais.

Arquivos de configuração relevantes em `cfg/`:
- `profiles_map.yml`, `matching_pesos.yml`, `regex_tokens.yml`, `cfop_expectativas.csv`, `issues_rules.yml`, `ui_schema.json`.

---

## 4) VS Code — Tasks/Launch para orquestração

Crie **.vscode/tasks.json** com tarefas idempotentes (o agente **pode** gerar/atualizar este arquivo):

```jsonc
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "01: loader",
      "type": "shell",
      "command": "${workspaceFolder}/.venv/bin/python",
      "windows": { "command": "${workspaceFolder}\.venv\Scripts\python.exe" },
      "args": [
        "src/loader.py",
        "--inputs",
        "dados/sucessor.csv",
        "dados/suprema_entradas.csv",
        "dados/suprema_saidas.csv",
        "dados/suprema_servicos.csv",
        "dados/fornecedores.csv",
        "dados/plano_contas.csv",
        "--profiles", "cfg/profiles_map.yml",
        "--staging", "out/staging"
      ],
      "group": "build",
      "problemMatcher": []
    },
    {
      "label": "02: normalizer",
      "type": "shell",
      "command": "${workspaceFolder}/.venv/bin/python",
      "windows": { "command": "${workspaceFolder}\.venv\Scripts\python.exe" },
      "args": [
        "src/normalizer.py",
        "--staging", "out/staging",
        "--out", "out/normalized",
        "--profiles", "cfg/profiles_map.yml"
      ]
    },
    {
      "label": "03: matcher",
      "type": "shell",
      "command": "${workspaceFolder}/.venv/bin/python",
      "windows": { "command": "${workspaceFolder}\.venv\Scripts\python.exe" },
      "args": [
        "src/matcher.py",
        "--sucessor", "out/normalized/sucessor.parquet",
        "--entradas", "out/normalized/entradas.parquet",
        "--saidas", "out/normalized/saidas.parquet",
        "--servicos", "out/normalized/servicos.parquet",
        "--fornecedores", "out/normalized/fornecedores.parquet",
        "--plano-contas", "out/normalized/plano_contas.parquet",
        "--cfg-pesos", "cfg/matching_pesos.yml",
        "--cfg-tokens", "cfg/regex_tokens.yml",
        "--cfop-map", "cfg/cfop_expectativas.csv",
        "--out", "out/match"
      ]
    },
    {
      "label": "04: issues",
      "type": "shell",
      "command": "${workspaceFolder}/.venv/bin/python",
      "windows": { "command": "${workspaceFolder}\.venv\Scripts\python.exe" },
      "args": [
        "src/issues_engine.py",
        "--grid", "out/match/reconc_grid.csv",
        "--rules", "cfg/issues_rules.yml",
        "--out-issues", "out/match/issues.jsonl",
        "--out-grid", "out/match/reconc_grid_issues.csv"
      ]
    },
    {
      "label": "05: dataset UI",
      "type": "shell",
      "command": "${workspaceFolder}/.venv/bin/python",
      "windows": { "command": "${workspaceFolder}\.venv\Scripts\python.exe" },
      "args": [
        "src/ui_dataset_builder.py",
        "--grid", "out/match/reconc_grid_issues.csv",
        "--meta", "out/ui_meta.json",
        "--schema", "cfg/ui_schema.json",
        "--out-jsonl", "out/ui_grid.jsonl"
      ]
    },
    {
      "label": "UI server (uvicorn)",
      "type": "shell",
      "options": { "env": { "DATA_DIR": "out" } },
      "command": "${workspaceFolder}/.venv/bin/uvicorn",
      "windows": { "command": "${workspaceFolder}\.venv\Scripts\uvicorn.exe" },
      "args": ["src.ui_server:app", "--port", "8000", "--reload"],
      "problemMatcher": []
    },
    {
      "label": "Relatório XLSX",
      "type": "shell",
      "command": "${workspaceFolder}/.venv/bin/python",
      "windows": { "command": "${workspaceFolder}\.venv\Scripts\python.exe" },
      "args": [
        "src/export_xlsx.py",
        "--grid", "out/match/reconc_grid.csv",
        "--sem-fonte", "out/match/reconc_sem_fonte.csv",
        "--sem-sucessor", "out/match/reconc_sem_sucessor.csv",
        "--out", "out/relatorio_conferencia.xlsx"
      ]
    },
    {
      "label": "Relatório PDF",
      "type": "shell",
      "command": "${workspaceFolder}/.venv/bin/python",
      "windows": { "command": "${workspaceFolder}\.venv\Scripts\python.exe" },
      "args": [
        "src/export_pdf.py",
        "--grid", "out/match/reconc_grid.csv",
        "--out", "out/relatorio_conferencia.pdf",
        "--cliente", "Cliente X",
        "--periodo", "08/2025"
      ]
    }
  ]
}
```

Crie **.vscode/launch.json** para iniciar o servidor local em modo debug:

```jsonc
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug UI Server",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": ["src.ui_server:app", "--port", "8000", "--reload"],
      "env": { "DATA_DIR": "out" },
      "jinja": true,
      "console": "integratedTerminal"
    }
  ]
}
```

> Dicas de VS Code úteis ao agente: **Tasks** orquestram CLIs de dentro do editor; **launch** configura depuração; multi‑root e providers também suportam tasks/workspaces avançados.

---

## 5) SOP — Procedimento padrão do agente
1. **Validar estrutura**: se `dados/`, `cfg/`, `out/`, `src/` não existirem, criar; verificar CSVs exigidos para o período.
2. **Checar configs**: abrir `cfg/profiles_map.yml`, `cfg/matching_pesos.yml`, `cfg/regex_tokens.yml`, `cfg/cfop_expectativas.csv`, `cfg/issues_rules.yml` e **comentar no PR** qualquer incoerência.
3. **Rodar pipeline** via Tasks: `01: loader` → `02: normalizer` → `03: matcher` → `04: issues` → `05: dataset UI`.
4. **Subir UI**: `UI server (uvicorn)` e abrir `src/ui_app.html` no navegador. Conferir contagens (KPIs) e cores por status.
5. **Gerar relatórios**: `Relatório XLSX` e/ou `Relatório PDF`.
6. **Auditar logs** (`out/match/match_log.jsonl`, `out/match/issues.jsonl`): verificar *scores*, *estratégias* e *motivos* por amostra.
7. **Documentar alterações** em CHANGELOG/PR: tolerâncias alteradas, novas regras, ajustes de profiles.

---

## 6) Prompts e instruções para o agente (prontos para colar)
**Sistema (estável):**
> Você é um *Repo Agent* operando no VS Code. Aja com **determinismo e auditabilidade**. Use **VS Code Tasks** para orquestrar scripts. Não invente dados nem paths. Se faltar arquivo, gere o esqueleto mínimo. Emita saídas no diretório `out/`. Quando editar configurações, explique o motivo e preserve comentários. Nunca apague dados do cliente. Priorize *auto‑match* ≥ 70 e reporte métricas.

**Assistente → Ações típicas:**
- “Crie/atualize `.vscode/tasks.json` com as tarefas do pipeline e valide no terminal integrado.”
- “Ajuste `cfg/matching_pesos.yml` para aumentar o peso de S1 e penalisar CFOP×conta incoerente.”
- “Rode as tasks 01→05 e gere `out/relatorio_conferencia.xlsx`.”
- “Leia 30 linhas de `out/match/reconc_grid.csv` e explique 5 divergências com base em `issues_rules.yml`.”

**Verificações automáticas antes de *commit*:**
- `difflint`: alertar quando `issues_rules.yml` introduzir regra sem `mark_status`.
- `schema-check`: carregar `ui_schema.json` e validar colunas obrigatórias (score, status, S.doc, F.doc, delta.valor, delta.dias).

---

## 7) Integração com modelos OpenAI (opcional)
- **Chave** via env (`OPENAI_API_KEY`) para chamadas a modelos (ex.: gerar *regex* ou refatorar código).
- **Mensagens** estilo Chat Completions (role `system|user|assistant`) e *function calling* para automatizar passos.
- **Agentes/Responses**: usar *ferramentas* (pesquisa/arquivos/código) e *tracing*, quando aplicável no SDK atual.

> Observações operacionais: migrações de *Completions* para **Chat Completions** e uso de plataformas de **Agentes**/“Responses API” são recomendados nas versões atuais do SDK.

---

## 8) Segurança e limites operacionais
- Rodar Tasks exige **confiar** na pasta (Workspace Trust). Evitar executar tasks de repositórios desconhecidos.
- Nunca fazer chamadas externas com dados sensíveis dos CSVs sem aprovação explícita.
- Operar sempre no escopo **um cliente × um período**; para outro cliente/período, duplicar pasta `dados/` e `cfg/`.

---

## 9) Dev Containers (opcional para ambiente reprodutível)
Crie `.devcontainer/devcontainer.json` com Python 3.12, extensões e dependências. O agente pode propor a criação desse arquivo para padronizar toolchain e extensões no VS Code.

---

## 10) Checklists rápidas
**Antes de rodar o matcher:**
- [ ] `sucessor.csv` tem colunas de `data`, `debito`, `credito`, `valor`, `doc`, `historico` mapeadas por profile.
- [ ] Entradas/Saídas/Serviços têm `data`, `doc`, `participante`, `valor`, `cfop`, `situacao` mapeados.
- [ ] `fornecedores.csv` e `plano_contas.csv` carregam sem erro.
- [ ] Tolerâncias padrão: `max(0,01; 0,2%)` valor; janela datas ±3 dias (ajustável).

**Antes de exportar:**
- [ ] `out/ui_grid.jsonl` existe e `/api/schema` responde.
- [ ] KPIs (OK/Alertas/Divergências) batem com amostras visuais.
- [ ] `issues.jsonl` tem códigos e mensagens consistentes.

---

## 11) Estruturas de arquivos úteis (snippet)
**`cfg/matching_pesos.yml` (exemplo resumido):**
```yaml
version: 1
limiar_auto: 70
pesos:
  S1: 50
  S2: 35
  S3: 25
  S4: 20
  S5: 10
penalidades:
  participante_divergente: 15
  cfop_contas_incoerente: 10
tolerancias:
  valor_abs: 0.01
  valor_pct: 0.002
  janela_dias: 3
```

**`cfg/ui_schema.json` (campos mínimos):**
```json
{
  "columns": [
    {"id":"strategy","label":"Regra"},
    {"id":"score","type":"number"},
    {"id":"status"},
    {"id":"S.doc"}, {"id":"S.valor","type":"money"}, {"id":"S.data"},
    {"id":"F.doc"}, {"id":"F.valor","type":"money"}, {"id":"F.cfop"},
    {"id":"delta.valor","type":"money"}, {"id":"delta.dias","type":"number"},
    {"id":"motivos"}
  ]
}
```

---

## 12) Erros comuns e como o agente corrige
- **CSV com separador errado** → ajustar `delimiter` e `decimal` no `profiles_map.yml` e reprocessar `01→05`.
- **Baixo match** → relaxar janelas/pesos, habilitar S5 com regex de histórico e revisar `doc_num_regex`.
- **UI vazia** → conferir `DATA_DIR=out` no servidor e existência de `out/ui_grid.jsonl`.
- **Export PDF falha** → usar fallback HTML (gerado pelo script).

---

## 13) Glossário mínimo para o agente
- **Task**: comando automatizado configurado em `tasks.json` no VS Code.
- **Launch**: configuração de depuração em `launch.json`.
- **Blocking key**: chave composta para limitar candidatos no matching.
- **Auto‑match**: casamento com score ≥ limiar (default 70).

---

## 14) Fim
O agente que seguir este manual conseguirá **rodar o pipeline ponta‑a‑ponta**, ajustar regras e produzir relatórios auditáveis, sem guesswork e sem “mágica”.


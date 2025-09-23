
-- =============================================================
--  SCHEMA (SQLite/DuckDB): Conferência Sucessor × Fontes
--  Versão: 1.1
-- =============================================================
PRAGMA foreign_keys = ON;

-- ==============
-- 1) STAGING
-- ==============
CREATE TABLE IF NOT EXISTS stg_sucessor_lanc (
  id INTEGER PRIMARY KEY,
  chave               TEXT,
  transacao           TEXT,
  data_br             TEXT,
  data_iso            TEXT,
  debito_cod          TEXT,
  debito_alias        TEXT,
  participante_d_cod  TEXT,
  credito_cod         TEXT,
  credito_alias       TEXT,
  participante_c_cod  TEXT,
  valor_raw           TEXT,
  valor_num           REAL,
  hist_cod            TEXT,
  numero_docto_raw    TEXT,
  complemento_raw     TEXT,
  status_transacao    TEXT,
  raw_json            TEXT
);
CREATE INDEX IF NOT EXISTS idx_stg_suc_doc  ON stg_sucessor_lanc(numero_docto_raw);
CREATE INDEX IF NOT EXISTS idx_stg_suc_data ON stg_sucessor_lanc(data_iso);
CREATE INDEX IF NOT EXISTS idx_stg_suc_val  ON stg_sucessor_lanc(valor_num);

CREATE TABLE IF NOT EXISTS stg_suprema_ent (
  id INTEGER PRIMARY KEY,
  chave               TEXT,
  fornecedor_cod      TEXT,
  fornecedor_nome     TEXT,
  data_entrada_br     TEXT,
  data_emissao_br     TEXT,
  data_entrada_iso    TEXT,
  data_emissao_iso    TEXT,
  documento_raw       TEXT,
  cfop                TEXT,
  modelo              TEXT,
  especie             TEXT,
  condicao_pagto      TEXT,
  valor_contabil_raw  TEXT,
  valor_contabil_num  REAL,
  debito_alias        TEXT,
  credito_alias       TEXT,
  desconto_num        REAL,
  chave_xml           TEXT,
  usuario_inc         TEXT,
  data_inc_iso        TEXT,
  tipo_inc            TEXT,
  usuario_alt         TEXT,
  data_alt_iso        TEXT,
  situacao            TEXT,
  raw_json            TEXT
);
CREATE INDEX IF NOT EXISTS idx_stg_ent_doc  ON stg_suprema_ent(documento_raw);
CREATE INDEX IF NOT EXISTS idx_stg_ent_data ON stg_suprema_ent(data_emissao_iso);
CREATE INDEX IF NOT EXISTS idx_stg_ent_val  ON stg_suprema_ent(valor_contabil_num);

CREATE TABLE IF NOT EXISTS stg_suprema_sai (
  id INTEGER PRIMARY KEY,
  chave               TEXT,
  cliente_cod         TEXT,
  cliente_nome        TEXT,
  data_entrada_br     TEXT,
  data_emissao_br     TEXT,
  data_entrada_iso    TEXT,
  data_emissao_iso    TEXT,
  documento_raw       TEXT,
  cfop                TEXT,
  modelo              TEXT,
  especie             TEXT,
  condicao_pagto      TEXT,
  valor_contabil_raw  TEXT,
  valor_contabil_num  REAL,
  debito_alias        TEXT,
  credito_alias       TEXT,
  desconto_num        REAL,
  chave_xml           TEXT,
  usuario_inc         TEXT,
  data_inc_iso        TEXT,
  tipo_inc            TEXT,
  usuario_alt         TEXT,
  data_alt_iso        TEXT,
  situacao            TEXT,
  raw_json            TEXT
);
CREATE INDEX IF NOT EXISTS idx_stg_sai_doc  ON stg_suprema_sai(documento_raw);
CREATE INDEX IF NOT EXISTS idx_stg_sai_data ON stg_suprema_sai(data_emissao_iso);
CREATE INDEX IF NOT EXISTS idx_stg_sai_val  ON stg_suprema_sai(valor_contabil_num);

CREATE TABLE IF NOT EXISTS stg_suprema_srv (
  id INTEGER PRIMARY KEY,
  chave               TEXT,
  participante_cod    TEXT,
  participante_nome   TEXT,
  data_entrada_iso    TEXT,
  data_emissao_iso    TEXT,
  documento_raw       TEXT,
  cfop                TEXT,
  modelo              TEXT,
  especie             TEXT,
  condicao_pagto      TEXT,
  valor_contabil_num  REAL,
  debito_alias        TEXT,
  credito_alias       TEXT,
  desconto_num        REAL,
  situacao            TEXT,
  raw_json            TEXT
);

CREATE TABLE IF NOT EXISTS stg_fornecedores (
  id INTEGER PRIMARY KEY,
  codigo   TEXT,
  nome     TEXT,
  cnpj     TEXT,
  tipo     TEXT,
  raw_json TEXT
);

CREATE TABLE IF NOT EXISTS stg_plano_contas (
  id INTEGER PRIMARY KEY,
  codigo   TEXT,
  alias    TEXT,
  nome     TEXT,
  natureza TEXT,
  raw_json TEXT
);

CREATE TABLE IF NOT EXISTS stg_legacy_bank (
  id INTEGER PRIMARY KEY,
  data_iso      TEXT,
  valor_num     REAL,
  debito_cod    TEXT,
  credito_cod   TEXT,
  historico_raw TEXT,
  docto_raw     TEXT,
  participante  TEXT,
  raw_json      TEXT
);

-- ==============
-- 2) DIMENSÕES
-- ==============
CREATE TABLE IF NOT EXISTS dim_participante (
  part_cod TEXT PRIMARY KEY,
  nome     TEXT,
  cnpj     TEXT,
  tipo     TEXT
);
CREATE TABLE IF NOT EXISTS dim_conta (
  conta_cod TEXT PRIMARY KEY,
  alias     TEXT,
  nome      TEXT,
  natureza  TEXT
);
CREATE TABLE IF NOT EXISTS dim_cfop (
  cfop      TEXT PRIMARY KEY,
  direcao   TEXT,
  classe    TEXT,
  flags     TEXT
);

-- ==============
-- 3) FATO
-- ==============
CREATE TABLE IF NOT EXISTS f_sucessor_lanc (
  id INTEGER PRIMARY KEY,
  chave               TEXT,
  transacao           TEXT,
  data_iso            TEXT,
  mes_ref             TEXT,
  debito_cod          TEXT,
  debito_alias        TEXT,
  participante_d_cod  TEXT,
  credito_cod         TEXT,
  credito_alias       TEXT,
  participante_c_cod  TEXT,
  valor_num           REAL,
  hist_cod            TEXT,
  doc_num_norm        TEXT,
  doc_serie_norm      TEXT,
  complemento_norm    TEXT,
  status_transacao    TEXT
);
CREATE INDEX IF NOT EXISTS idx_f_suc_doc ON f_sucessor_lanc(doc_num_norm);
CREATE INDEX IF NOT EXISTS idx_f_suc_mes ON f_sucessor_lanc(mes_ref);
CREATE INDEX IF NOT EXISTS idx_f_suc_val ON f_sucessor_lanc(valor_num);

CREATE TABLE IF NOT EXISTS f_fiscal_doc (
  id INTEGER PRIMARY KEY,
  tipo_fonte      TEXT,      -- ENTRADA/SAIDA/SERVICO
  participante_cod TEXT,
  participante_nome TEXT,
  data_doc_iso    TEXT,
  mes_ref         TEXT,
  doc_num_norm    TEXT,
  doc_serie_norm  TEXT,
  cfop            TEXT,
  valor_num       REAL,
  situacao        TEXT,
  chave_xml       TEXT
);
CREATE INDEX IF NOT EXISTS idx_f_fiscal_doc ON f_fiscal_doc(doc_num_norm);
CREATE INDEX IF NOT EXISTS idx_f_fiscal_mes ON f_fiscal_doc(mes_ref);
CREATE INDEX IF NOT EXISTS idx_f_fiscal_val ON f_fiscal_doc(valor_num);

-- ==============
-- 4) MATCHES/ISSUES
-- ==============
CREATE TABLE IF NOT EXISTS match_candidates (
  id INTEGER PRIMARY KEY,
  suc_id      INTEGER NOT NULL,
  fonte_id    INTEGER NOT NULL,
  tipo_fonte  TEXT NOT NULL,
  estrategia  TEXT NOT NULL,
  score       REAL NOT NULL,
  delta_valor REAL,
  delta_dias  INTEGER,
  motivos     TEXT,
  regra_id    TEXT,
  created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_mc_suc   ON match_candidates(suc_id);
CREATE INDEX IF NOT EXISTS idx_mc_fonte ON match_candidates(fonte_id);
CREATE INDEX IF NOT EXISTS idx_mc_score ON match_candidates(score);

CREATE TABLE IF NOT EXISTS matches (
  id INTEGER PRIMARY KEY,
  suc_id      INTEGER NOT NULL,
  fonte_id    INTEGER,
  tipo_fonte  TEXT,
  status      TEXT NOT NULL,   -- OK/ALERTA/DIVERGENCIA/SEM_FONTE/SEM_SUCESSOR
  estrategia  TEXT,
  score       REAL,
  diffs_json  TEXT,
  regra_id    TEXT,
  decided_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_m_suc    ON matches(suc_id);
CREATE INDEX IF NOT EXISTS idx_m_status ON matches(status);

CREATE TABLE IF NOT EXISTS issues (
  id INTEGER PRIMARY KEY,
  tipo       TEXT,
  severidade TEXT,
  contexto   TEXT,
  mensagem   TEXT,
  sugestao   TEXT,
  resolved   INTEGER DEFAULT 0,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- ==============
-- 5) CONFIG
-- ==============
CREATE TABLE IF NOT EXISTS config_profiles (
  id INTEGER PRIMARY KEY,
  nome        TEXT UNIQUE,
  schema_json TEXT,
  parse_rules TEXT,
  ativo       INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS config_tolerancias (
  id INTEGER PRIMARY KEY,
  contexto TEXT,
  chave    TEXT,
  tol_abs  REAL,
  tol_pct  REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_tol_ctx ON config_tolerancias(contexto, chave);

-- ==============
-- 6) VIEWS
-- ==============
CREATE VIEW IF NOT EXISTS v_pending_matches AS
SELECT s.id AS suc_id, s.doc_num_norm, s.valor_num, s.data_iso, m.status
FROM f_sucessor_lanc s
LEFT JOIN matches m ON m.suc_id = s.id
WHERE m.id IS NULL OR m.status IN ('ALERTA','DIVERGENCIA','SEM_FONTE');

CREATE VIEW IF NOT EXISTS v_divergencias AS
SELECT m.id, m.suc_id, m.fonte_id, m.status, m.diffs_json, m.score, m.regra_id
FROM matches m
WHERE m.status = 'DIVERGENCIA';

-- seed tolerância global
INSERT INTO config_tolerancias (contexto, chave, tol_abs, tol_pct)
SELECT 'GLOBAL','GLOBAL',0.01,0.002
WHERE NOT EXISTS (SELECT 1 FROM config_tolerancias WHERE contexto='GLOBAL' AND chave='GLOBAL');

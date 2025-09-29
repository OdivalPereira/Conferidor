# issues_engine.py — 27/28
# Aplica regras DSL (issues_rules.yml) sobre a grid consolidada (CSV/JSONL),
# gera issues.jsonl e uma grid anotada (status ajustado + colunas extras).
#
# Uso:
#   python issues_engine.py \
#     --grid reconc_grid.csv \
#     --rules issues_rules.yml \
#     --out-issues issues.jsonl \
#     --out-grid reconc_grid_issues.csv
#
# Suporta grid em CSV (.csv) ou JSON Lines (.jsonl). Regras em YAML (.yml/.yaml) ou JSON (.json).
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ====== Loader de regras (YAML/JSON) ======
def load_rules(path: str | os.PathLike[str]) -> Dict[str, Any]:
    path_str = os.fspath(path)
    ext = os.path.splitext(path_str)[1].lower()
    if ext in (".json",):
        with open(path_str, "r", encoding="utf-8") as f:
            return json.load(f)
    if ext in (".yml", ".yaml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("PyYAML não está instalado. Instale com: pip install pyyaml") from e
        with open(path_str, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    raise RuntimeError(f"Extensão de regras não suportada: {ext}")


_SUPPORTED_OPERATORS = {
    "exists",
    "eq",
    "ne",
    "lt",
    "lte",
    "gt",
    "gte",
    "abs_gt",
    "abs_gte",
    "in",
    "not_in",
    "regex",
    "startswith",
    "endswith",
}

def _listify(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
        return out
    return []


def _validate_condition_list(conditions: Any, *, prefix: str, errors: List[str]) -> None:
    if conditions is None:
        return
    if not isinstance(conditions, list):
        errors.append(f"{prefix} deve ser uma lista de condições.")
        return
    for idx, cond in enumerate(conditions):
        if not isinstance(cond, dict):
            errors.append(f"{prefix}[{idx}] deve ser um objeto com 'field' e 'op'.")
            continue
        field = cond.get("field")
        op = cond.get("op")
        if not field or not isinstance(field, str):
            errors.append(f"{prefix}[{idx}].field deve ser uma string não vazia.")
        if not op or not isinstance(op, str):
            errors.append(f"{prefix}[{idx}].op deve ser uma string não vazia.")
        elif op not in _SUPPORTED_OPERATORS:
                errors.append(f"{prefix}[{idx}].op='{op}' não é suportado pelo issues_engine.")


def _validate_condition_block(
    block: Any,
    *,
    prefix: str,
    errors: List[str],
    condition_defs: Optional[Dict[str, Any]] = None,
) -> None:
    if block is None:
        return
    if not isinstance(block, dict):
        errors.append(f"{prefix} deve ser um objeto (dict).")
        return

    use_val = block.get("use")
    if use_val is not None and not isinstance(use_val, (list, tuple, str)):
        errors.append(f"{prefix}.use deve ser string ou lista de strings.")
    refs = _listify(use_val)
    if refs and condition_defs is None:
        errors.append(f"{prefix}.use informado mas nenhuma definitions.conditions foi declarada.")
    elif condition_defs is not None:
        for ref in refs:
            if ref not in condition_defs:
                errors.append(f"{prefix}.use faz referência desconhecida a '{ref}'.")

    _validate_condition_list(block.get("all"), prefix=f"{prefix}.all", errors=errors)
    _validate_condition_list(block.get("any"), prefix=f"{prefix}.any", errors=errors)


def _validate_emit_block(
    emit: Any,
    *,
    prefix: str,
    errors: List[str],
    action_defs: Optional[Dict[str, Any]] = None,
) -> None:
    if not isinstance(emit, dict):
        errors.append(f"{prefix} deve ser um objeto (dict).")
        return

    use_val = emit.get("use")
    if use_val is not None and not isinstance(use_val, (list, tuple, str)):
        errors.append(f"{prefix}.use deve ser string ou lista de strings.")
    refs = _listify(use_val)
    if refs and action_defs is None:
        errors.append(f"{prefix}.use informado mas nenhuma definitions.actions foi declarada.")
    elif action_defs is not None:
        for ref in refs:
            if ref not in action_defs:
                errors.append(f"{prefix}.use faz referência desconhecida a '{ref}'.")

    message = emit.get("message")
    if message is not None and not isinstance(message, str):
        errors.append(f"{prefix}.message deve ser string se fornecida.")
    mark_status = emit.get("mark_status")
    if mark_status is not None and not isinstance(mark_status, str):
        errors.append(f"{prefix}.mark_status deve ser string se fornecida.")
    code = emit.get("code")
    if code is not None and not isinstance(code, str):
        errors.append(f"{prefix}.code deve ser string se fornecida.")
    severity = emit.get("severity")
    if severity is not None and not isinstance(severity, str):
        errors.append(f"{prefix}.severity deve ser string se fornecida.")


def validate_rules_document(doc: Any, *, source: Optional[str | os.PathLike[str]] = None) -> None:
    errors: List[str] = []
    label = f" ({os.fspath(source)})" if source is not None else ""

    if not isinstance(doc, dict):
        raise ValueError(f"Documento de regras inválido{label}: conteúdo deve ser um objeto mapeável.")

    defaults = doc.get("defaults")
    if defaults is not None and not isinstance(defaults, dict):
        errors.append("defaults deve ser um objeto mapeável (dict).")

    definitions = doc.get("definitions") or {}
    if definitions and not isinstance(definitions, dict):
        errors.append("definitions deve ser um objeto mapeável (dict).")

    cond_defs = definitions.get("conditions") if isinstance(definitions, dict) else None
    if cond_defs is not None and not isinstance(cond_defs, dict):
        errors.append("definitions.conditions deve ser um objeto (dict).")
    elif isinstance(cond_defs, dict):
        for name, block in cond_defs.items():
            _validate_condition_block(
                block,
                prefix=f"definitions.conditions['{name}']",
                errors=errors,
                condition_defs=cond_defs,
            )

    action_defs = definitions.get("actions") if isinstance(definitions, dict) else None
    if action_defs is not None and not isinstance(action_defs, dict):
        errors.append("definitions.actions deve ser um objeto (dict).")
    elif isinstance(action_defs, dict):
        for name, block in action_defs.items():
            if not isinstance(block, dict):
                errors.append(f"definitions.actions['{name}'] deve ser um objeto (dict).")
                continue
            _validate_emit_block(
                block,
                prefix=f"definitions.actions['{name}']",
                errors=errors,
                action_defs=action_defs,
            )

    rules = doc.get("rules")
    if not isinstance(rules, list) or not rules:
        errors.append("A chave 'rules' deve conter uma lista com pelo menos uma regra.")
    else:
        for idx, rule in enumerate(rules):
            if not isinstance(rule, dict):
                errors.append(f"rules[{idx}] deve ser um objeto (dict).")
                continue

            rule_id = rule.get("id")
            if not rule_id or not isinstance(rule_id, str):
                errors.append(f"rules[{idx}].id deve ser uma string não vazia.")

            emit = rule.get("emit")
            _validate_emit_block(
                emit,
                prefix=f"rules[{idx}].emit",
                errors=errors,
                action_defs=action_defs if isinstance(action_defs, dict) else None,
            )

            for section_name in ("when", "and", "but"):
                section = rule.get(section_name)
                _validate_condition_block(
                    section,
                    prefix=f"rules[{idx}].{section_name}",
                    errors=errors,
                    condition_defs=cond_defs if isinstance(cond_defs, dict) else None,
                )

    if errors:
        raise ValueError(f"Regras inválidas{label}: " + "; ".join(errors))


def validate_rules_file(path: str | os.PathLike[str]) -> Dict[str, Any]:
    doc = load_rules(path)
    validate_rules_document(doc, source=path)
    return doc


def _merge_condition_blocks(*blocks: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    result_all: List[Dict[str, Any]] = []
    result_any: List[Dict[str, Any]] = []
    for block in blocks:
        for cond in (block.get("all") or []):
            result_all.append(dict(cond))
        for cond in (block.get("any") or []):
            result_any.append(dict(cond))
    merged: Dict[str, List[Dict[str, Any]]] = {}
    if result_all:
        merged["all"] = result_all
    if result_any:
        merged["any"] = result_any
    return merged


def _prepare_condition_resolver(condition_defs: Dict[str, Any]):
    cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    def expand(name: str, stack: Tuple[str, ...]) -> Dict[str, List[Dict[str, Any]]]:
        if name in cache:
            return cache[name]
        if name in stack:
            cycle = " -> ".join(stack + (name,))
            raise ValueError(f"Referência cíclica em definitions.conditions: {cycle}")
        block = condition_defs.get(name)
        if block is None:
            raise ValueError(f"Condição referenciada não encontrada: {name}")
        merged = resolve_with_stack(block, stack + (name,))
        cache[name] = merged
        return merged

    def resolve_with_stack(block: Dict[str, Any], stack: Tuple[str, ...]) -> Dict[str, List[Dict[str, Any]]]:
        merged_blocks: List[Dict[str, List[Dict[str, Any]]]] = []
        for ref in _listify(block.get("use")):
            merged_blocks.append(expand(ref, stack))
        direct_block: Dict[str, List[Dict[str, Any]]] = {}
        if block.get("all"):
            direct_block["all"] = [dict(c) for c in block.get("all", [])]
        if block.get("any"):
            direct_block["any"] = [dict(c) for c in block.get("any", [])]
        merged_blocks.append(direct_block)
        return _merge_condition_blocks(*merged_blocks)

    def resolve_block(block: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        return resolve_with_stack(block, tuple())

    return resolve_block


def _prepare_action_resolver(action_defs: Dict[str, Dict[str, Any]]):
    cache: Dict[str, Dict[str, Any]] = {}

    def expand(name: str, stack: Tuple[str, ...]) -> Dict[str, Any]:
        if name in cache:
            return cache[name]
        if name in stack:
            cycle = " -> ".join(stack + (name,))
            raise ValueError(f"Referência cíclica em definitions.actions: {cycle}")
        block = action_defs.get(name)
        if block is None:
            raise ValueError(f"Ação referenciada não encontrada: {name}")
        resolved = resolve_with_stack(block, stack + (name,))
        cache[name] = resolved
        return resolved

    def resolve_with_stack(block: Dict[str, Any], stack: Tuple[str, ...]) -> Dict[str, Any]:
        resolved: Dict[str, Any] = {}
        for ref in _listify(block.get("use")):
            resolved.update(expand(ref, stack))
        for key, value in block.items():
            if key == "use":
                continue
            resolved[key] = value
        return dict(resolved)

    def resolve_block(block: Dict[str, Any]) -> Dict[str, Any]:
        return resolve_with_stack(block, tuple())

    return resolve_block


def prepare_rules_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    definitions = doc.get("definitions") or {}
    condition_defs: Dict[str, Any] = definitions.get("conditions") or {}
    action_defs: Dict[str, Dict[str, Any]] = definitions.get("actions") or {}

    condition_resolver = _prepare_condition_resolver(condition_defs)
    action_resolver = _prepare_action_resolver(action_defs)

    prepared_rules: List[Dict[str, Any]] = []
    for rule in doc.get("rules", []):
        new_rule = dict(rule)
        for section_name in ("when", "and", "but"):
            block = rule.get(section_name)
            if not block:
                new_rule.pop(section_name, None)
                continue
            if condition_defs:
                resolved = condition_resolver(block)
            else:
                resolved = _merge_condition_blocks(
                    {"all": block.get("all", []), "any": block.get("any", [])}
                )
            if resolved:
                new_rule[section_name] = resolved
            else:
                new_rule.pop(section_name, None)

        emit_block = rule.get("emit") or {}
        if action_defs:
            resolved_emit = action_resolver(emit_block)
        else:
            resolved_emit = dict(emit_block)
        resolved_emit.pop("use", None)
        new_rule["emit"] = resolved_emit

        prepared_rules.append(new_rule)

    doc = dict(doc)
    doc["rules"] = prepared_rules
    return doc

# ====== Loader da grid (CSV/JSONL) ======
def read_grid(path: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows
    elif ext == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    else:
        raise RuntimeError(f"Formato de grid não suportado: {ext}")

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    # order of columns: existing + appended
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})

# ====== Util: acesso a campos com "dot path" ======
def get_field(row: Dict[str, Any], path: str) -> Any:
    # suporta "F.doc", "S.valor" etc.
    cur: Any = row
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return row.get(path) if path in row else None
    return cur

def str_or_none(x: Any) -> Optional[str]:
    if x is None: return None
    return str(x)

def to_float(x: Any) -> Optional[float]:
    if x is None: return None
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip().replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def is_empty(x: Any) -> bool:
    return x is None or (isinstance(x, str) and x == "")

# ====== Operadores ======
_REGEX_CACHE: Dict[str, re.Pattern] = {}

def _rx(pat: str) -> re.Pattern:
    k = ("(?i)" + pat)
    if k not in _REGEX_CACHE:
        _REGEX_CACHE[k] = re.compile(pat, re.IGNORECASE)
    return _REGEX_CACHE[k]

def op_eval(field_val: Any, op: str, value: Any) -> bool:
    if op == "exists":
        want = bool(value)
        return (not is_empty(field_val)) if want else is_empty(field_val)
    if op in ("eq", "ne"):
        a, b = field_val, value
        res = (a == b)
        return res if op == "eq" else (not res)
    if op in ("lt", "lte", "gt", "gte", "abs_gt", "abs_gte"):
        fv = to_float(field_val); tv = to_float(value)
        if fv is None or tv is None: return False
        if op == "lt": return fv < tv
        if op == "lte": return fv <= tv
        if op == "gt": return fv > tv
        if op == "gte": return fv >= tv
        if op == "abs_gt": return abs(fv) > tv
        if op == "abs_gte": return abs(fv) >= tv
    if op == "in":
        if isinstance(value, (list, tuple, set)):
            return field_val in value
        return False
    if op == "not_in":
        if isinstance(value, (list, tuple, set)):
            return field_val not in value
        return False
    if op == "regex":
        if field_val is None: return False
        return bool(_rx(str(value)).search(str(field_val)))
    if op == "startswith":
        if field_val is None: return False
        return str(field_val).lower().startswith(str(value).lower())
    if op == "endswith":
        if field_val is None: return False
        return str(field_val).lower().endswith(str(value).lower())
    # op desconhecido = False
    return False

@dataclass
class Condition:
    field: str
    op: str
    value: Any

def eval_group(row: Dict[str, Any], group: Dict[str, Any]) -> bool:
    # group = {"all": [conds], "any": [conds]}
    if not group: return True
    if "all" in group:
        for c in group["all"] or []:
            if not op_eval(get_field(row, c.get("field")), c.get("op"), c.get("value")):
                return False
    if "any" in group:
        any_conds = group["any"] or []
        if any_conds:
            if not any(op_eval(get_field(row, c.get("field")), c.get("op"), c.get("value")) for c in any_conds):
                return False
    return True

def eval_rule(row: Dict[str, Any], rule: Dict[str, Any]) -> bool:
    # suporta chaves: when, and, but — todas precisam passar (AND)
    for key in ("when", "and", "but"):
        grp = rule.get(key)
        if grp is not None and not eval_group(row, grp):
            return False
    return True

# ====== Mensagens com placeholders {campo} ou {A or B} ======
_FIELD_EXPR = re.compile(r"\{([^}]+)\}")

def format_msg(template: str, row: Dict[str, Any]) -> str:
    def repl(m):
        expr = m.group(1).strip()
        # suporta "F.doc or S.doc"
        if " or " in expr:
            for alt in [s.strip() for s in expr.split(" or ")]:
                v = get_field(row, alt)
                if not is_empty(v): return str(v)
            return ""
        v = get_field(row, expr)
        return "" if is_empty(v) else str(v)
    return _FIELD_EXPR.sub(repl, template or "")

# ====== Status precedence ======
_STATUS_ORDER = {"DIVERGENCIA": 3, "ALERTA": 2, "OK": 1, "SEM_FONTE": 0, "SEM_SUCESSOR": 0, None: -1}

def merge_status(curr: Optional[str], new: Optional[str]) -> Optional[str]:
    if new is None: return curr
    cc = _STATUS_ORDER.get((curr or "").upper(), -1)
    nn = _STATUS_ORDER.get((new or "").upper(), -1)
    return new if nn > cc else curr

# ====== Motor principal ======
def run_issues(grid_path: str, rules_path: str, out_issues: str, out_grid: str) -> Dict[str, Any]:
    rules_doc = validate_rules_file(rules_path)
    rules_doc = prepare_rules_document(rules_doc)
    rules = rules_doc.get("rules") or []
    defaults = rules_doc.get("defaults") or {}
    grid = read_grid(grid_path)

    issues_out: List[Dict[str, Any]] = []
    grid_out: List[Dict[str, Any]] = []

    for idx, row in enumerate(grid):
        # determina o campo de status
        status_field = "status" if "status" in row else ("match.status" if "match.status" in row else "status")
        curr_status = str(row.get(status_field) or "").upper() or None

        new_status = curr_status
        hit_codes: List[str] = []
        hit_msgs: List[str] = []

        for rule in rules:
            if eval_rule(row, rule):
                emit = rule.get("emit") or {}
                code = emit.get("code") or rule.get("id")
                msg = format_msg(emit.get("message") or "", row)
                # sugestão opcional
                sug = emit.get("suggest")
                sev = rule.get("severity")
                hit = {"row_id": row.get("id", idx), "rule_id": rule.get("id"), "code": code, "severity": sev, "message": msg, "suggest": sug}
                issues_out.append(hit)
                hit_codes.append(str(code))
                if msg: hit_msgs.append(str(msg))
                # marca status se definido
                new_status = merge_status(new_status, emit.get("mark_status"))

        # anotações no row
        row2 = dict(row)
        row2["status_prev"] = curr_status
        if new_status and new_status != curr_status:
            row2[status_field] = new_status
        row2["issues"] = ";".join(hit_codes)
        row2["issues_msgs"] = " | ".join(hit_msgs)
        grid_out.append(row2)

    # escreve saídas
    write_jsonl(out_issues, issues_out)
    # escolha formato do out_grid pelo sufixo
    ext = os.path.splitext(out_grid)[1].lower()
    if ext == ".jsonl":
        write_jsonl(out_grid, grid_out)
    else:
        write_csv(out_grid, grid_out)

    return {"ok": True, "grid_in": grid_path, "rules": rules_path, "out_issues": out_issues, "out_grid": out_grid, "rules_count": len(rules), "rows": len(grid)}

def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Aplica issues_rules sobre a grid e gera issues + grid anotada")
    ap.add_argument("--grid", required=True, help="Caminho da grid (CSV ou JSONL)")
    ap.add_argument("--rules", required=True, help="Caminho das regras (YAML ou JSON)")
    ap.add_argument("--out-issues", default="issues.jsonl", help="Saída issues.jsonl")
    ap.add_argument("--out-grid", default="reconc_grid_issues.csv", help="Saída da grid anotada (.csv ou .jsonl)")
    args = ap.parse_args(argv)

    res = run_issues(args.grid, args.rules, args.out_issues, args.out_grid)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

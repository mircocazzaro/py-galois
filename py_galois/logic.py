
from typing import Dict, List, Tuple, Any, Optional
import re, time, json, math
from .scan import TableScan, KeyScan, classify_atoms, ask_confidence, ScanResult
from .llm_client import BaseLLM
from .sql_utils import extract_where_atoms, exec_sqlite_query, strip_schema
from .sql_utils import referenced_tables

class LogicalPlan:
    """A simple logical plan representation keeping per-table pushdowns and physical choice."""
    def __init__(self, per_table: Dict[str, Dict[str, Any]]):
        self.per_table = per_table  # {table: { 'strategy': 'none'|'single'|'all', 'atoms': [...], 'physical': 'table'|'key', 'pushed_cond': str }}

def choose_pushdown_for_table(llm: BaseLLM, table: str, atoms: List[str]) -> Dict[str, Any]:
    """Apply the policy: if exactly one 'high' => push just that one; if >1 'high' => push all; if none => no push."""
    labels = classify_atoms(llm, table, atoms)
    print("LABELS:", labels)
    highs = [a for a, lab in labels.items() if lab == 'high']
    if len(highs) == 0:
        return {'strategy': 'none', 'atoms': [], 'pushed_cond': None}
    elif len(highs) == 1:
        return {'strategy': 'single', 'atoms': highs, 'pushed_cond': highs[0]}
    else:
        cond = ' AND '.join(highs)
        return {'strategy': 'all', 'atoms': highs, 'pushed_cond': cond}

def build_plans(llm: BaseLLM, sql: str, table_metas: Dict[str, Dict],
                select_attrs_by_table: Dict[str, List[str]],
                key_attrs_by_table: Dict[str, List[str]], tau: float) -> LogicalPlan:
    atoms_by_table = extract_where_atoms(sql)
    if len(table_metas) == 1 and "__unknown__" in atoms_by_table:
        only_table = next(iter(table_metas.keys()))
        atoms_by_table[only_table] = atoms_by_table.get(only_table, []) + atoms_by_table["__unknown__"]
    per_table = {}
    for tname, tmeta in table_metas.items():
        atoms = atoms_by_table.get(tname, [])
        push = choose_pushdown_for_table(llm, tname, atoms)
        # Physical choice via confidence rule
        key_attrs = key_attrs_by_table.get(tname, [])
        select_attrs = select_attrs_by_table.get(tname, [])
        conf = ask_confidence(llm, tname, key_attrs, push['atoms'], select_attrs)
        conf_q = conf ** max(1, len(select_attrs))
        physical = 'key' if conf_q >= tau else 'table'
        push['physical'] = physical
        push['conf_keys'] = conf      # LLM_conf(keys|conds)
        push['conf_q'] = conf_q       # conf(q) = conf^n
        per_table[tname] = push
    return LogicalPlan(per_table)


def execute_plan(llm: BaseLLM, plan: LogicalPlan, table_metas: Dict[str, Dict],
                 select_attrs_by_table: Dict[str, List[str]]
                 ) -> Tuple[Dict[str, List[Dict[str, Any]]], int, float, List[str]]:
    """Execute the plan: run the scans per table and return rows per table + tokens/time + log lines."""
    total_tokens = 0
    total_time = 0.0
    rows_by_table: Dict[str, List[Dict[str, Any]]] = {}
    logs: List[str] = []
    for tname, cfg in plan.per_table.items():
        meta = table_metas[tname]
        select_attrs = select_attrs_by_table[tname]
        if cfg['physical'] == 'key':
            key_attrs = [c for c in meta.get('keys', [])]
            nonkeys = [c for c in select_attrs if c not in key_attrs]
            scanner = KeyScan(llm, meta, key_attrs, nonkeys, pushed_cond=cfg['pushed_cond'])
        else:
            scanner = TableScan(llm, meta, select_attrs, pushed_cond=cfg['pushed_cond'])
        res = scanner.run()
        if isinstance(res, ScanResult):
            scan_res, chat_log = res, ""
        else:
            scan_res, chat_log = res  # unpack (ScanResult, log)
        
        rows_by_table[tname] = scan_res.rows
        total_tokens += scan_res.tokens
        total_time += scan_res.latency
        logs.append(
            f"{cfg['physical'].capitalize()}Scan[{tname}] "
            f"strategy={cfg['strategy']} pushed={cfg['pushed_cond']} "
            f"rows={len(scan_res.rows)} tokens={scan_res.tokens} time={scan_res.latency:.2f}s"
            f"fetched_rows={scan_res.rows}\n\n{chat_log}"
        )
        
        print(
            f"{cfg['physical'].capitalize()}Scan[{tname}] "
            f"strategy={cfg['strategy']} pushed={cfg['pushed_cond']} "
            f"rows={len(scan_res.rows)} tokens={scan_res.tokens} time={scan_res.latency:.2f}s"
            f"fetched_rows={scan_res.rows}\n\n{chat_log}"
        )
    return rows_by_table, total_tokens, total_time, logs


def final_answer_from_scans(rows_by_table: Dict[str, List[Dict[str, Any]]], sql: str):
    # Tables that the SQL query actually mentions in FROM/JOIN
    refs = referenced_tables(sql)
    # Only pass referenced tables into SQLite
    subset = {t: rows_by_table[t] for t in refs if t in rows_by_table}
    return exec_sqlite_query(subset, strip_schema(sql))

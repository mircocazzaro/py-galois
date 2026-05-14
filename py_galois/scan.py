
import json, math, time
from typing import Dict, List, Any, Optional, Tuple
from .llm_client import BaseLLM, LLMResponse
from .prompts import (
    json_schema_for_table, table_scan_first_prompt, table_scan_iter_prompt,
    key_scan_first_prompt, key_scan_iter_prompt, tuple_by_key_prompt,
    classify_where_atoms_prompt, confidence_prompt
)
from .json_utils import try_parse_json, ensure_list_of_dicts, normalize_row

def llm_messages(context: List[Tuple[str, str]], user_content: str) -> List[Dict[str, str]]:
    msgs = []
    for role, content in context:
        msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": user_content})
    return msgs

class ScanResult:
    def __init__(self, rows: List[Dict[str, Any]], tokens: int, latency: float):
        self.rows = rows
        self.tokens = tokens
        self.latency = latency

class TableScan:
    def __init__(self, llm: BaseLLM, table_meta: Dict, select_attrs: List[str], pushed_cond: Optional[str] = None, max_iter: int = 6):
        self.llm = llm
        self.table_meta = table_meta
        self.select_attrs = select_attrs
        self.pushed_cond = pushed_cond
        self.max_iter = max_iter

    def run(self) -> Tuple[ScanResult, str]:
        schema = json_schema_for_table(self.table_meta)
        tname = self.table_meta["name"]
        context: List[Tuple[str,str]] = []
        seen = set()
        rows: List[Dict[str, Any]] = []
        total_tokens = 0
        total_latency = 0.0

        # Make values safe to put inside a set key
        def make_hashable(v: Any) -> Any:
            if isinstance(v, (list, dict)):
                # stable string representation
                return json.dumps(v, sort_keys=True)
            return v

        # First prompt
        first = table_scan_first_prompt(
            sql_query="", table_name=tname, attrs=self.select_attrs, json_schema=schema, cond=self.pushed_cond
        )
        print("TableScan first prompt:", first)
        resp: LLMResponse = self.llm.chat(llm_messages(context, first))
        print("TableScan first response:", resp.text)
        chat_log = f"FIRST PROMPT:\n{first}\nRESPONSE:\n{resp.text}\n"
        total_tokens += resp.usage_tokens; total_latency += resp.latency_s
        data = try_parse_json(resp.text)
        for obj in ensure_list_of_dicts(data):
            row = normalize_row(obj, self.select_attrs)
            key = tuple(make_hashable(row.get(c)) for c in self.select_attrs)
            if key not in seen and len(key) > 0:
                seen.add(key); rows.append(row)
        # iterations
        it = 1
        while it < self.max_iter:
            nxt = table_scan_iter_prompt()
            print("TableScan iter prompt:", nxt)
            context.extend([("user", first), ("assistant", json.dumps(ensure_list_of_dicts(data)))])
            resp = self.llm.chat(llm_messages(context, nxt))
            print("TableScan iter response:", resp.text)
            chat_log += f"ITER {it} PROMPT:\n{nxt}\nRESPONSE:\n{resp.text}\n"
            total_tokens += resp.usage_tokens; total_latency += resp.latency_s
            data = try_parse_json(resp.text)
            new_rows = 0
            for obj in ensure_list_of_dicts(data):
                row = normalize_row(obj, self.select_attrs)
                key = tuple(make_hashable(row.get(c)) for c in self.select_attrs)
                if key not in seen:
                    seen.add(key); rows.append(row); new_rows += 1
            if new_rows == 0:
                break
            it += 1
        return ScanResult(rows, total_tokens, total_latency), chat_log

class KeyScan:
    def __init__(self, llm: BaseLLM, table_meta: Dict, key_attrs: List[str], nonkey_attrs: List[str], pushed_cond: Optional[str] = None, max_iter: int = 3):
        self.llm = llm
        self.table_meta = table_meta
        self.key_attrs = key_attrs
        self.nonkey_attrs = nonkey_attrs
        self.pushed_cond = pushed_cond
        self.max_iter = max_iter

    def run(self) -> ScanResult:
        schema_keys = {"type": "array", "items": {"type": "object", "properties": {k: {"type": ["string", "integer", "number"]} for k in self.key_attrs}}}
        schema_tuple = json_schema_for_table({"columns": [{"name": a, "dtype": "string"} for a in self.nonkey_attrs]})
        tname = self.table_meta["name"]
        context: List[Tuple[str,str]] = []
        keys_seen = set()
        keys: List[Dict[str, Any]] = []
        rows: List[Dict[str, Any]] = []
        total_tokens = 0
        total_latency = 0.0

        # Phase 1: collect keys iteratively
        first = key_scan_first_prompt(tname, self.key_attrs, schema_keys, self.pushed_cond)
        print("KeyScan first prompt:", first)
        resp = self.llm.chat(llm_messages(context, first))
        print("KeyScan first response:", resp.text)
        total_tokens += resp.usage_tokens; total_latency += resp.latency_s
        data = try_parse_json(resp.text)
        for obj in ensure_list_of_dicts(data):
            key = tuple(obj.get(k) for k in self.key_attrs)
            # scarta tool-calls e righe “vuote”
            if all(v in (None, "", []) for v in key):
                continue
            if key not in keys_seen:
                keys_seen.add(key); keys.append(obj)

        it = 1
        while it < self.max_iter:
            nxt = key_scan_iter_prompt()
            context.extend([("user", first), ("assistant", json.dumps(ensure_list_of_dicts(data)))])
            resp = self.llm.chat(llm_messages(context, nxt))
            total_tokens += resp.usage_tokens; total_latency += resp.latency_s
            data = try_parse_json(resp.text)
            new = 0
            for obj in ensure_list_of_dicts(data):
                key = tuple(obj.get(k) for k in self.key_attrs)
                if all(v in (None, "", []) for v in key):
                    continue
                if key not in keys_seen:
                    keys_seen.add(key); keys.append(obj); new += 1
            if new == 0:
                break
            it += 1

        # Phase 2: tuple-by-key
        for kobj in keys:
            key_json = json.dumps(kobj, ensure_ascii=False)
            prompt = tuple_by_key_prompt(tname, self.nonkey_attrs, key_json, schema_tuple)
            print("Tuple-by-key prompt:", prompt)
            resp = self.llm.chat(llm_messages([], prompt))
            print("Tuple-by-key response:", resp.text)
            total_tokens += resp.usage_tokens; total_latency += resp.latency_s
            data = try_parse_json(resp.text)
            candidates = ensure_list_of_dicts(data)

            # Normalizza e filtra righe completamente vuote
            normalized = []
            for obj in candidates:
                row_nonkeys = normalize_row(obj, self.nonkey_attrs)
                if all(row_nonkeys[c] in (None, "", []) for c in self.nonkey_attrs):
                    continue
                normalized.append(row_nonkeys)

            if not normalized:
                continue

            raw = {**kobj, **normalized[0]}
            attrs = list(dict.fromkeys(self.key_attrs + self.nonkey_attrs))
            row = normalize_row(raw, attrs)
            rows.append(row)


        return ScanResult(rows, total_tokens, total_latency)


def classify_atoms(llm: BaseLLM, table_name: str, atoms: List[str]) -> Dict[str, str]:
    if not atoms:
        return {}
    prompt = classify_where_atoms_prompt(table_name, atoms)
    resp = llm.chat([{"role": "user", "content": prompt}])
    data = try_parse_json(resp.text) or []
    out = {}
    for obj in data:
        atom = obj.get("atom")
        conf = str(obj.get("confidence", "low")).lower()
        if atom:
            out[atom] = "high" if conf.startswith("h") else "low"
    # default low for unmentioned atoms
    for a in atoms:
        out.setdefault(a, "low")
    return out

def ask_confidence(llm: BaseLLM, table_name: str, key_attrs: List[str], conds: List[str], select_attrs: List[str]) -> float:
    prompt = confidence_prompt(table_name, key_attrs, conds, select_attrs)
    resp = llm.chat([{"role": "user", "content": prompt}])
    try:
        data = try_parse_json(resp.text) or {}
        c = float(data.get("confidence", 0.0))
        return max(0.0, min(1.0, c))
    except Exception:
        return 0.0

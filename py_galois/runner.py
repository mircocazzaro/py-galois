import argparse, json, time, re, traceback
from pathlib import Path
from typing import List, Dict, Any
from .llm_client import OpenAIClient, MockLLM, BaseLLM, WatsonxClient, FoundryOpenAIClient, OpenRouterClient, OllamaClient, AzureGrokClient
from .dataio import load_dataset_meta, load_queries, load_nl_queries
from .logic import build_plans, execute_plan, final_answer_from_scans
from .json_utils import try_parse_json, ensure_list_of_dicts


def make_llm(provider: str) -> BaseLLM:
    if provider == "azure-foundry-openai":
        return FoundryOpenAIClient()
    if provider == "mock":
        return MockLLM()
    if provider in {"grok", "azure-grok"}:
        return AzureGrokClient()
    if provider.startswith("openai"):
        parts = provider.split(":", 1)
        model = parts[1] if len(parts) == 2 else None
        return OpenAIClient(model=model, azure=False)
    if provider.startswith("azure"):
        return OpenAIClient(model=None, azure=True)
    if provider.startswith("watsonx"):
        parts = provider.split(":", 1)
        model_id = parts[1] if len(parts) == 2 else None
        return WatsonxClient(model_id=model_id)
    if provider == "openrouter":
        return OpenRouterClient()
    if provider.startswith("ollama"):
        # examples:
        #   ollama
        #   ollama:llama3.1:8b
        parts = provider.split(":", 1)
        model = parts[1] if len(parts) == 2 else "llama3.1:8b"
        return OllamaClient(model=model)
    raise ValueError(f"Unknown provider: {provider}")

def parse_select_columns(sql: str) -> List[str]:
    s = sql.replace("\n", " ")
    m = re.search(r"select (.+?) from ", s, flags=re.IGNORECASE)
    if not m:
        return []
    part = m.group(1)
    cols = [c.strip() for c in part.split(",") if c.strip()]
    out = []
    for c in cols:
        # Prefer explicit alias: "expr AS alias"
        m_as = re.search(r"\bas\b\s+([A-Za-z_][\w]*)", c, flags=re.IGNORECASE)
        if m_as:
            out.append(m_as.group(1))
            continue
        # Drop qualifiers like "t.col"
        c2 = re.sub(r"[A-Za-z_][\w]*\.", "", c)
        # Crude handling of functions: turn "func(col)" into "func_col"
        c2 = re.sub(r"\((.*?)\)", r"_\1", c2)
        # Strip non-alphanumerics
        c2 = re.sub(r"[^A-Za-z0-9_]+", "_", c2).strip("_")
        out.append(c2 or "col")

    # Deduplicate while preserving order
    seen = set()
    out2 = []
    for c in out:
        if c not in seen:
            seen.add(c)
            out2.append(c)
    return out2


def baseline_prompt_and_eval(
    llm: BaseLLM,
    mode: str,
    sql: str,
    nl: str,
    ds_tables_meta: Dict[str, Any],
) -> tuple[list[dict], int, float]:
    # Decide output columns
    cols = parse_select_columns(sql)
    if not cols:
        cols = []
        for t in ds_tables_meta.values():
            for c in t.get("columns", []):
                cols.append(c["name"])
                if len(cols) >= 3:
                    break
            if len(cols) >= 3:
                break

    # JSON schema per "array di record" con queste colonne
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                c: {"type": ["string", "number", "integer", "boolean"]} for c in cols
            },
            "required": cols,
        },
    }

    if mode == "nl" and nl:
        prompt = (
            f"Answer the following question by returning structured rows. "
            f"Question: {nl} "
            f"Respond with JSON only, no commentary, no duplicates. This task must be completed within a limited time, so act quickly. Use this JSON schema: "
            f"{json.dumps(schema, ensure_ascii=False)}"
        )
    else:
        prompt = (
            f"Execute the following SQL over your internal knowledge and return the SELECT results as rows. "
            f"SQL: {sql} "
            f"Respond with JSON only, no commentary, no duplicates. This task must be completed within a limited time, so act quickly. Use this JSON schema: "
            f"{json.dumps(schema, ensure_ascii=False)}"
        )

    t0 = time.perf_counter()
    print(f"Baseline prompt:\n{prompt}\n", flush=True)
    resp = llm.chat([{"role": "user", "content": prompt}])
    print(f"Response:\n{resp.text}\n", flush=True)
    elapsed = time.perf_counter() - t0
    data = try_parse_json(resp.text) or []
    rows = ensure_list_of_dicts(data)
    tokens = resp.usage_tokens or 0
    return rows, tokens, elapsed


# ---------------------------------------------------------------------
# Helper: extract tables from FROM/JOIN, handling schema.table & aliases
# ---------------------------------------------------------------------
def extract_tables_in_query(sql: str, ds_tables_meta: Dict[str, Any]) -> set[str]:
    """
    Heuristically extract the set of dataset tables actually used in the query,
    based on the FROM / JOIN clauses.

    Handles:
      - Simple:   FROM airports a, flights f
      - Joins:    FROM airports a JOIN flights f ON ...
      - Schema:   FROM my_schema.airports a
      - Quoted:   FROM "my_schema"."airports" AS a

    Returns a set of *metadata keys* (as in ds_tables_meta.keys()).
    If nothing can be matched, falls back to all tables.
    """
    meta_keys = list(ds_tables_meta.keys())
    if not meta_keys:
        return set()

    # Map lowercased keys for case-insensitive matching
    lower_map = {k.lower(): k for k in meta_keys}

    # Normalize whitespace
    s = re.sub(r"\s+", " ", sql).strip()
    lower = s.lower()

    # Find FROM
    m_from = re.search(r"\bfrom\b", lower)
    if not m_from:
        # No FROM clause (weird): fall back to all tables
        return set(meta_keys)

    start = m_from.end()

    # Find end of FROM ... (before WHERE / GROUP BY / ORDER BY / etc.)
    end = len(s)
    tail = lower[start:]
    # Look for earliest occurrence of any clause keyword
    for kw in [" where ", " group by ", " order by ", " having ", " limit ", " union ", " intersect ", " except "]:
        m_kw = re.search(kw, tail)
        if m_kw:
            candidate_end = start + m_kw.start()
            if candidate_end < end:
                end = candidate_end
                break

    from_clause = s[start:end].strip()
    if not from_clause:
        return set(meta_keys)

    # Regex: capture table refs after start, commas or JOINs:
    # ( ^ | , | JOIN ) <whitespace> (schema.table | table) [alias...]
    table_refs: list[str] = []
    pattern = r'(?:^|,|\bjoin\b)\s*([A-Za-z_"][\w"]*(?:\.[A-Za-z_"][\w"]*)?)'
    for m in re.finditer(pattern, from_clause, flags=re.IGNORECASE):
        ref = m.group(1).strip()
        if ref:
            table_refs.append(ref)

    used_meta_keys: set[str] = set()

    for ref in table_refs:
        # Split schema.table, strip quotes
        parts = [p.strip('"` ') for p in ref.split(".") if p.strip('"` ')]
        candidate_full = ".".join(parts) if parts else ref.strip('"` ')
        candidate_short = parts[-1] if parts else candidate_full

        # Try exact matches
        if candidate_full in ds_tables_meta:
            used_meta_keys.add(candidate_full)
            continue
        if candidate_short in ds_tables_meta:
            used_meta_keys.add(candidate_short)
            continue

        # Try case-insensitive matches
        full_lower = candidate_full.lower()
        short_lower = candidate_short.lower()
        if full_lower in lower_map:
            used_meta_keys.add(lower_map[full_lower])
            continue
        if short_lower in lower_map:
            used_meta_keys.add(lower_map[short_lower])
            continue

    # If we failed to resolve anything, be conservative: use all tables
    if not used_meta_keys:
        return set(meta_keys)

    return used_meta_keys


def run_dataset(ds_dir: Path, provider: str, tau: float, out_base: Path, mode: str):
    """
    Writes one JSON per query in: out_base/<dataset>/qXX.json
    And one log per query in:     out_base/<dataset>/qXX.log
    JSON format:
    {
      "result_set": [ {...}, {...} ],
      "time": <float_seconds>,
      "tokens": <int_total_tokens>
    }
    """
    llm = make_llm(provider)
    meta = load_dataset_meta(ds_dir)
    tables = meta["tables"]

    # Global metadata (all tables) for the dataset
    select_attrs_by_table = {
        tname: [c["name"] for c in tmeta["columns"]]
        for tname, tmeta in tables.items()
    }
    key_attrs_by_table = {
        tname: list(tmeta.get("keys", []))
        for tname, tmeta in tables.items()
    }

    queries = load_queries(ds_dir)
    nlqs = load_nl_queries(ds_dir)

    ds_out = out_base / ds_dir.name  # sumbissions/<dataset>
    ds_out.mkdir(parents=True, exist_ok=True)

    for idx, sql in enumerate(queries, 1):
        qid = f"query{idx}"
        json_path = ds_out / f"{qid}.json"
        log_path = ds_out / f"{qid}.log"

        t_query = time.perf_counter()

        try:
            if mode in ("nl", "sql"):
                # Baselines: no TableScan/KeyScan
                nl = nlqs[idx - 1] if mode == "nl" and idx - 1 < len(nlqs) else ""
                rows, tokens, elapsed = baseline_prompt_and_eval(llm, mode, sql, nl, tables)

                # Write JSON
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {"result_set": rows, "time": elapsed, "tokens": tokens},
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

                # Log
                with log_path.open("w", encoding="utf-8") as f:
                    f.write(f"[{ds_dir.name}] {qid} mode={mode}\n")
                    if mode == "nl":
                        f.write(f"NL: {nl}\n")
                    f.write(f"SQL: {sql}\n")
                    f.write(f"Result rows: {len(rows)}\n")
                    f.write(f"Tokens: {tokens}\n")
                    f.write(f"Time (s): {elapsed:.3f}\n")

                print(f"[{ds_dir.name}] {qid} -> {json_path.name} (rows={len(rows)})")

            else:
                # Full Galois variants (use only tables actually appearing in FROM/JOIN)
                from .logic import LogicalPlan

                # Determine which tables the query really uses
                tables_in_query = extract_tables_in_query(sql, tables)

                # Restrict metadata and attribute maps to those tables
                query_tables_meta = {t: tables[t] for t in tables_in_query}
                select_attrs_q = {t: select_attrs_by_table[t] for t in tables_in_query}
                key_attrs_q = {t: key_attrs_by_table[t] for t in tables_in_query}

                t_all = time.perf_counter()

                if mode == "galois_wo":
                    # No pushdown; always Key-Scan, only on tables in FROM
                    per_table = {
                        tname: {
                            "strategy": "none",
                            "atoms": [],
                            "pushed_cond": None,
                            "physical": "key",
                            "conf_keys": None,
                            "conf_q": None,
                        }
                        for tname in query_tables_meta
                    }
                    plan = LogicalPlan(per_table)

                elif mode == "galois_s":
                    # Stessa pianificazione di Galois_F ma pushdown "single" per tabella
                    plan = build_plans(llm, sql, query_tables_meta, select_attrs_q, key_attrs_q, tau)
                    for cfg in plan.per_table.values():
                        atoms = cfg.get("atoms") or []
                        if atoms:
                            cfg["atoms"] = atoms[:1]
                            cfg["strategy"] = "single"
                            cfg["pushed_cond"] = atoms[0]
                        else:
                            cfg["atoms"] = []
                            cfg["strategy"] = "none"
                            cfg["pushed_cond"] = None

                elif mode == "galois_a":
                    # Stessa pianificazione di Galois_F ma pushdown "all"
                    plan = build_plans(llm, sql, query_tables_meta, select_attrs_q, key_attrs_q, tau)
                    for cfg in plan.per_table.values():
                        atoms = cfg.get("atoms") or []
                        if atoms:
                            cfg["strategy"] = "all"
                            cfg["pushed_cond"] = " AND ".join(atoms)
                        else:
                            cfg["strategy"] = "none"
                            cfg["pushed_cond"] = None

                else:
                    # Galois_F (default): full logical + physical optimization
                    plan = build_plans(llm, sql, query_tables_meta, select_attrs_q, key_attrs_q, tau)

                # Execute only on tables used in the query
                rows_by_table, tokens, scan_time, exec_logs = execute_plan(
                    llm, plan, query_tables_meta, select_attrs_q
                )

                actual_rows = final_answer_from_scans(rows_by_table, sql)
                elapsed_total = time.perf_counter() - t_all

                # Write JSON
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {"result_set": actual_rows, "time": elapsed_total, "tokens": tokens},
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

                # Write log
                with log_path.open("w", encoding="utf-8") as f:
                    f.write(f"[{ds_dir.name}] {qid} mode={mode} tau={tau}\n")
                    f.write(f"SQL: {sql}\n\n")
                    f.write(f"Tables in query: {', '.join(sorted(tables_in_query))}\n\n")
                    f.write("Plan per table:\n")
                    for tname, cfg in plan.per_table.items():
                        f.write(
                            f"  - {tname}: strategy={cfg['strategy']}, "
                            f"pushed={cfg['pushed_cond']}, physical={cfg['physical']}, "
                            f"conf_keys={cfg.get('conf_keys')}, conf_q={cfg.get('conf_q')}\n"
                        )
                    f.write("\nExecution:\n")
                    for line in exec_logs:
                        f.write("  " + line + "\n")
                    f.write(
                        f"\nTotals: tokens={tokens}, scan_time={scan_time:.3f}s, "
                        f"elapsed_total={elapsed_total:.3f}s\n"
                    )

                print(f"[{ds_dir.name}] {qid} -> {json_path.name} (rows={len(actual_rows)})")

        except Exception as e:
            elapsed = time.perf_counter() - t_query
            err_text = str(e)
            tb = traceback.format_exc()

            # Se Azure/OpenAI blocca per content filter, salta la query
            if ("content_filter" in err_text) or ("content management policy" in err_text):
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {"result_set": [], "time": elapsed, "tokens": 0},
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

                with log_path.open("w", encoding="utf-8") as f:
                    f.write(f"[{ds_dir.name}] {qid} mode={mode} tau={tau}\n")
                    f.write(f"SQL: {sql}\n\n")
                    f.write("STATUS: skipped due to provider content filter\n\n")
                    f.write(f"Error: {err_text}\n\n")
                    f.write("Traceback:\n")
                    f.write(tb)

                print(f"[{ds_dir.name}] {qid} skipped due to content filter")
                continue

            # Qualsiasi altro errore lo rilanciamo
            raise

def main():
    ap = argparse.ArgumentParser(description="Run Galois-like executor over datasets")
    ap.add_argument("--data-root", type=str, required=True, help="Path to data root with dataset folders")
    ap.add_argument(
        "--datasets",
        type=str,
        default="flight-2,flight-4,geo,movies,presidents,world",
        help="Comma-separated list",
    )
    ap.add_argument(
        "--provider",
        type=str,
        default="openai:gpt-4o-mini",
        help="mock | openai[:model] | azure | azure-foundry-openai | watsonx[:model_id] | grok",
    )
    ap.add_argument(
        "--tau",
        type=float,
        default=0.6,
        help="Confidence threshold for Key-Scan vs Table-Scan",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="sumbissions",
        help="Base output folder (will create sumbissions/<dataset>)",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="galois_f",
        choices=["galois_f", "galois_wo", "galois_s", "galois_a", "nl", "sql"],
        help="Execution mode",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_base = Path(args.out)

    for name in args.datasets.split(","):
        ds_dir = data_root / name.strip()
        if not ds_dir.exists():
            print(f"Skip missing dataset: {ds_dir}")
            continue
        run_dataset(ds_dir, args.provider, args.tau, out_base, args.mode)


if __name__ == "__main__":
    main()

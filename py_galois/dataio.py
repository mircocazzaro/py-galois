
import json, re
from pathlib import Path
from typing import Dict, List, Tuple

def load_dataset_meta(ds_dir: Path) -> Dict:
    # Find a *.json file with schema info
    jfiles = list(ds_dir.glob("*.json"))
    if not jfiles:
        raise FileNotFoundError(f"No schema JSON in {ds_dir}")
    meta = json.loads(jfiles[0].read_text(encoding='utf-8'))
    # Normalize to our internal format:
    tables = {}
    for t in meta.get("tables", []):
        name = t["name"].replace("target.", "").replace("target_", "")
        tables[name] = {
            "name": name,
            "keys": t.get("keys", []),
            "columns": t.get("columns", [])
        }
    return {"tables": tables}

def load_queries(ds_dir: Path) -> List[str]:
    # Read queries_*.sql file
    qfiles = sorted(ds_dir.glob("queries_*.sql"))
    if not qfiles:
        raise FileNotFoundError(f"No queries_*.sql in {ds_dir}")
    text = qfiles[0].read_text(encoding='utf-8')
    # split by --query markers or semicolons
    chunks = []
    current = []
    for line in text.splitlines():
        if line.strip().lower().startswith("--query"):
            if current:
                chunks.append("\n".join(current).strip())
                current = []
            continue
        current.append(line)
    if current:
        chunks.append("\n".join(current).strip())
    # Split further on semicolons if needed
    queries = []
    for c in chunks:
        for q in c.split(";"):
            s = q.strip()
            if s:
                queries.append(s + ";")
    return queries


def load_nl_queries(ds_dir: Path) -> List[str]:
    qfiles = sorted(ds_dir.glob("queries_*.txt"))
    if not qfiles:
        # Some datasets may not include NL; return empty placeholders
        return []
    lines = qfiles[0].read_text(encoding='utf-8').strip().splitlines()
    # remove leading numbering like "--query 1:" if present
    cleaned = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if ln.lower().startswith('--query'):
            continue
        cleaned.append(ln)
    return cleaned

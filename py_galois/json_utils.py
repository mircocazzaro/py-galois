
import json, re
from typing import Any, List, Dict, Tuple, Optional

def try_parse_json(text: str) -> Any:
    """Attempt to parse the LLM output as JSON. Try a few simple repairs heuristics."""
    text = text.strip()
    # If it looks like code fences, strip them
    text = re.sub(r'^```(json)?', '', text).strip()
    text = re.sub(r'```$', '', text).strip()
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to find the first [...] array
    m = re.search(r'\[.*\]', text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # Try to make it a list
    if text and not (text.startswith('[') and text.endswith(']')):
        cand = f"[{text}]"
        try:
            return json.loads(cand)
        except Exception:
            pass
    # Remove trailing commas
    text2 = re.sub(r',\s*([\}\]])', r'\1', text)
    try:
        return json.loads(text2)
    except Exception:
        pass
    # Last resort
    return None

def normalize_row(row: Dict[str, Any], columns: List[str]) -> Dict[str, Any]:
    """Keep only known columns, drop extras, cast primitives if possible."""
    out = {}
    for c in columns:
        v = row.get(c, None)
        out[c] = v
    return out

def ensure_list_of_dicts(obj: Any) -> List[Dict[str, Any]]:
    if obj is None:
        return []
    if isinstance(obj, dict):
        # sometimes LLM returns a single object
        return [obj]
    if isinstance(obj, list):
        # ensure dicts
        return [x for x in obj if isinstance(x, dict)]
    return []


from typing import Dict, List, Optional
import json

def json_schema_for_table(table: Dict) -> Dict:
    """Build a minimal JSON schema for a list of tuples (array of objects)."""
    props = {}
    required = []
    for col in table.get("columns", []):
        name = col["name"]
        dtype = col.get("dtype", "string").lower()
        if dtype in ("int", "integer", "bigint"):
            t = "integer"
        elif dtype in ("float", "double", "real", "numeric", "decimal"):
            t = "number"
        elif dtype in ("boolean", "bool"):
            t = "boolean"
        else:
            t = "string"
        props[name] = {"type": [t, "string"] if t != "string" else ["string"]}
        required.append(name)
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": props,
            "required": required
        }
    }

def table_scan_first_prompt(sql_query: str, table_name: str, attrs: List[str], json_schema: Dict, cond: Optional[str] = None) -> str:
    attrs_str = ", ".join(attrs)
    schema_str = json.dumps(json_schema, ensure_ascii=False)
    if cond:
        return (
            f"Given the following query, populate the table with actual values. "
            f"query: select {attrs_str} from {table_name} where {cond}. "
            f"Respond with JSON only. Don’t add any comment. "
            f"Use the following JSON schema: {schema_str}."
        )
    else:
        return (
            f"Given the following query, populate the table with actual values. "
            f"query: select {attrs_str} from {table_name}. "
            f"Respond with JSON only. Don’t add any comment. "
            f"Use the following JSON schema: {schema_str}."
        )

def table_scan_iter_prompt() -> str:
    return "List more values if there are more, otherwise return an empty JSON. Respond with JSON only."

def key_scan_first_prompt(table_name: str, key_attrs: List[str], json_schema: Dict, cond: Optional[str] = None) -> str:
    keys = ", ".join(key_attrs)
    schema_str = json.dumps(json_schema, ensure_ascii=False)
    if cond:
        return (
            f"List the key of {table_name} (where the following condition holds: {cond}). "
            f"Respond with JSON only. Use the following JSON schema: {schema_str}."
        )
    else:
        return (
            f"List the key of {table_name}. "
            f"Respond with JSON only. Use the following JSON schema: {schema_str}."
        )

def key_scan_iter_prompt() -> str:
    return "List more unique values if there are more, otherwise return an empty response. Don’t repeat the previous values."

def tuple_by_key_prompt(table_name: str, nonkey_attrs: List[str], key_value_json: str, json_schema: Dict) -> str:
    attrs_str = ", ".join(nonkey_attrs)
    schema_str = json.dumps(json_schema, ensure_ascii=False)
    return (
        f"List the attributes of the table for {key_value_json}. "
        f"Respond with JSON only. Use the following JSON schema: {schema_str}."
    )


def classify_where_atoms_prompt(table_name: str, atoms: List[str]) -> str:
    atoms_json = json.dumps(atoms, ensure_ascii=False)
    return (
        "You are helping to optimize SQL over an LLM. "
        + f"For table {table_name}, consider the following WHERE atoms: {atoms_json}. "
        + "For each atom, label your confidence that the LLM can reliably answer it when pushed into the scan as 'high' or 'low'. "
        + "Respond with pure JSON as a list of objects: [{\"atom\": <atom>, \"confidence\": \"high\"|\"low\"}]. Adhere to the JSON format STRICTLY and do not add any comments or explanations outside the JSON."
    )

def confidence_prompt(table_name: str, key_attrs: List[str], conds: List[str], select_attrs: List[str]) -> str:
    conds_text = " AND ".join(conds) if conds else "(none)"
    n = len(select_attrs)
    return (
        "We will use an LLM to list all key values for table "
        + f"{table_name} given WHERE conditions: {conds_text}. "
        + "On a scale from 0 to 1, what's your confidence you can list ALL distinct key values accurately? "
        + "Respond with JSON only as {\"confidence\": <number between 0 and 1>}. "
        + f"Note: we will use conf(q) = confidence^{n} where n=#selected attributes."
    )

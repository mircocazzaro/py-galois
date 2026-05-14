
from typing import Dict, List, Any
from pathlib import Path
import pandas as pd
import os, re, json

def load_csv_tables(ds_dir: Path) -> Dict[str, pd.DataFrame]:
    csvs = {}
    for p in ds_dir.glob("*.csv"):
        name = p.stem  # e.g., city, country, etc.
        df = pd.read_csv(p)
        csvs[name] = df
    return csvs

def expected_with_duckdb(csvs: Dict[str, Any], sql: str) -> List[Dict[str, Any]]:
    import duckdb  # requires duckdb installed
    con = duckdb.connect(database=':memory:')
    # Register tables
    for name, df in csvs.items():
        con.register(name, df)
    # Replace target. prefix
    sql = sql.replace('target.', '')
    res = con.execute(sql).df()
    return res.to_dict(orient='records')

def expected_with_sqlite(csvs: Dict[str, Any], sql: str) -> List[Dict[str, Any]]:
    import sqlite3
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    # Create and populate
    for name, df in csvs.items():
        cols = ', '.join([f'{c} TEXT' for c in df.columns])
        cur.execute(f'CREATE TABLE {name}({cols})')
        for _, r in df.iterrows():
            cur.execute(f'INSERT INTO {name} VALUES ({",".join(["?"]*len(df.columns))})', [str(v) if v is not None else None for v in r.values.tolist()])
    # crude translation
    s = sql.replace('target.', '')
    s = re.sub(r'TRY_CAST\s*\(', 'CAST(', s, flags=re.IGNORECASE)
    s = re.sub(r'\bTRUE\b', '1', s, flags=re.IGNORECASE)
    s = re.sub(r'\bFALSE\b', '0', s, flags=re.IGNORECASE)
    try:
        cur.execute(s)
        out = [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()
    return out

def expected_rows(ds_dir: Path, sql: str) -> List[Dict[str, Any]]:
    csvs = load_csv_tables(ds_dir)
    # Try duckdb first
    try:
        return expected_with_duckdb(csvs, sql)
    except Exception:
        # fallback to sqlite
        return expected_with_sqlite(csvs, sql)

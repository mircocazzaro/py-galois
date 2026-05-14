
from typing import List, Tuple, Any, Dict
import math, re

def levenshtein(a: str, b: str) -> int:
    # classic DP
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = tmp
    return dp[m]

def normalize_cell(x: Any) -> Tuple[str, bool, float]:
    """Return (string_value, is_numeric, numeric_value)."""
    if x is None:
        return ("", False, 0.0)
    if isinstance(x, (int, float)):
        return (str(x), True, float(x))
    s = str(x).strip()
    # Try to parse numeric in the string
    m = re.search(r'-?\d+(?:\.\d+)?', s.replace(',', ''))
    if m:
        try:
            val = float(m.group(0))
            return (s, True, val)
        except Exception:
            pass
    return (s, False, 0.0)

def close_enough(expected: Any, actual: Any) -> bool:
    se, isne, ve = normalize_cell(expected)
    sa, isna, va = normalize_cell(actual)
    if isne and isna:
        if ve == 0:
            return abs(va) < 1e-9
        return abs(va - ve) <= 0.10 * abs(ve)  # ±10%
    # string-ish compare by edit distance 10% of expected length
    a, b = se.lower(), sa.lower()
    if not a and not b:
        return True
    if not a or not b:
        return False
    dist = levenshtein(a, b)
    thr = max(1, int(0.10 * len(a)))
    return dist <= thr

def f1_cell(expected_rows: List[Dict[str, Any]], actual_rows: List[Dict[str, Any]]) -> float:
    # Flatten to multiset of cells (string repr) with tolerance; We'll do conservative match count
    exp_cells = []
    for r in expected_rows:
        for v in r.values():
            exp_cells.append(v)
    act_cells = []
    for r in actual_rows:
        for v in r.values():
            act_cells.append(v)
    # greedy matching
    matched = 0
    used = [False] * len(act_cells)
    for e in exp_cells:
        found = False
        for i, a in enumerate(act_cells):
            if not used[i] and close_enough(e, a):
                used[i] = True
                matched += 1
                found = True
                break
    prec = matched / len(act_cells) if act_cells else 0.0
    rec = matched / len(exp_cells) if exp_cells else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def cardinality_score(expected_rows: List[Dict[str, Any]], actual_rows: List[Dict[str, Any]]) -> float:
    a, b = len(expected_rows), len(actual_rows)
    if max(a,b) == 0:
        return 1.0
    return min(a,b) / max(a,b)

def tuple_constraint(expected_rows: List[Dict[str, Any]], actual_rows: List[Dict[str, Any]]) -> float:
    # fraction of expected tuples present in actual (order-insensitive), with tolerant cell matching
    matched = 0
    used = [False] * len(actual_rows)
    for e in expected_rows:
        found = False
        for i, a in enumerate(actual_rows):
            if used[i]: 
                continue
            if len(e) != len(a):
                continue
            ok = True
            for k, v in e.items():
                if k not in a or not close_enough(v, a[k]):
                    ok = False; break
            if ok:
                used[i] = True
                matched += 1
                found = True
                break
    return matched / len(expected_rows) if expected_rows else 1.0

def avg_score(expected_rows: List[Dict[str, Any]], actual_rows: List[Dict[str, Any]]) -> float:
    f1 = f1_cell(expected_rows, actual_rows)
    card = cardinality_score(expected_rows, actual_rows)
    tc = tuple_constraint(expected_rows, actual_rows)
    return (f1 + card + tc) / 3.0

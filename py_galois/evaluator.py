
from typing import Dict, List, Any, Tuple
from pathlib import Path
import time, json
from .metrics import f1_cell, cardinality_score, tuple_constraint, avg_score

class EvalResult:
    def __init__(self, f1: float, card: float, tc: float, avg: float, tokens: int, latency: float):
        self.f1 = f1; self.card = card; self.tc = tc; self.avg = avg
        self.tokens = tokens; self.latency = latency
    def to_dict(self):
        return {"f1_cell": self.f1, "cardinality": self.card, "tuple_constraint": self.tc, "avg_score": self.avg, "tokens": self.tokens, "time_s": self.latency}

def evaluate(expected: List[Dict[str, Any]], actual: List[Dict[str, Any]], tokens: int, latency: float) -> EvalResult:
    f1 = f1_cell(expected, actual)
    card = cardinality_score(expected, actual)
    tc = tuple_constraint(expected, actual)
    avg = avg_score(expected, actual)
    return EvalResult(f1, card, tc, avg, tokens, latency)

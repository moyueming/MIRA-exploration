import ast
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT_DIR / "ATENA-A-EDA" / "benchmark"


def avp_terms(schema: str, dataset_number: int) -> Dict[str, List[str]]:
    path = _avp_path(schema, dataset_number)
    if not path.exists():
        return {}

    tree = ast.parse(path.read_text(encoding="utf-8"))
    terms: Dict[str, List[str]] = defaultdict(list)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or getattr(node.func, "id", None) != "FilterAction":
            continue
        kwargs = {keyword.arg: keyword.value for keyword in node.keywords}
        column = _literal_column(kwargs.get("filtered_column"))
        term = _literal_constant(kwargs.get("filter_term"))
        if column and term is not None and str(term) not in terms[column]:
            terms[column].append(str(term))
    return dict(terms)


def avp_details(schema: str, dataset_number: int) -> Dict[str, object]:
    path = _avp_path(schema, dataset_number)
    terms = avp_terms(schema, dataset_number)
    encoded = json.dumps(
        terms, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema": schema.lower(),
        "dataset": int(dataset_number),
        "source_kind": "official_evaluator_reference",
        "source_path": path.as_posix(),
        "terms": terms,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _avp_path(schema: str, dataset_number: int) -> Path:
    return (
        BENCHMARK_DIR
        / "atena"
        / "evaluation"
        / "references"
        / schema.lower()
        / "dataset{}.py".format(int(dataset_number))
    )


def _literal_column(node) -> Optional[str]:
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Column" and node.args:
        value = _literal_constant(node.args[0])
        return None if value is None else str(value)
    return None


def _literal_constant(node):
    if isinstance(node, ast.Constant):
        return node.value
    return None

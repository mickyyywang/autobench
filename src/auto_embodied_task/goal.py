from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Callable

from .models import normalize_relation


PredicateFn = Callable[[str, list[Any]], bool]


@dataclass
class GoalEvaluation:
    success: bool
    expression: Any
    checks: list[dict[str, Any]] = field(default_factory=list)


def evaluate_goal_expression(expression: Any, predicate_fn: PredicateFn) -> GoalEvaluation:
    checks: list[dict[str, Any]] = []
    success = _evaluate(expression, predicate_fn, checks)
    return GoalEvaluation(success=success, expression=expression, checks=checks)


def normalize_goal_expression(criterion: Any) -> Any:
    if isinstance(criterion, dict):
        if "final" in criterion:
            return normalize_goal_expression(criterion["final"])
        if "all" in criterion and "and" not in criterion:
            return {"and": [normalize_goal_expression(item) for item in _as_list(criterion["all"])]}
        if "any" in criterion and "or" not in criterion:
            return {"or": [normalize_goal_expression(item) for item in _as_list(criterion["any"])]}
        return criterion
    if isinstance(criterion, list):
        return criterion
    if isinstance(criterion, str):
        return _legacy_criterion_to_expression(criterion)
    return criterion


def extract_goal_predicates(expression: Any) -> list[str]:
    predicates: list[str] = []

    def visit(expr: Any) -> None:
        if isinstance(expr, dict):
            if "predicate" in expr:
                predicates.append(normalize_relation(str(expr["predicate"])))
                return
            for key in ("and", "or", "all", "any"):
                if key in expr:
                    for item in _as_list(expr[key]):
                        visit(item)
            if "not" in expr:
                visit(expr["not"])
            if "final" in expr:
                visit(expr["final"])
            return
        if isinstance(expr, list):
            if expr and isinstance(expr[0], str):
                head = normalize_relation(expr[0])
                if head not in {"AND", "OR", "NOT"}:
                    predicates.append(head)
            for item in expr:
                if isinstance(item, (dict, list)):
                    visit(item)

    visit(expression)
    return list(dict.fromkeys(predicates))


def _evaluate(expression: Any, predicate_fn: PredicateFn, checks: list[dict[str, Any]]) -> bool:
    expression = normalize_goal_expression(expression)
    if expression is None or expression == "":
        return False
    if isinstance(expression, dict):
        if "and" in expression:
            return all(_evaluate(item, predicate_fn, checks) for item in _as_list(expression["and"]))
        if "or" in expression:
            return any(_evaluate(item, predicate_fn, checks) for item in _as_list(expression["or"]))
        if "not" in expression:
            return not _evaluate(expression["not"], predicate_fn, checks)
        if "predicate" in expression:
            predicate = normalize_relation(str(expression["predicate"]))
            args = _as_list(expression.get("args", expression.get("arguments", [])))
            return _evaluate_predicate(predicate, args, predicate_fn, checks)
        if "final" in expression:
            return _evaluate(expression["final"], predicate_fn, checks)
        raise ValueError(f"unsupported goal expression object: {expression}")
    if isinstance(expression, list):
        if not expression:
            return False
        head = expression[0]
        if isinstance(head, str) and normalize_relation(head) in {"AND", "OR"}:
            children = expression[1:]
            if normalize_relation(head) == "AND":
                return all(_evaluate(item, predicate_fn, checks) for item in children)
            return any(_evaluate(item, predicate_fn, checks) for item in children)
        if isinstance(head, str) and normalize_relation(head) == "NOT":
            if len(expression) != 2:
                raise ValueError(f"not expression needs exactly one child: {expression}")
            return not _evaluate(expression[1], predicate_fn, checks)
        if isinstance(head, str):
            predicate = normalize_relation(head)
            args = list(expression[1:])
            return _evaluate_predicate(predicate, args, predicate_fn, checks)
        raise ValueError(f"unsupported goal list expression: {expression}")
    raise ValueError(f"unsupported goal expression: {expression!r}")


def _evaluate_predicate(
    predicate: str,
    args: list[Any],
    predicate_fn: PredicateFn,
    checks: list[dict[str, Any]],
) -> bool:
    success = bool(predicate_fn(predicate, args))
    checks.append({"predicate": predicate, "args": list(args), "success": success})
    return success


def _legacy_criterion_to_expression(criterion: str) -> Any:
    relations = []
    for match in re.finditer(r"\(([A-Za-z_]+),\s*([^)]+)\)", criterion):
        relation = normalize_relation(match.group(1))
        args = [item.strip() for item in match.group(2).split(",") if item.strip()]
        relations.append([relation, *args])
    if not relations:
        return None
    if len(relations) == 1:
        return relations[0]
    return {"and": relations}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]

from __future__ import annotations

import re
from typing import Any

from .action_model import ACTION_SCHEMAS


ZERO_ARGUMENT_ACTIONS = {"observe"}


def parse_manual_action(value: str) -> dict[str, Any]:
    match = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*", value)
    if not match:
        raise ValueError(f"manual action must use name(arg[, target]) syntax: {value!r}")
    name = match.group(1).lower()
    if name in ACTION_SCHEMAS:
        expected_arity = len(ACTION_SCHEMAS[name].parameters)
    elif name in ZERO_ARGUMENT_ACTIONS:
        expected_arity = 0
    else:
        raise ValueError(f"unknown manual action {name!r}")
    node_ids = [part.strip() for part in re.split(r"[,，]", match.group(2)) if part.strip()]
    if len(node_ids) != expected_arity:
        raise ValueError(f"manual action {name!r} expects {expected_arity} arguments, got {len(node_ids)}")
    action: dict[str, Any] = {
        "name": name,
        "base_name": name,
        "node_ids": node_ids,
    }
    if node_ids:
        action["object"] = node_ids[0]
    if len(node_ids) > 1:
        action["target"] = node_ids[1]
    return action

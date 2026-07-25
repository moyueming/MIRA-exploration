from typing import Dict, Iterable, Mapping


def action_structure_features(action_reprs: Iterable[str]) -> Dict[str, float]:
    actions = [str(action) for action in action_reprs]
    kinds = [_action_kind(action) for action in actions]
    fields = [_action_field(action) for action in actions if _action_kind(action) in {"FILTER", "GROUP"}]
    group_fields = [_action_field(action) for action in actions if _action_kind(action) == "GROUP"]
    root_children = _root_children(kinds)
    filter_back_alternations = sum(
        1 for idx in range(1, len(kinds))
        if {kinds[idx - 1], kinds[idx]} == {"FILTER", "BACK"}
    )
    unique_actions = len(set(actions))
    unique_fields = len({field for field in fields if field})
    unique_groups = len({field for field in group_fields if field})
    selection_score = (
        (0.35 * root_children)
        + (0.35 * unique_fields)
        + (0.15 * unique_actions)
        - (0.20 * filter_back_alternations)
    )
    return {
        "selection_score": float(selection_score),
        "action_root_children": float(root_children),
        "action_unique_fields": float(unique_fields),
        "action_unique_groups": float(unique_groups),
        "action_unique_actions": float(unique_actions),
        "action_filter_back_alternations": float(filter_back_alternations),
    }


def attach_selection_features(row: Mapping[str, object], action_reprs: Iterable[str]) -> Dict[str, object]:
    enriched = dict(row)
    enriched.update(action_structure_features(action_reprs))
    return enriched


def _action_kind(action_repr: str) -> str:
    parts = str(action_repr).split()
    return parts[0] if parts else ""


def _action_field(action_repr: str) -> str:
    parts = str(action_repr).split()
    return parts[1] if len(parts) > 1 else ""


def _root_children(kinds) -> int:
    next_node = 1
    stack = [0]
    children = {0: []}
    for kind in kinds:
        if kind == "BACK":
            if len(stack) > 1:
                stack.pop()
            continue
        parent = stack[-1]
        node = next_node
        next_node += 1
        children.setdefault(parent, []).append(node)
        children.setdefault(node, [])
        stack.append(node)
    return len(children.get(0, []))

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from .abstract import run_abstract
from .ast import collect_variables, expr_to_str, stmt_to_str
from .cfg import build_cfg
from .concrete import ConcreteResult, Value, run_concrete
from .parser import parse_program


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural Abstract Interpretation MVP")
    parser.add_argument("source", help="Path to the source file.")
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--abstract-iters", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=16)
    args = parser.parse_args()

    source_path = Path(args.source)
    source_text = source_path.read_text()

    program = parse_program(source_text)
    cfg = build_cfg(program)
    entry_args = _prompt_entry_args(cfg.functions["main"].params)
    concrete = run_concrete(cfg, max_steps=args.max_steps, entry_args=entry_args)

    variables = collect_variables(program)
    abstract = run_abstract(
        cfg,
        variables=variables,
        hidden_dim=args.hidden_dim,
        max_iters=args.abstract_iters,
    )

    output = {
        "trace": _serialize_trace(concrete),
        "abstract_states": abstract.states,
        "cfg": _serialize_cfg(cfg),
        "steps": concrete.steps,
        "halted": concrete.halted,
    }
    print(json.dumps(output, indent=2))


def _serialize_trace(concrete: ConcreteResult) -> List[Dict]:
    return [
        {"function": event.function, "pc": event.pc, "env": event.env}
        for event in concrete.trace
    ]


def _prompt_entry_args(params: List[str]) -> Dict[str, Value]:
    if not params:
        return {}
    values: Dict[str, Value] = {}
    for name in params:
        raw = input(f"Enter value for {name} (int/bool): ").strip()
        values[name] = _parse_value(raw)
    return values


def _parse_value(raw: str) -> Value:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid value: {raw}") from exc


def _serialize_cfg(cfg) -> Dict:
    data: Dict[str, Dict] = {"functions": {}}
    for func_name, func_cfg in cfg.functions.items():
        nodes = []
        for node_id, node in func_cfg.nodes.items():
            nodes.append(
                {
                    "id": node_id,
                    "kind": node.kind,
                    "stmt": stmt_to_str(node.stmt),
                    "expr": expr_to_str(node.expr) if node.expr is not None else None,
                }
            )
        edges = [
            {"source": edge.source, "target": edge.target, "label": edge.label}
            for edge_list in func_cfg.edges.values()
            for edge in edge_list
        ]
        data["functions"][func_name] = {
            "entry": func_cfg.entry,
            "exit": func_cfg.exit,
            "nodes": nodes,
            "edges": edges,
        }
    return data


if __name__ == "__main__":
    main()

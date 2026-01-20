from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from .abstract import create_domain, run_abstract, run_abstract_with_domain
from .ast import collect_variables, expr_to_str, stmt_to_str
from .cfg import build_cfg
from .concrete import ConcreteResult, Value, run_concrete
from .div_zero import analyze_div_by_zero
from .parser import parse_program
from .train import train_domain_on_concrete_traces, write_loss_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural Abstract Interpretation MVP")
    parser.add_argument("source", help="Path to the source file.")
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument(
        "--concrete-samples",
        type=int,
        default=1,
        help="Number of concrete executions to run (requires --random-entry-args if > 1).",
    )
    parser.add_argument(
        "--random-entry-args",
        action="store_true",
        help="Randomly generate entry args for main instead of prompting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for --random-entry-args (default: nondeterministic).",
    )
    parser.add_argument(
        "--int-min",
        type=int,
        default=-10,
        help="Minimum random int value (inclusive) for --random-entry-args.",
    )
    parser.add_argument(
        "--int-max",
        type=int,
        default=10,
        help="Maximum random int value (inclusive) for --random-entry-args.",
    )
    parser.add_argument(
        "--bool-prob",
        type=float,
        default=0.0,
        help="Probability a random argument is boolean (otherwise int).",
    )
    parser.add_argument("--abstract-iters", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=0,
        help="Train the neural domain using collected concrete traces.",
    )
    parser.add_argument(
        "--train-lr",
        type=float,
        default=1e-3,
        help="Learning rate for --train-epochs.",
    )
    parser.add_argument(
        "--loss-plot",
        type=Path,
        default=None,
        help="Write training loss curve using matplotlib (e.g. loss.png / loss.svg).",
    )
    parser.add_argument(
        "--div-zero-report",
        type=Path,
        help="Write div-by-zero report JSON to the given path.",
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    source_text = source_path.read_text()

    program = parse_program(source_text)
    cfg = build_cfg(program)
    params = cfg.functions["main"].params
    if args.concrete_samples > 1 and not args.random_entry_args:
        raise ValueError(
            "--concrete-samples > 1 requires --random-entry-args to collect varied inputs."
        )

    rng = random.Random(args.seed)
    concrete_runs: List[Dict] = []
    concrete_results: List[ConcreteResult] = []
    for _ in range(args.concrete_samples):
        if args.random_entry_args:
            entry_args = _random_entry_args(
                params=params,
                rng=rng,
                int_min=args.int_min,
                int_max=args.int_max,
                bool_prob=args.bool_prob,
            )
        else:
            entry_args = _prompt_entry_args(params)
        concrete = run_concrete(cfg, max_steps=args.max_steps, entry_args=entry_args)
        if concrete.error is None:
            concrete_results.append(concrete)
        concrete_runs.append(
            {
                "entry_args": entry_args,
                "trace": _serialize_trace(concrete),
                "steps": concrete.steps,
                "halted": concrete.halted,
                "error": concrete.error,
            }
        )

    variables = collect_variables(program)
    training_stats: List[Dict] = []
    domain = None
    if args.train_epochs > 0:
        if not concrete_results:
            raise ValueError(
                "No successful concrete runs to train on (all runs errored)."
            )
        domain = create_domain(variables=variables, hidden_dim=args.hidden_dim)
        stats = train_domain_on_concrete_traces(
            cfg=cfg,
            domain=domain,
            variables=variables,
            runs=concrete_results,
            epochs=args.train_epochs,
            lr=args.train_lr,
        )
        if args.loss_plot is not None:
            write_loss_plot(stats=stats, path=args.loss_plot)
        training_stats = [
            {
                "total_loss": s.total_loss,
                "transfer_loss": s.transfer_loss,
                "predicate_loss": s.predicate_loss,
                "steps": s.steps,
            }
            for s in stats
        ]
    elif args.loss_plot is not None:
        raise ValueError("--loss-plot requires --train-epochs > 0.")
    if args.div_zero_report is not None:
        abstract, domain, tensor_states = run_abstract_with_domain(
            cfg,
            variables=variables,
            hidden_dim=args.hidden_dim,
            max_iters=args.abstract_iters,
            domain=domain,
        )
        report = analyze_div_by_zero(
            cfg=cfg,
            abstract_states=tensor_states,
            domain=domain,
            source=str(source_path),
        )
        args.div_zero_report.write_text(json.dumps(report, indent=2))
    else:
        if domain is None:
            abstract = run_abstract(
                cfg,
                variables=variables,
                hidden_dim=args.hidden_dim,
                max_iters=args.abstract_iters,
            )
        else:
            abstract, _domain, _tensor_states = run_abstract_with_domain(
                cfg,
                variables=variables,
                hidden_dim=args.hidden_dim,
                max_iters=args.abstract_iters,
                domain=domain,
            )

    output = {
        "runs": concrete_runs,
        "training": training_stats,
        "abstract_states": abstract.states,
        "cfg": _serialize_cfg(cfg),
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


def _random_entry_args(
    *,
    params: List[str],
    rng: random.Random,
    int_min: int,
    int_max: int,
    bool_prob: float,
) -> Dict[str, Value]:
    if not params:
        return {}
    if int_min > int_max:
        raise ValueError("--int-min must be <= --int-max.")
    if not (0.0 <= bool_prob <= 1.0):
        raise ValueError("--bool-prob must be in [0, 1].")

    values: Dict[str, Value] = {}
    for name in params:
        if rng.random() < bool_prob:
            values[name] = bool(rng.getrandbits(1))
        else:
            values[name] = rng.randint(int_min, int_max)
    return values


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

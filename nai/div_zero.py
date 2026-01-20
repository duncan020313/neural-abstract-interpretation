from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch

from .abstract import _encode_expr
from .ast import (
    Assign,
    BinOp,
    BoolLiteral,
    Call,
    Expr,
    IntLiteral,
    Return,
    SourceSpan,
    UnaryOp,
    Var,
    expr_to_str,
)
from .cfg import CFG, CFGEdge, CFGNode, FunctionCFG
from .neural_domain import NeuralDomain


@dataclass(frozen=True)
class DivZeroFinding:
    function: str
    location: Dict[str, Optional[int]]
    expr: str
    path_condition: List[str]
    confidence: float


def analyze_div_by_zero(
    cfg: CFG,
    abstract_states: Dict[str, Dict[int, torch.Tensor]],
    domain: NeuralDomain,
    source: str,
) -> Dict[str, object]:
    findings: List[DivZeroFinding] = []
    with torch.no_grad():
        for func_name, func_cfg in cfg.functions.items():
            node_states = abstract_states.get(func_name, {})
            incoming = _incoming_edges(func_cfg)
            for node_id, node in func_cfg.nodes.items():
                state = node_states.get(node_id)
                if state is None:
                    continue
                for expr in _node_exprs(node):
                    for binop in _iter_div_binops(expr):
                        confidence = _confidence_for_denom(
                            denom=binop.right,
                            state=state,
                            domain=domain,
                        )
                        findings.append(
                            DivZeroFinding(
                                function=func_name,
                                location=_span_to_location(binop.span),
                                expr=expr_to_str(binop),
                                path_condition=_path_conditions(
                                    func_cfg, incoming, node_id
                                ),
                                confidence=confidence,
                            )
                        )
    return {
        "source": source,
        "findings": [finding.__dict__ for finding in findings],
    }


def _node_exprs(node: CFGNode) -> Iterable[Expr]:
    if node.kind in {"branch", "loop_cond"} and node.expr is not None:
        return [node.expr]
    if node.kind == "stmt" and node.stmt is not None:
        if isinstance(node.stmt, Assign):
            return [node.stmt.expr]
        if isinstance(node.stmt, Call):
            return list(node.stmt.args)
    if node.kind == "return" and isinstance(node.stmt, Return):
        return [node.stmt.expr]
    return []


def _iter_div_binops(expr: Expr) -> Iterable[BinOp]:
    if isinstance(expr, BinOp):
        if expr.op in {"/", "%"}:
            yield expr
        yield from _iter_div_binops(expr.left)
        yield from _iter_div_binops(expr.right)
        return
    if isinstance(expr, UnaryOp):
        yield from _iter_div_binops(expr.operand)
        return
    if isinstance(expr, (IntLiteral, BoolLiteral, Var)):
        return
    raise TypeError(f"Unsupported expr: {type(expr).__name__}")


def _confidence_for_denom(
    denom: Expr,
    state: torch.Tensor,
    domain: NeuralDomain,
) -> float:
    if isinstance(denom, IntLiteral):
        return 1.0 if denom.value == 0 else 0.0
    if isinstance(denom, BoolLiteral):
        return 1.0 if denom.value is False else 0.0
    eq_expr = BinOp(op="==", left=denom, right=IntLiteral(value=0))
    vec = _encode_expr(eq_expr)
    return float(domain.predicate(state, vec).item())


def _path_conditions(
    func_cfg: FunctionCFG,
    incoming: Dict[int, List[CFGEdge]],
    node_id: int,
) -> List[str]:
    conditions: List[str] = []
    seen = set()
    for edge in incoming.get(node_id, []):
        if edge.label not in {"true", "false"}:
            continue
        pred = func_cfg.nodes.get(edge.source)
        if pred is None or pred.expr is None:
            continue
        cond = expr_to_str(pred.expr)
        if edge.label == "false":
            cond = f"!({cond})"
        if cond in seen:
            continue
        seen.add(cond)
        conditions.append(cond)
    return conditions


def _incoming_edges(func_cfg: FunctionCFG) -> Dict[int, List[CFGEdge]]:
    incoming: Dict[int, List[CFGEdge]] = {node_id: [] for node_id in func_cfg.nodes}
    for edges in func_cfg.edges.values():
        for edge in edges:
            incoming[edge.target].append(edge)
    return incoming


def _span_to_location(span: Optional[SourceSpan]) -> Dict[str, Optional[int]]:
    if span is None:
        return {"line": None, "column": None, "end_line": None, "end_column": None}
    return {
        "line": span.line,
        "column": span.column,
        "end_line": span.end_line,
        "end_column": span.end_column,
    }

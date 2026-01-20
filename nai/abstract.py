from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .ast import (
    Assign,
    BinOp,
    BoolLiteral,
    Call,
    Expr,
    IntLiteral,
    Return,
    UnaryOp,
    Var,
)
from .cfg import CFG, CFGEdge, CFGNode, FunctionCFG
from .neural_domain import NeuralDomain

STMT_KINDS = ["assign", "call", "return", "noop"]
EXPR_KINDS = ["int", "bool", "var", "binop", "unary"]
EXPR_OPS = ["+", "-", "*", "/", "%", "==", "<", "<=", "&&", "||", "!", "neg"]


@dataclass(frozen=True)
class AbstractResult:
    states: Dict[str, Dict[int, List[float]]]


def create_domain(*, variables: List[str], hidden_dim: int = 16) -> NeuralDomain:
    return NeuralDomain(
        env_dim=len(variables),
        stmt_dim=len(STMT_KINDS),
        expr_dim=len(EXPR_KINDS) + len(EXPR_OPS),
        hidden_dim=hidden_dim,
    )


def run_abstract(
    cfg: CFG,
    variables: List[str],
    hidden_dim: int = 16,
    max_iters: int = 20,
    eps: float = 1e-3,
    beta: float = 0.5,
) -> AbstractResult:
    result, _domain, _tensor_states = run_abstract_with_domain(
        cfg=cfg,
        variables=variables,
        hidden_dim=hidden_dim,
        max_iters=max_iters,
        eps=eps,
        beta=beta,
    )
    return result


def run_abstract_with_domain(
    cfg: CFG,
    variables: List[str],
    hidden_dim: int = 16,
    max_iters: int = 20,
    eps: float = 1e-3,
    beta: float = 0.5,
    domain: Optional[NeuralDomain] = None,
) -> Tuple[AbstractResult, NeuralDomain, Dict[str, Dict[int, torch.Tensor]]]:
    if domain is None:
        domain = create_domain(variables=variables, hidden_dim=hidden_dim)
    results: Dict[str, Dict[int, List[float]]] = {}
    tensor_states: Dict[str, Dict[int, torch.Tensor]] = {}
    with torch.no_grad():
        for func_name, func_cfg in cfg.functions.items():
            states = _analyze_function(
                func_cfg=func_cfg,
                domain=domain,
                max_iters=max_iters,
                eps=eps,
                beta=beta,
            )
            tensor_states[func_name] = states
            results[func_name] = {
                node_id: state.tolist() for node_id, state in states.items()
            }
    return AbstractResult(states=results), domain, tensor_states


def _analyze_function(
    func_cfg: FunctionCFG,
    domain: NeuralDomain,
    max_iters: int,
    eps: float,
    beta: float,
) -> Dict[int, torch.Tensor]:
    node_ids = list(func_cfg.nodes.keys())
    incoming = _incoming_edges(func_cfg)
    edge_states: Dict[Tuple[int, int], torch.Tensor] = {}
    node_states = {node_id: torch.zeros(domain.hidden_dim) for node_id in node_ids}

    for _ in range(max_iters):
        max_delta = 0.0
        for node_id in node_ids:
            preds = incoming.get(node_id, [])
            if preds:
                pred_states = [
                    edge_states.get(
                        (edge.source, edge.target), torch.zeros(domain.hidden_dim)
                    )
                    for edge in preds
                ]
                new_in = _join_all(domain, pred_states)
            else:
                new_in = domain.init_state()

            node = func_cfg.nodes[node_id]
            if node.kind == "loop_cond":
                new_in = (1.0 - beta) * node_states[node_id] + beta * new_in

            delta = torch.norm(new_in - node_states[node_id]).item()
            max_delta = max(max_delta, delta)
            node_states[node_id] = new_in

            _update_out_edges(func_cfg, domain, node, new_in, edge_states)

        if max_delta < eps:
            break

    return node_states


def _incoming_edges(func_cfg: FunctionCFG) -> Dict[int, List[CFGEdge]]:
    incoming: Dict[int, List[CFGEdge]] = {node_id: [] for node_id in func_cfg.nodes}
    for edges in func_cfg.edges.values():
        for edge in edges:
            incoming[edge.target].append(edge)
    return incoming


def _join_all(domain: NeuralDomain, states: List[torch.Tensor]) -> torch.Tensor:
    if not states:
        return torch.zeros(domain.hidden_dim)
    merged = states[0]
    for state in states[1:]:
        merged = domain.join(merged, state)
    return merged


def _update_out_edges(
    func_cfg: FunctionCFG,
    domain: NeuralDomain,
    node: CFGNode,
    in_state: torch.Tensor,
    edge_states: Dict[Tuple[int, int], torch.Tensor],
) -> None:
    edges = func_cfg.edges.get(node.id, [])
    if not edges:
        return
    if node.kind in {"branch", "loop_cond"}:
        if node.expr is None:
            raise ValueError("Missing condition expression.")
        expr_vec = _encode_expr(node.expr)
        prob = domain.predicate(in_state, expr_vec)
        for edge in edges:
            if edge.label == "true":
                weight = prob
            elif edge.label == "false":
                weight = 1.0 - prob
            else:
                weight = torch.tensor(1.0)
            edge_states[(edge.source, edge.target)] = in_state * weight
        return

    out_state = _transfer_state(domain, node, in_state)
    for edge in edges:
        edge_states[(edge.source, edge.target)] = out_state


def _transfer_state(
    domain: NeuralDomain, node: CFGNode, in_state: torch.Tensor
) -> torch.Tensor:
    if node.kind in {"stmt", "return"}:
        stmt_vec = _encode_stmt(node.stmt)
        return domain.transfer(in_state, stmt_vec)
    return in_state


def _encode_stmt(stmt) -> torch.Tensor:
    vec = torch.zeros(len(STMT_KINDS))
    kind = "noop"
    if isinstance(stmt, Assign):
        kind = "assign"
    elif isinstance(stmt, Call):
        kind = "call"
    elif isinstance(stmt, Return):
        kind = "return"
    index = STMT_KINDS.index(kind)
    vec[index] = 1.0
    return vec


def _encode_expr(expr: Expr) -> torch.Tensor:
    vec = torch.zeros(len(EXPR_KINDS) + len(EXPR_OPS))
    kind_index = EXPR_KINDS.index(_expr_kind(expr))
    vec[kind_index] = 1.0
    op = _expr_op(expr)
    if op in EXPR_OPS:
        op_index = EXPR_OPS.index(op)
        vec[len(EXPR_KINDS) + op_index] = 1.0
    return vec


def _expr_kind(expr: Expr) -> str:
    if isinstance(expr, IntLiteral):
        return "int"
    if isinstance(expr, BoolLiteral):
        return "bool"
    if isinstance(expr, Var):
        return "var"
    if isinstance(expr, BinOp):
        return "binop"
    if isinstance(expr, UnaryOp):
        return "unary"
    return "var"


def _expr_op(expr: Expr) -> str:
    if isinstance(expr, BinOp):
        return expr.op
    if isinstance(expr, UnaryOp):
        return "!" if expr.op == "!" else "neg"
    return ""


def encode_stmt(stmt) -> torch.Tensor:
    return _encode_stmt(stmt)


def encode_expr(expr: Expr) -> torch.Tensor:
    return _encode_expr(expr)

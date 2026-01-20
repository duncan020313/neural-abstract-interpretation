from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

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
from .cfg import CFG, CFGNode, FunctionCFG

Value = Union[int, bool]


@dataclass(frozen=True)
class Frame:
    function: str
    env: Dict[str, Value]
    return_pc: int


@dataclass(frozen=True)
class ConcreteState:
    function: str
    pc: Optional[int]
    env: Dict[str, Value]
    stack: List[Frame]


@dataclass(frozen=True)
class TraceEvent:
    function: str
    pc: int
    env: Dict[str, Value]


@dataclass(frozen=True)
class ConcreteResult:
    trace: List[TraceEvent]
    steps: int
    halted: bool


def eval_expr(expr: Expr, env: Dict[str, Value]) -> Value:
    if isinstance(expr, IntLiteral):
        return expr.value
    if isinstance(expr, BoolLiteral):
        return expr.value
    if isinstance(expr, Var):
        if expr.name not in env:
            raise ValueError(f"Unbound variable: {expr.name}")
        return env[expr.name]
    if isinstance(expr, UnaryOp):
        value = eval_expr(expr.operand, env)
        if expr.op == "!":
            return not bool(value)
        if expr.op == "-":
            return -int(value)
        raise ValueError(f"Unsupported unary op: {expr.op}")
    if isinstance(expr, BinOp):
        left = eval_expr(expr.left, env)
        right = eval_expr(expr.right, env)
        if expr.op == "+":
            return int(left) + int(right)
        if expr.op == "-":
            return int(left) - int(right)
        if expr.op == "*":
            return int(left) * int(right)
        if expr.op == "==":
            return left == right
        if expr.op == "<":
            return int(left) < int(right)
        if expr.op == "<=":
            return int(left) <= int(right)
        if expr.op == "&&":
            return bool(left) and bool(right)
        if expr.op == "||":
            return bool(left) or bool(right)
        raise ValueError(f"Unsupported binary op: {expr.op}")
    raise TypeError(f"Unsupported expr: {type(expr).__name__}")


def run_concrete(
    cfg: CFG,
    program_entry: str = "main",
    max_steps: int = 10_000,
    entry_args: Optional[Dict[str, Value]] = None,
) -> ConcreteResult:
    if program_entry not in cfg.functions:
        raise ValueError(f"Missing entry function: {program_entry}")
    entry_cfg = cfg.functions[program_entry]
    env = _init_entry_env(entry_cfg, entry_args)
    state = ConcreteState(
        function=program_entry,
        pc=entry_cfg.entry,
        env=env,
        stack=[],
    )
    trace: List[TraceEvent] = []
    steps = 0
    while state.pc is not None and steps < max_steps:
        func_cfg = cfg.functions[state.function]
        node = func_cfg.nodes[state.pc]
        trace.append(
            TraceEvent(function=state.function, pc=state.pc, env=dict(state.env))
        )
        state = _step(cfg, state, node)
        steps += 1
    return ConcreteResult(trace=trace, steps=steps, halted=state.pc is None)


def _step(cfg: CFG, state: ConcreteState, node: CFGNode) -> ConcreteState:
    func_cfg = cfg.functions[state.function]
    if node.kind in {"entry", "exit", "merge", "noop"}:
        next_pc = _select_edge(func_cfg, node.id, None)
        return ConcreteState(
            function=state.function,
            pc=next_pc,
            env=state.env,
            stack=state.stack,
        )

    if node.kind in {"branch", "loop_cond"}:
        if node.expr is None:
            raise ValueError("Missing condition expression.")
        cond_val = eval_expr(node.expr, state.env)
        label = "true" if bool(cond_val) else "false"
        next_pc = _select_edge(func_cfg, node.id, label)
        return ConcreteState(
            function=state.function,
            pc=next_pc,
            env=state.env,
            stack=state.stack,
        )

    if node.kind == "stmt":
        if isinstance(node.stmt, Assign):
            new_env = dict(state.env)
            new_env[node.stmt.name] = eval_expr(node.stmt.expr, state.env)
            next_pc = _select_edge(func_cfg, node.id, None)
            return ConcreteState(
                function=state.function,
                pc=next_pc,
                env=new_env,
                stack=state.stack,
            )
        if isinstance(node.stmt, Call):
            return _step_call(cfg, state, node.stmt)
        raise TypeError(f"Unsupported stmt: {type(node.stmt).__name__}")

    if node.kind == "return":
        if not isinstance(node.stmt, Return):
            raise TypeError("Return node missing return statement.")
        return _step_return(cfg, state, node.stmt)

    raise ValueError(f"Unsupported node kind: {node.kind}")


def _step_call(cfg: CFG, state: ConcreteState, call: Call) -> ConcreteState:
    if call.name not in cfg.functions:
        raise ValueError(f"Unknown function: {call.name}")
    if state.pc is None:
        raise ValueError("Missing program counter for call.")
    caller_cfg = cfg.functions[state.function]
    return_pc = _select_edge(caller_cfg, state.pc, None)
    if return_pc is None:
        raise ValueError("Call has no fall-through edge.")
    args = [eval_expr(arg, state.env) for arg in call.args]
    callee_cfg = cfg.functions[call.name]
    if len(args) != len(callee_cfg.params):
        raise ValueError(
            f"Arity mismatch for {call.name}: expected {len(callee_cfg.params)} got {len(args)}"
        )
    callee_env = {}
    for name, value in zip(callee_cfg.params, args):
        callee_env[name] = value
    new_stack = list(state.stack)
    new_stack.append(Frame(function=state.function, env=state.env, return_pc=return_pc))
    return ConcreteState(
        function=call.name,
        pc=callee_cfg.entry,
        env=callee_env,
        stack=new_stack,
    )


def _step_return(cfg: CFG, state: ConcreteState, stmt: Return) -> ConcreteState:
    _ = eval_expr(stmt.expr, state.env)
    if not state.stack:
        return ConcreteState(
            function=state.function,
            pc=None,
            env=state.env,
            stack=state.stack,
        )
    new_stack = list(state.stack)
    frame = new_stack.pop()
    return ConcreteState(
        function=frame.function,
        pc=frame.return_pc,
        env=frame.env,
        stack=new_stack,
    )


def _init_entry_env(
    entry_cfg: FunctionCFG, entry_args: Optional[Dict[str, Value]]
) -> Dict[str, Value]:
    if not entry_cfg.params:
        return {}
    if entry_args is None:
        raise ValueError("Entry arguments required for main.")
    missing = [name for name in entry_cfg.params if name not in entry_args]
    if missing:
        raise ValueError(f"Missing entry arguments: {', '.join(missing)}")
    extra = [name for name in entry_args if name not in entry_cfg.params]
    if extra:
        raise ValueError(f"Unknown entry arguments: {', '.join(extra)}")
    return {name: entry_args[name] for name in entry_cfg.params}


def _select_edge(
    func_cfg: FunctionCFG, node_id: int, label: Optional[str]
) -> Optional[int]:
    edges = func_cfg.edges.get(node_id, [])
    if label is None:
        if not edges:
            return None
        if len(edges) == 1:
            return edges[0].target
        for edge in edges:
            if edge.label == "next":
                return edge.target
        return edges[0].target
    for edge in edges:
        if edge.label == label:
            return edge.target
    return None

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set


@dataclass(frozen=True)
class SourceSpan:
    line: int
    column: int
    end_line: int
    end_column: int


class Expr:
    pass


@dataclass(frozen=True)
class IntLiteral(Expr):
    value: int
    span: Optional[SourceSpan] = None


@dataclass(frozen=True)
class BoolLiteral(Expr):
    value: bool
    span: Optional[SourceSpan] = None


@dataclass(frozen=True)
class Var(Expr):
    name: str
    span: Optional[SourceSpan] = None


@dataclass(frozen=True)
class BinOp(Expr):
    op: str
    left: Expr
    right: Expr
    span: Optional[SourceSpan] = None


@dataclass(frozen=True)
class UnaryOp(Expr):
    op: str
    operand: Expr
    span: Optional[SourceSpan] = None


class Stmt:
    pass


@dataclass(frozen=True)
class Assign(Stmt):
    name: str
    expr: Expr
    span: Optional[SourceSpan] = None


@dataclass(frozen=True)
class Call(Stmt):
    name: str
    args: List[Expr]
    span: Optional[SourceSpan] = None


@dataclass(frozen=True)
class Return(Stmt):
    expr: Expr
    span: Optional[SourceSpan] = None


@dataclass(frozen=True)
class If(Stmt):
    condition: Expr
    then_body: List[Stmt]
    else_body: List[Stmt]
    span: Optional[SourceSpan] = None


@dataclass(frozen=True)
class While(Stmt):
    condition: Expr
    body: List[Stmt]
    span: Optional[SourceSpan] = None


@dataclass(frozen=True)
class Function:
    name: str
    params: List[str]
    body: List[Stmt]


@dataclass(frozen=True)
class Program:
    functions: Dict[str, Function]


def expr_to_str(expr: Expr) -> str:
    if isinstance(expr, IntLiteral):
        return str(expr.value)
    if isinstance(expr, BoolLiteral):
        return "true" if expr.value else "false"
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, UnaryOp):
        return f"({expr.op}{expr_to_str(expr.operand)})"
    if isinstance(expr, BinOp):
        left = expr_to_str(expr.left)
        right = expr_to_str(expr.right)
        return f"({left} {expr.op} {right})"
    raise TypeError(f"Unsupported expr: {type(expr).__name__}")


def stmt_to_str(stmt: Optional[Stmt]) -> Optional[str]:
    if stmt is None:
        return None
    if isinstance(stmt, Assign):
        return f"{stmt.name} = {expr_to_str(stmt.expr)};"
    if isinstance(stmt, Call):
        args = ", ".join(expr_to_str(arg) for arg in stmt.args)
        return f"{stmt.name}({args});"
    if isinstance(stmt, Return):
        return f"return {expr_to_str(stmt.expr)};"
    if isinstance(stmt, If):
        return f"if ({expr_to_str(stmt.condition)}) {{ ... }} else {{ ... }}"
    if isinstance(stmt, While):
        return f"while ({expr_to_str(stmt.condition)}) {{ ... }}"
    raise TypeError(f"Unsupported stmt: {type(stmt).__name__}")


def collect_variables(program: Program) -> List[str]:
    seen: Set[str] = set()
    for func in program.functions.values():
        seen.update(func.params)
        for stmt in func.body:
            _collect_vars_stmt(stmt, seen)
    return sorted(seen)


def _collect_vars_stmt(stmt: Stmt, seen: Set[str]) -> None:
    if isinstance(stmt, Assign):
        seen.add(stmt.name)
        _collect_vars_expr(stmt.expr, seen)
        return
    if isinstance(stmt, Call):
        for arg in stmt.args:
            _collect_vars_expr(arg, seen)
        return
    if isinstance(stmt, Return):
        _collect_vars_expr(stmt.expr, seen)
        return
    if isinstance(stmt, If):
        _collect_vars_expr(stmt.condition, seen)
        for inner in stmt.then_body:
            _collect_vars_stmt(inner, seen)
        for inner in stmt.else_body:
            _collect_vars_stmt(inner, seen)
        return
    if isinstance(stmt, While):
        _collect_vars_expr(stmt.condition, seen)
        for inner in stmt.body:
            _collect_vars_stmt(inner, seen)
        return
    raise TypeError(f"Unsupported stmt: {type(stmt).__name__}")


def _collect_vars_expr(expr: Expr, seen: Set[str]) -> None:
    if isinstance(expr, Var):
        seen.add(expr.name)
        return
    if isinstance(expr, (IntLiteral, BoolLiteral)):
        return
    if isinstance(expr, UnaryOp):
        _collect_vars_expr(expr.operand, seen)
        return
    if isinstance(expr, BinOp):
        _collect_vars_expr(expr.left, seen)
        _collect_vars_expr(expr.right, seen)
        return
    raise TypeError(f"Unsupported expr: {type(expr).__name__}")

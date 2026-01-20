from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .ast import Assign, Call, Expr, Function, If, Program, Return, Stmt, While


@dataclass(frozen=True)
class CFGNode:
    id: int
    kind: str
    function: str
    stmt: Optional[Stmt] = None
    expr: Optional[Expr] = None


@dataclass(frozen=True)
class CFGEdge:
    source: int
    target: int
    label: str


@dataclass
class FunctionCFG:
    name: str
    entry: int
    exit: int
    nodes: Dict[int, CFGNode]
    edges: Dict[int, List[CFGEdge]]
    params: List[str] = field(default_factory=list)


@dataclass
class CFG:
    functions: Dict[str, FunctionCFG]


class FunctionCFGBuilder:
    def __init__(self, function: Function) -> None:
        self.function = function
        self.nodes: Dict[int, CFGNode] = {}
        self.edges: Dict[int, List[CFGEdge]] = {}
        self._next_id = 0
        self._return_nodes: List[int] = []

    def build(self) -> FunctionCFG:
        entry_id = self._new_node(kind="entry")
        exit_id = self._new_node(kind="exit")
        block_entry, block_exits = self._build_block(self.function.body)
        self._add_edge(entry_id, block_entry, "next")
        for exit_node in block_exits:
            self._add_edge(exit_node, exit_id, "next")
        for return_node in self._return_nodes:
            self._add_edge(return_node, exit_id, "return")
        return FunctionCFG(
            name=self.function.name,
            entry=entry_id,
            exit=exit_id,
            nodes=self.nodes,
            edges=self.edges,
            params=list(self.function.params),
        )

    def _new_node(
        self, kind: str, stmt: Optional[Stmt] = None, expr: Optional[Expr] = None
    ) -> int:
        node_id = self._next_id
        self._next_id += 1
        self.nodes[node_id] = CFGNode(
            id=node_id, kind=kind, function=self.function.name, stmt=stmt, expr=expr
        )
        self.edges[node_id] = []
        return node_id

    def _add_edge(self, source: int, target: int, label: str) -> None:
        self.edges[source].append(CFGEdge(source=source, target=target, label=label))

    def _build_block(self, stmts: List[Stmt]) -> Tuple[int, List[int]]:
        entry: Optional[int] = None
        exits: List[int] = []
        for stmt in stmts:
            stmt_entry, stmt_exits = self._build_stmt(stmt)
            if entry is None:
                entry = stmt_entry
            for exit_node in exits:
                self._add_edge(exit_node, stmt_entry, "next")
            exits = stmt_exits
            if not exits:
                break
        if entry is None:
            entry = self._new_node(kind="noop")
            exits = [entry]
        return entry, exits

    def _build_stmt(self, stmt: Stmt) -> Tuple[int, List[int]]:
        if isinstance(stmt, (Assign, Call)):
            node_id = self._new_node(kind="stmt", stmt=stmt)
            return node_id, [node_id]
        if isinstance(stmt, Return):
            node_id = self._new_node(kind="return", stmt=stmt)
            self._return_nodes.append(node_id)
            return node_id, []
        if isinstance(stmt, If):
            cond_id = self._new_node(kind="branch", expr=stmt.condition)
            then_entry, then_exits = self._build_block(stmt.then_body)
            else_entry, else_exits = self._build_block(stmt.else_body)
            self._add_edge(cond_id, then_entry, "true")
            self._add_edge(cond_id, else_entry, "false")
            if then_exits or else_exits:
                merge_id = self._new_node(kind="merge")
                for exit_node in then_exits:
                    self._add_edge(exit_node, merge_id, "next")
                for exit_node in else_exits:
                    self._add_edge(exit_node, merge_id, "next")
                return cond_id, [merge_id]
            return cond_id, []
        if isinstance(stmt, While):
            cond_id = self._new_node(kind="loop_cond", expr=stmt.condition)
            body_entry, body_exits = self._build_block(stmt.body)
            loop_exit = self._new_node(kind="merge")
            self._add_edge(cond_id, body_entry, "true")
            self._add_edge(cond_id, loop_exit, "false")
            for exit_node in body_exits:
                self._add_edge(exit_node, cond_id, "next")
            return cond_id, [loop_exit]
        raise TypeError(f"Unsupported stmt: {type(stmt).__name__}")


def build_cfg(program: Program) -> CFG:
    functions: Dict[str, FunctionCFG] = {}
    for func in program.functions.values():
        functions[func.name] = FunctionCFGBuilder(func).build()
    return CFG(functions=functions)

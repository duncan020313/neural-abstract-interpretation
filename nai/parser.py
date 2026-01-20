from __future__ import annotations

from typing import List

from lark import Lark, Transformer

from .ast import (
    Assign,
    BinOp,
    BoolLiteral,
    Call,
    Function,
    If,
    IntLiteral,
    Program,
    Return,
    UnaryOp,
    Var,
    While,
)

GRAMMAR = r"""
    ?start: program
    program: function+

    function: CNAME "(" [params] ")" block
    params: CNAME ("," CNAME)*

    stmt: assign ";"                -> assign_stmt
        | "if" "(" expr ")" block "else" block                  -> if_stmt
        | "while" "(" expr ")" block                            -> while_stmt
        | "return" expr ";"                                     -> return_stmt
        | call ";"                                              -> call_stmt

    assign: CNAME "=" expr
    call: CNAME "(" [args] ")"
    args: expr ("," expr)*

    ?expr: or_expr
    ?or_expr: and_expr (OR_OP and_expr)*
    ?and_expr: equality_expr (AND_OP equality_expr)*
    ?equality_expr: relational_expr (EQ_OP relational_expr)*
    ?relational_expr: add_expr ((LE_OP | LT_OP) add_expr)*
    ?add_expr: mul_expr ((ADD_OP | SUB_OP) mul_expr)*
    ?mul_expr: unary_expr (MUL_OP unary_expr)*
    ?unary_expr: NOT_OP unary_expr    -> not_expr
        | SUB_OP unary_expr           -> neg_expr
        | atom
    ?atom: INT                        -> int
        | BOOL                        -> bool
        | CNAME                       -> var
        | "(" expr ")"

    block: "{" stmt* "}"

    OR_OP: "||"
    AND_OP: "&&"
    EQ_OP: "=="
    LE_OP: "<="
    LT_OP: "<"
    ADD_OP: "+"
    SUB_OP: "-"
    MUL_OP: "*"
    NOT_OP: "!"

    BOOL: "true" | "false"
    INT: /[0-9]+/

    %import common.CNAME
    %import common.WS
    %ignore WS
    %ignore /\/\/[^\n]*/
"""


class ASTBuilder(Transformer):
    def program(self, items: List[Function]) -> Program:
        functions = {}
        for func in items:
            if func.name in functions:
                raise ValueError(f"Duplicate function: {func.name}")
            functions[func.name] = func
        return Program(functions=functions)

    def function(self, items: List) -> Function:
        name = str(items[0])
        params = []
        body_index = 1
        if len(items) > 1 and isinstance(items[1], list):
            params = [str(param) for param in items[1]]
            body_index = 2
        body = items[body_index]
        return Function(name=name, params=params, body=body)

    def params(self, items: List) -> List[str]:
        return [str(item) for item in items]

    def assign_stmt(self, items: List) -> Assign:
        return items[0]

    def assign(self, items: List) -> Assign:
        name = str(items[0])
        return Assign(name=name, expr=items[1])

    def call_stmt(self, items: List) -> Call:
        return items[0]

    def call(self, items: List) -> Call:
        name = str(items[0])
        args = []
        if len(items) > 1:
            args = items[1]
        return Call(name=name, args=args)

    def args(self, items: List) -> List:
        return list(items)

    def if_stmt(self, items: List) -> If:
        condition = items[0]
        then_body = items[1]
        else_body = items[2]
        return If(condition=condition, then_body=then_body, else_body=else_body)

    def while_stmt(self, items: List) -> While:
        condition = items[0]
        body = items[1]
        return While(condition=condition, body=body)

    def return_stmt(self, items: List) -> Return:
        return Return(expr=items[0])

    def int(self, items: List) -> IntLiteral:
        return IntLiteral(value=int(items[0]))

    def bool(self, items: List) -> BoolLiteral:
        value = str(items[0]) == "true"
        return BoolLiteral(value=value)

    def var(self, items: List) -> Var:
        return Var(name=str(items[0]))

    def not_expr(self, items: List) -> UnaryOp:
        return UnaryOp(op="!", operand=items[-1])

    def neg_expr(self, items: List) -> UnaryOp:
        return UnaryOp(op="-", operand=items[-1])

    def or_expr(self, items: List):
        return _fold_binops(items)

    def and_expr(self, items: List):
        return _fold_binops(items)

    def equality_expr(self, items: List):
        return _fold_binops(items)

    def relational_expr(self, items: List):
        return _fold_binops(items)

    def add_expr(self, items: List):
        return _fold_binops(items)

    def mul_expr(self, items: List):
        return _fold_binops(items)

    def block(self, items: List) -> List:
        return list(items)


def _fold_binops(items: List):
    if len(items) == 1:
        return items[0]
    expr = items[0]
    idx = 1
    while idx < len(items):
        op = str(items[idx])
        right = items[idx + 1]
        expr = BinOp(op=op, left=expr, right=right)
        idx += 2
    return expr


def parse_program(source: str) -> Program:
    parser = Lark(GRAMMAR, parser="lalr", start="program")
    tree = parser.parse(source)
    return ASTBuilder().transform(tree)

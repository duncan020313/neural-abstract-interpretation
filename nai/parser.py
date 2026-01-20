from __future__ import annotations

from typing import List, Optional

from lark import Lark, Transformer, v_args

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
    SourceSpan,
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

    assign: CNAME "=" expr          -> assign
        | CNAME DIV_ASSIGN expr     -> div_assign
        | CNAME MOD_ASSIGN expr     -> mod_assign
    call: CNAME "(" [args] ")"
    args: expr ("," expr)*

    ?expr: or_expr
    ?or_expr: and_expr (OR_OP and_expr)*
    ?and_expr: equality_expr (AND_OP equality_expr)*
    ?equality_expr: relational_expr (EQ_OP relational_expr)*
    ?relational_expr: add_expr ((LE_OP | LT_OP) add_expr)*
    ?add_expr: mul_expr ((ADD_OP | SUB_OP) mul_expr)*
    ?mul_expr: unary_expr ((MUL_OP | DIV_OP | MOD_OP) unary_expr)*
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
    DIV_ASSIGN: "/="
    MOD_ASSIGN: "%="
    MUL_OP: "*"
    DIV_OP: "/"
    MOD_OP: "%"
    NOT_OP: "!"

    BOOL: "true" | "false"
    INT: /[0-9]+/

    %import common.CNAME
    %import common.WS
    %ignore WS
    %ignore /\/\/[^\n]*/
"""


@v_args(meta=True)
class ASTBuilder(Transformer):
    def program(self, meta, items: List[Function]) -> Program:
        functions = {}
        for func in items:
            if func.name in functions:
                raise ValueError(f"Duplicate function: {func.name}")
            functions[func.name] = func
        return Program(functions=functions)

    def function(self, meta, items: List) -> Function:
        name = str(items[0])
        params = []
        body_index = 1
        if len(items) > 1 and isinstance(items[1], list):
            params = [str(param) for param in items[1]]
            body_index = 2
        body = items[body_index]
        return Function(name=name, params=params, body=body)

    def params(self, meta, items: List) -> List[str]:
        return [str(item) for item in items]

    def assign_stmt(self, meta, items: List) -> Assign:
        return items[0]

    def assign(self, meta, items: List) -> Assign:
        name = str(items[0])
        return Assign(name=name, expr=items[1], span=_span(meta))

    def div_assign(self, meta, items: List) -> Assign:
        name = str(items[0])
        left = Var(name=name, span=_span(meta))
        expr = BinOp(op="/", left=left, right=items[-1], span=_span(meta))
        return Assign(name=name, expr=expr, span=_span(meta))

    def mod_assign(self, meta, items: List) -> Assign:
        name = str(items[0])
        left = Var(name=name, span=_span(meta))
        expr = BinOp(op="%", left=left, right=items[-1], span=_span(meta))
        return Assign(name=name, expr=expr, span=_span(meta))

    def call_stmt(self, meta, items: List) -> Call:
        return items[0]

    def call(self, meta, items: List) -> Call:
        name = str(items[0])
        args = []
        if len(items) > 1:
            args = items[1]
        return Call(name=name, args=args, span=_span(meta))

    def args(self, meta, items: List) -> List:
        return list(items)

    def if_stmt(self, meta, items: List) -> If:
        condition = items[0]
        then_body = items[1]
        else_body = items[2]
        return If(
            condition=condition,
            then_body=then_body,
            else_body=else_body,
            span=_span(meta),
        )

    def while_stmt(self, meta, items: List) -> While:
        condition = items[0]
        body = items[1]
        return While(condition=condition, body=body, span=_span(meta))

    def return_stmt(self, meta, items: List) -> Return:
        return Return(expr=items[0], span=_span(meta))

    def int(self, meta, items: List) -> IntLiteral:
        return IntLiteral(value=int(items[0]), span=_span(meta))

    def bool(self, meta, items: List) -> BoolLiteral:
        value = str(items[0]) == "true"
        return BoolLiteral(value=value, span=_span(meta))

    def var(self, meta, items: List) -> Var:
        return Var(name=str(items[0]), span=_span(meta))

    def not_expr(self, meta, items: List) -> UnaryOp:
        return UnaryOp(op="!", operand=items[-1], span=_span(meta))

    def neg_expr(self, meta, items: List) -> UnaryOp:
        return UnaryOp(op="-", operand=items[-1], span=_span(meta))

    def or_expr(self, meta, items: List):
        return _fold_binops(items, _span(meta))

    def and_expr(self, meta, items: List):
        return _fold_binops(items, _span(meta))

    def equality_expr(self, meta, items: List):
        return _fold_binops(items, _span(meta))

    def relational_expr(self, meta, items: List):
        return _fold_binops(items, _span(meta))

    def add_expr(self, meta, items: List):
        return _fold_binops(items, _span(meta))

    def mul_expr(self, meta, items: List):
        return _fold_binops(items, _span(meta))

    def block(self, meta, items: List) -> List:
        return list(items)


def _fold_binops(items: List, span: Optional[SourceSpan]):
    if len(items) == 1:
        return items[0]
    expr = items[0]
    idx = 1
    while idx < len(items):
        op = str(items[idx])
        right = items[idx + 1]
        expr = BinOp(op=op, left=expr, right=right, span=span)
        idx += 2
    return expr


def parse_program(source: str) -> Program:
    parser = Lark(GRAMMAR, parser="lalr", start="program", propagate_positions=True)
    tree = parser.parse(source)
    return ASTBuilder().transform(tree)


def _span(meta) -> Optional[SourceSpan]:
    if meta is None:
        return None
    return SourceSpan(
        line=meta.line,
        column=meta.column,
        end_line=meta.end_line,
        end_column=meta.end_column,
    )

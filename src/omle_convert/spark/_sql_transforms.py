"""Convert Spark SQLTransformer statements to OMLE Derive nodes.

Uses ``sqlglot`` (pure-Python, zero-JVM dependency) to parse the SQL statement
stored in the ``SQLTransformer`` saved metadata.  Each computed column becomes a
``Derive`` node with an embedded ``Expression`` tree; ``SELECT *`` columns pass
through unchanged via the existing ``col_to_tensor`` mapping.

Supported SQL:
  arithmetic  : +  -  *  /  %  unary -
  comparison  : =  !=  <>  <  <=  >  >=
  boolean     : AND  OR  NOT
  math        : ABS SQRT EXP LN LOG LOG2 LOG10 FLOOR CEIL CEILING ROUND SIGN POW POWER
  string      : CONCAT LOWER UPPER TRIM SUBSTRING SUBSTR
  conditional : IF  CASE WHEN ... THEN ... [ELSE ...] END  COALESCE
  null checks : IS NULL  IS NOT NULL
  type cast   : CAST(x AS type) → pass-through (type widening handled at runtime)

Not supported (raises NotImplementedError):
  aggregates (SUM/COUNT/AVG with GROUP BY), window functions, JOINs, subqueries,
  REGEXP_* functions, array/map functions, custom-base LOG(base, x).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import omle
from omle.ir.expression import Expression
from omle.ir.types import Scalar

if TYPE_CHECKING:
    from ._builder import Builder

try:
    import sqlglot
    import sqlglot.expressions as sge
    _SQLGLOT_AVAILABLE = True
except ImportError:
    sqlglot = None  # type: ignore[assignment]
    sge = None  # type: ignore[assignment]
    _SQLGLOT_AVAILABLE = False


# ── sqlglot binary op type → OMLE function name ────────────────────────────

_BINARY_OP_MAP: dict = {}
_UNARY_FN_MAP:  dict = {}
_MAPS_INITIALIZED = False


def _init_maps() -> None:
    global _BINARY_OP_MAP, _UNARY_FN_MAP, _MAPS_INITIALIZED
    if _MAPS_INITIALIZED:
        return
    _BINARY_OP_MAP = {
        sge.Add: "omle.functions.add",
        sge.Sub: "omle.functions.sub",
        sge.Mul: "omle.functions.mul",
        sge.Div: "omle.functions.div",
        sge.Mod: "omle.functions.mod",
        sge.Pow: "omle.functions.pow",
        sge.EQ:  "omle.functions.equal",
        sge.NEQ: "omle.functions.not_equal",
        sge.LT:  "omle.functions.less_than",
        sge.LTE: "omle.functions.less_or_equal",
        sge.GT:  "omle.functions.greater_than",
        sge.GTE: "omle.functions.greater_or_equal",
        sge.And: "omle.functions.and",
        sge.Or:  "omle.functions.or",
    }
    _UNARY_FN_MAP = {
        sge.Abs:   "omle.functions.abs",
        sge.Sqrt:  "omle.functions.sqrt",
        sge.Exp:   "omle.functions.exp",
        sge.Ln:    "omle.functions.log",
        sge.Floor: "omle.functions.floor",
        sge.Ceil:  "omle.functions.ceil",
        sge.Round: "omle.functions.round",
        sge.Sign:  "omle.functions.sign",
        sge.Lower: "omle.functions.lower",
        sge.Upper: "omle.functions.upper",
        sge.Trim:  "omle.functions.trim",
        sge.Neg:   "omle.functions.neg",
    }
    _MAPS_INITIALIZED = True


# ── Expression-type inference ──────────────────────────────────────────────────

_STRING_NODE_TYPES: set = set()  # sqlglot types that produce string output


def _init_string_types() -> None:
    global _STRING_NODE_TYPES
    if _STRING_NODE_TYPES:
        return
    _STRING_NODE_TYPES = {sge.Lower, sge.Upper, sge.Trim, sge.Concat, sge.Substring}


def _is_string_producing(node) -> bool:
    """Return True when a SQL expression node always produces a string value."""
    _init_string_types()
    if isinstance(node, (sge.Paren, sge.Cast, sge.TryCast)):
        return _is_string_producing(node.this)
    if type(node) in _STRING_NODE_TYPES:
        return True
    if isinstance(node, sge.Anonymous):
        return (node.name or "").upper() in ("LOWER", "UPPER", "TRIM", "CONCAT", "SUBSTRING", "SUBSTR")
    return False


# ── Anonymous SQL function name → OMLE function name ───────────────────────

_ANON_FN_MAP: dict[str, str] = {
    "ABS":       "omle.functions.abs",
    "SQRT":      "omle.functions.sqrt",
    "EXP":       "omle.functions.exp",
    "FLOOR":     "omle.functions.floor",
    "CEIL":      "omle.functions.ceil",
    "CEILING":   "omle.functions.ceil",
    "ROUND":     "omle.functions.round",
    "SIGN":      "omle.functions.sign",
    "CONCAT":    "omle.functions.concat",
    "LOWER":     "omle.functions.lower",
    "UPPER":     "omle.functions.upper",
    "TRIM":      "omle.functions.trim",
    "SUBSTRING": "omle.functions.substring",
    "SUBSTR":    "omle.functions.substring",
    "IF":        "omle.functions.if",
    "COALESCE":  "omle.functions.coalesce",
    "LOG2":      "omle.functions.log2",
    "LOG10":     "omle.functions.log10",
    "POW":       "omle.functions.pow",
    "POWER":     "omle.functions.pow",
    "MOD":       "omle.functions.mod",
    "LN":        "omle.functions.log",
    "LOG":       "omle.functions.log",
}


# ── Expression builder ─────────────────────────────────────────────────────────

def _build_expr(node, col_to_tensor: dict) -> Expression:
    """Recursively convert a sqlglot expression node to an OMLE Expression."""
    _init_maps()

    # Transparent wrappers
    if isinstance(node, sge.Paren):
        return _build_expr(node.this, col_to_tensor)

    # Column reference
    if isinstance(node, sge.Column):
        col_name   = node.name
        tensor_name = col_to_tensor.get(col_name, col_name)
        return Expression.from_ref(tensor_name)

    # Literal constant
    if isinstance(node, sge.Literal):
        if node.is_number:
            raw = node.this
            try:
                return Expression.from_literal(Scalar.int(int(raw)))
            except (ValueError, TypeError):
                return Expression.from_literal(Scalar.float(float(raw)))
        return Expression.from_literal(Scalar.string(node.this))

    # CAST / TRY_CAST — pass through the inner expression; runtime widens types
    if isinstance(node, (sge.Cast, sge.TryCast)):
        return _build_expr(node.this, col_to_tensor)

    # Binary operators (all use .this and .expression)
    if type(node) in _BINARY_OP_MAP:
        fn    = _BINARY_OP_MAP[type(node)]
        left  = _build_expr(node.this,       col_to_tensor)
        right = _build_expr(node.expression, col_to_tensor)
        return Expression.from_apply(fn, left, right)

    # Unary operators / single-arg functions (all use .this)
    if type(node) in _UNARY_FN_MAP:
        fn    = _UNARY_FN_MAP[type(node)]
        inner = _build_expr(node.this, col_to_tensor)
        return Expression.from_apply(fn, inner)

    # LOG(base, value) → log2 / log10 if base is 2 / 10; otherwise unsupported
    if isinstance(node, sge.Log):
        base_node  = node.this
        value_node = node.expression
        if isinstance(base_node, sge.Literal) and base_node.is_number:
            base_int = int(float(base_node.this))
            if base_int == 2:
                return Expression.from_apply("omle.functions.log2", _build_expr(value_node, col_to_tensor))
            if base_int == 10:
                return Expression.from_apply("omle.functions.log10", _build_expr(value_node, col_to_tensor))
        raise NotImplementedError(
            f"LOG with custom base ({base_node.sql()!r}) is not supported in SQLTransformer. "
            "Use LOG2(x) or LOG10(x) for base-2/10, or LN(x) for the natural logarithm."
        )

    # NOT — check for IS NOT NULL special case, otherwise boolean NOT
    if isinstance(node, sge.Not):
        inner = node.this
        # Unwrap Paren if present
        if isinstance(inner, sge.Paren):
            inner = inner.this
        if isinstance(inner, sge.Is) and isinstance(inner.expression, sge.Null):
            return Expression.from_apply(
                "omle.functions.is_not_missing",
                _build_expr(inner.this, col_to_tensor),
            )
        return Expression.from_apply("omle.functions.not", _build_expr(node.this, col_to_tensor))

    # IS NULL
    if isinstance(node, sge.Is) and isinstance(node.expression, sge.Null):
        return Expression.from_apply("omle.functions.is_missing", _build_expr(node.this, col_to_tensor))

    # IF(cond, then, else)
    if isinstance(node, sge.If):
        cond  = _build_expr(node.this,             col_to_tensor)
        then  = _build_expr(node.args["true"],     col_to_tensor)
        else_ = _build_expr(node.args["false"],    col_to_tensor)
        return Expression.from_apply("omle.functions.if", cond, then, else_)

    # CASE WHEN ... THEN ... [WHEN ...] [ELSE ...] END → nested if chain
    if isinstance(node, sge.Case):
        branches = node.args.get("ifs", [])
        default  = node.args.get("default")
        if not branches:
            raise NotImplementedError("CASE with no WHEN branches cannot be converted")

        def _case_chain(idx: int) -> Expression:
            w    = branches[idx]
            cond = _build_expr(w.this,          col_to_tensor)
            then = _build_expr(w.args["true"],  col_to_tensor)
            if idx + 1 < len(branches):
                else_ = _case_chain(idx + 1)
            elif default is not None:
                else_ = _build_expr(default, col_to_tensor)
            else:
                else_ = Expression.from_literal(Scalar.float(float("nan")))
            return Expression.from_apply("omle.functions.if", cond, then, else_)

        return _case_chain(0)

    # COALESCE(a, b, ...) — .this is the first arg, .expressions is the rest
    if isinstance(node, sge.Coalesce):
        all_args = [node.this, *node.expressions]
        exprs    = [_build_expr(a, col_to_tensor) for a in all_args]
        return Expression.from_apply("omle.functions.coalesce", *exprs)

    # CONCAT(a, b, ...) — chain binary concat for >2 args
    if isinstance(node, sge.Concat):
        parts = node.expressions
        if not parts:
            raise NotImplementedError("CONCAT with no arguments")
        exprs  = [_build_expr(p, col_to_tensor) for p in parts]
        result = exprs[0]
        for e in exprs[1:]:
            result = Expression.from_apply("omle.functions.concat", result, e)
        return result

    # SUBSTRING(str, start, length)
    if isinstance(node, sge.Substring):
        s_expr   = _build_expr(node.this, col_to_tensor)
        start    = node.args.get("start")
        length   = node.args.get("length")
        pos_expr = _build_expr(start,  col_to_tensor) if start  is not None else Expression.from_literal(Scalar.int(1))
        if length is not None:
            len_expr = _build_expr(length, col_to_tensor)
            return Expression.from_apply("omle.functions.substring", s_expr, pos_expr, len_expr)
        return Expression.from_apply("omle.functions.substring", s_expr, pos_expr)

    # Anonymous functions (sqlglot falls back to this for any unknown SQL function)
    if isinstance(node, sge.Anonymous):
        fn_upper = (node.name or "").upper()
        mapped   = _ANON_FN_MAP.get(fn_upper)
        if mapped is None:
            raise NotImplementedError(
                f"Unsupported SQL function in SQLTransformer: {fn_upper!r}"
            )
        args = [_build_expr(a, col_to_tensor) for a in node.expressions]
        return Expression.from_apply(mapped, *args)

    raise NotImplementedError(
        f"Unsupported SQL expression type in SQLTransformer: {type(node).__name__!r} "
        f"({node.sql()!r}) — only row-level arithmetic, comparisons, math functions, "
        "conditionals, and string operations are supported"
    )


def _collect_column_refs(node) -> list[str]:
    """Collect Column names referenced by an expression, in order, without duplicates."""
    seen:   set[str]  = set()
    result: list[str] = []
    for sub in node.walk():
        if isinstance(sub, sge.Column):
            name = sub.name
            if name and name not in seen:
                seen.add(name)
                result.append(name)
    return result


# ── Public entry point ─────────────────────────────────────────────────────────

def apply_sql_transformer(
    stmt: str,
    col_to_tensor: dict,
    prefix: str,
    builder: "Builder",
) -> dict:
    """Parse a SQLTransformer statement and emit Derive nodes for computed columns.

    Parameters
    ----------
    stmt:
        The SQL statement from the transformer metadata, e.g.
        ``"SELECT *, age + income AS total FROM __THIS__"``.
    col_to_tensor:
        Current mapping from DataFrame column names to OMLE tensor names.
    prefix:
        Node name prefix (e.g. ``"s2_sqltran"``).
    builder:
        The graph builder for adding nodes.

    Returns
    -------
    dict
        Updated col_to_tensor mapping that reflects the SELECT output columns.
        Computed aliases are added; star-selected columns pass through unchanged.
    """
    if not _SQLGLOT_AVAILABLE:
        raise ImportError(
            "sqlglot is required for SQLTransformer support. "
            "Install it with: pip install sqlglot"
        )

    ast = sqlglot.parse_one(stmt, dialect="spark")
    if not isinstance(ast, sge.Select):
        raise ValueError(
            f"SQLTransformer statement must be a SELECT query, got: {type(ast).__name__!r}"
        )

    new_col_to_tensor = dict(col_to_tensor)
    output_cols: list[str] = []
    has_star = False

    for sel_expr in ast.expressions:
        if isinstance(sel_expr, sge.Star):
            has_star = True
            # Pass through all currently known columns
            for col_name in col_to_tensor:
                if col_name not in output_cols:
                    output_cols.append(col_name)
            continue

        if isinstance(sel_expr, sge.Alias):
            alias_name = sel_expr.alias
            inner = sel_expr.this

            if isinstance(inner, sge.Column):
                # Simple rename / passthrough
                src_tensor = col_to_tensor.get(inner.name, inner.name)
                new_col_to_tensor[alias_name] = src_tensor
                output_cols.append(alias_name)
            else:
                # Computed expression → Derive node
                ir_expr = _build_expr(inner, col_to_tensor)
                col_refs      = _collect_column_refs(inner)
                input_tensors = [col_to_tensor.get(c, c) for c in col_refs]
                node_name = builder.unique_name(f"{prefix}_{alias_name}")
                builder.require_namespace("omle.core")
                out_dtype = (
                    omle.DataType.STRING if _is_string_producing(inner)
                    else omle.DataType.FLOAT64
                )
                out_type = omle.TensorType(dtype=out_dtype, shape=[-1])
                builder.add_node(omle.Node(
                    name=node_name,
                    domain="omle.core",
                    op="Derive",
                    inputs=[omle.NodeInput(name=t) for t in input_tensors],
                    outputs=[omle.NodeOutput(
                        name=alias_name,
                        role=omle.OutputRole.TRANSFORMED_VALUE,
                        type=out_type,
                    )],
                    attributes=[omle.Attribute(name="expr", expr=ir_expr)],
                ))
                new_col_to_tensor[alias_name] = alias_name
                output_cols.append(alias_name)

        elif isinstance(sel_expr, sge.Column):
            # SELECT col (no alias) — pass through unchanged
            col_name = sel_expr.name
            output_cols.append(col_name)

        else:
            raise NotImplementedError(
                f"Bare computed expression in SELECT requires an alias: {sel_expr.sql()!r}. "
                "Use 'expr AS alias_name' form."
            )

    # For non-star queries, restrict the mapping to only the selected columns
    if not has_star:
        new_col_to_tensor = {
            name: new_col_to_tensor[name]
            for name in output_cols
            if name in new_col_to_tensor
        }

    return new_col_to_tensor

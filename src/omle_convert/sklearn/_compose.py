"""sklearn.compose → OMLE: ColumnTransformer."""
from __future__ import annotations

import re

import omle

from ._builder import Builder, out_type


def _attr(name, **kw):
    return omle.Attribute(name=name, **kw)


def _core_node(op, input_names, output_name, attributes, builder, output_type=None):
    out = omle.NodeOutput(name=output_name)
    if output_type is not None:
        out = omle.NodeOutput(name=output_name, type=output_type)
    builder.add_node(omle.Node(
        name=output_name, domain="omle.core", op=op,
        inputs=[omle.NodeInput(name=n) for n in input_names],
        outputs=[out],
        attributes=attributes,
    ))
    return output_name


def _column_transformer(ct, input_name, prefix: str, builder: Builder) -> str:
    """Convert a ColumnTransformer.  input_name may be a str (flat tensor) or
    list[str] (per-column named tensors from a mixed-type DataFrame)."""
    from ._transformers import convert_transformer  # lazy to avoid circular

    is_list = isinstance(input_name, list)
    col_name_to_idx = _build_col_name_map(ct) if not is_list else {}

    group_outputs = []
    n_before = len(builder.nodes)

    for t_name, transformer, cols in ct.transformers_:
        if transformer == "drop":
            continue
        sub_prefix = f"{prefix}_{t_name}"

        cols_list = cols.tolist() if hasattr(cols, "tolist") else list(cols)

        if is_list:
            # Per-column named tensors (mixed-type DataFrame): reference them directly.
            group_cols = cols_list
            if not group_cols:
                continue
            has_strings = any(c in builder.str_columns for c in group_cols)
            if has_strings:
                slot_out = group_cols
            elif len(group_cols) == 1:
                slot_out = group_cols[0]
            else:
                slot_out = group_cols  # multi-input operators concat internally
        elif bool(cols_list) and isinstance(cols_list[0], str) and bool(col_name_to_idx):
            # Flat "X" input, CT selects by string name → TakeSlots with column names.
            # The runtime resolves names → column indices at load time.
            slot_out = builder.unique_name(f"{sub_prefix}_slots")
            _core_node("TakeSlots", [input_name], slot_out,
                       [_attr("names", strings=cols_list)], builder,
                       output_type=out_type(len(cols_list), omle.DataType.FLOAT32))
        else:
            # Flat "X" input, CT selects by integer index → TakeSlots with indices.
            col_indices = _resolve_col_indices(cols, col_name_to_idx)
            if not col_indices:
                continue
            slot_out = builder.unique_name(f"{sub_prefix}_slots")
            _core_node("TakeSlots", [input_name], slot_out,
                       [_attr("indices", ints=col_indices)], builder,
                       output_type=out_type(len(col_indices), omle.DataType.FLOAT32))

        if transformer == "passthrough":
            group_outputs.append(slot_out)
        else:
            sub_out = convert_transformer(transformer, slot_out, sub_prefix, builder)
            group_outputs.append(sub_out)

    if not group_outputs:
        return input_name[0] if is_list else input_name
    if len(group_outputs) == 1:
        return group_outputs[0]

    # Multiple groups: wrap in CompositeNode so the outer graph sees one node.
    # Flatten any list elements (is_list passthrough groups contribute multiple columns).
    inner_concat = builder.unique_name(f"{prefix}_concat")
    flat_group_outputs: list[str] = []
    for go in group_outputs:
        if isinstance(go, list):
            flat_group_outputs.extend(go)
        else:
            flat_group_outputs.append(go)
    _core_node("Concat", flat_group_outputs, inner_concat, [], builder)

    inner_nodes = list(builder.nodes[n_before:])
    del builder.nodes[n_before:]

    op_suffix = re.sub(r'(?<!^)(?=[A-Z])', '_', type(ct).__name__).lower()
    outer_out = builder.unique_name(f"{prefix}_{op_suffix}")

    field_names: list[str] = []
    if hasattr(ct, "get_feature_names_out"):
        try:
            field_names = list(ct.get_feature_names_out())
        except Exception:
            pass

    ct_inputs = ([omle.NodeInput(name=n) for n in input_name]
                 if is_list else [omle.NodeInput(name=input_name)])

    composite = omle.CompositeNode(
        nodes=inner_nodes,
        output_aliases=[omle.NameAlias(from_name=inner_concat, to_name=outer_out)],
    )
    composite_type = out_type(len(field_names)) if field_names else None
    builder.add_node(omle.Node(
        name=outer_out, domain="omle.core", op="Composite",
        inputs=ct_inputs,
        outputs=[omle.NodeOutput(name=outer_out, field_names=field_names, type=composite_type)],
        composite=composite,
    ))
    return outer_out


def _build_col_name_map(ct) -> dict[str, int]:
    names = getattr(ct, "feature_names_in_", None)
    if names is None:
        return {}
    return {name: i for i, name in enumerate(names)}


def _resolve_col_indices(cols, col_name_to_idx: dict[str, int] | None = None) -> list[int]:
    if hasattr(cols, "tolist"):
        cols = cols.tolist()
    else:
        cols = list(cols)
    if cols and isinstance(cols[0], str):
        if not col_name_to_idx:
            raise NotImplementedError(
                "ColumnTransformer with string column names requires the transformer "
                "to be fitted on a DataFrame (feature_names_in_ not available)."
            )
        return [col_name_to_idx[c] for c in cols]
    return [int(c) for c in cols]

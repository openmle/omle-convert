"""sklearn.preprocessing → OMLE: StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
Normalizer, Binarizer, OneHotEncoder, OrdinalEncoder, FunctionTransformer (→ Derive nodes)."""
from __future__ import annotations

from typing import Optional

import numpy as np

import omle
from omle.ir.expression import Expression
from omle.ir.types import Scalar

from ._builder import Builder, out_type

# ── FunctionTransformer: numpy func → omle.functions expression ────────────
# Populated lazily; keys are id(numpy_func), values are (qualified_fn_name, extra_arg_exprs).

_NUMPY_EXPR_MAP: Optional[dict[int, tuple[str, list]]] = None


def _get_numpy_expr_map() -> dict[int, tuple[str, list]]:
    global _NUMPY_EXPR_MAP
    if _NUMPY_EXPR_MAP is not None:
        return _NUMPY_EXPR_MAP
    _NUMPY_EXPR_MAP = {
        id(np.log):       ("omle.functions.log",   []),
        id(np.log1p):     ("omle.functions.ln1p",  []),
        id(np.log2):      ("omle.functions.log2",  []),
        id(np.log10):     ("omle.functions.log10", []),
        id(np.exp):       ("omle.functions.exp",   []),
        id(np.sqrt):      ("omle.functions.sqrt",  []),
        id(np.abs):       ("omle.functions.abs",   []),
        id(np.absolute):  ("omle.functions.abs",   []),
        id(np.negative):  ("omle.functions.neg",   []),
        id(np.floor):     ("omle.functions.floor", []),
        id(np.ceil):      ("omle.functions.ceil",  []),
        id(np.sign):      ("omle.functions.sign",  []),
        id(np.square):    ("omle.functions.pow",
                           [Expression.from_literal(Scalar.int(2))]),
        id(np.cbrt):      ("omle.functions.pow",
                           [Expression.from_literal(Scalar.float(1.0 / 3.0))]),
    }
    return _NUMPY_EXPR_MAP


# ── Node helpers ──────────────────────────────────────────────────────────────

def _attr(name: str, **kw) -> omle.Attribute:
    return omle.Attribute(name=name, **kw)


def _feature_node(op: str, input_name, output_name: str,
                  attributes: list[omle.Attribute], builder: Builder,
                  node_name: str = "",
                  output_type: Optional[omle.TensorType] = None) -> str:
    inputs = ([omle.NodeInput(name=n) for n in input_name]
              if isinstance(input_name, list) else [omle.NodeInput(name=input_name)])
    builder.add_node(omle.Node(
        name=node_name or output_name,
        domain="omle.feature",
        op=op,
        inputs=inputs,
        outputs=[omle.NodeOutput(name=output_name, type=output_type)],
        attributes=attributes,
    ))
    return output_name


def _core_node(op: str, input_names: list[str], output_name: str,
               attributes: list[omle.Attribute], builder: Builder,
               node_name: str = "",
               output_type: Optional[omle.TensorType] = None) -> str:
    builder.add_node(omle.Node(
        name=node_name or output_name,
        domain="omle.core",
        op=op,
        inputs=[omle.NodeInput(name=n) for n in input_names],
        outputs=[omle.NodeOutput(name=output_name, type=output_type)],
        attributes=attributes,
    ))
    return output_name


# ── FunctionTransformer ───────────────────────────────────────────────────────

def _derive_node(col_name: str, fn_name: str, extra_args: list,
                 out_name: str, builder: Builder,
                 output_type: Optional[omle.TensorType] = None) -> str:
    """Emit a ``Derive`` node: ``out = fn_name(col_name, *extra_args)``."""
    col_expr = Expression.from_ref(col_name)
    expr = Expression.from_apply(fn_name, col_expr, *extra_args)
    builder.add_node(omle.Node(
        name=out_name,
        domain="omle.core",
        op="Derive",
        inputs=[omle.NodeInput(name=col_name)],
        outputs=[omle.NodeOutput(name=out_name, type=output_type)],
        attributes=[omle.Attribute(name="expr", expr=expr)],
    ))
    return out_name


def _function_transformer(t, input_name: str, prefix: str, builder: Builder) -> str:
    """Convert a ``FunctionTransformer`` to a single ``Derive`` node.

    ``None`` func is treated as passthrough.  Recognised numpy functions are
    mapped to the corresponding ``omle.functions.*`` expression.  A single
    Derive node handles any number of input columns; the runtime evaluates the
    expression column-by-column and assembles a matrix output.
    """
    func = getattr(t, "func", None)
    if func is None:
        return input_name  # passthrough

    expr_map = _get_numpy_expr_map()
    match = expr_map.get(id(func))
    if match is None:
        raise NotImplementedError(
            f"FunctionTransformer: unsupported func {func!r}. "
            "Supported numpy functions: log, log1p, log2, log10, exp, sqrt, "
            "abs/absolute, negative, floor, ceil, sign, square, cbrt."
        )
    fn_name, extra_args = match

    out = builder.unique_name(f"{prefix}_derive")
    n_cols = getattr(t, "n_features_in_", None)
    otype = out_type(n_cols) if n_cols is not None else None
    return _derive_node(input_name, fn_name, extra_args, out, builder, output_type=otype)


# ── Scalers ───────────────────────────────────────────────────────────────────

def _standard_scaler(t, input_name, prefix, builder):
    out = builder.unique_name(f"{prefix}_standard_scaler")
    return _feature_node("StandardScaler", input_name, out, [
        builder.tensor_attr("mean",  f"{prefix}_mean",  np.asarray(t.mean_)),
        builder.tensor_attr("scale", f"{prefix}_scale", np.asarray(t.scale_)),
    ], builder, output_type=out_type(t.n_features_in_))


def _minmax_scaler(t, input_name, prefix, builder):
    out = builder.unique_name(f"{prefix}_minmax_scaler")
    fr_min, fr_max = t.feature_range
    return _feature_node("MinMaxScaler", input_name, out, [
        builder.tensor_attr("data_min", f"{prefix}_data_min", np.asarray(t.data_min_)),
        # Runtime interprets data_max as the per-feature scale (max - min), not the absolute maximum.
        builder.tensor_attr("data_max", f"{prefix}_data_max", np.asarray(t.data_range_)),
        _attr("feature_range_min", f64=float(fr_min)),
        _attr("feature_range_max", f64=float(fr_max)),
    ], builder, output_type=out_type(t.n_features_in_))


def _robust_scaler(t, input_name, prefix, builder):
    out    = builder.unique_name(f"{prefix}_robust_scaler")
    n_feat = t.n_features_in_
    center = np.asarray(t.center_) if t.center_ is not None else np.zeros(n_feat)
    scale  = np.asarray(t.scale_)  if t.scale_  is not None else np.ones(n_feat)
    return _feature_node("RobustScaler", input_name, out, [
        builder.tensor_attr("center", f"{prefix}_center", center),
        builder.tensor_attr("scale",  f"{prefix}_scale",  scale),
    ], builder, output_type=out_type(n_feat))


def _maxabs_scaler(t, input_name, prefix, builder):
    out = builder.unique_name(f"{prefix}_maxabs_scaler")
    return _feature_node("MaxAbsScaler", input_name, out, [
        builder.tensor_attr("scale", f"{prefix}_scale", np.asarray(t.scale_)),
    ], builder, output_type=out_type(t.n_features_in_))


# ── Other feature transformers ────────────────────────────────────────────────

def _normalizer(t, input_name, prefix, builder):
    out = builder.unique_name(f"{prefix}_normalizer")
    return _feature_node("Normalizer", input_name, out, [
        _attr("norm", s=str(t.norm)),
    ], builder, output_type=out_type(t.n_features_in_))


def _binarizer(t, input_name, prefix, builder):
    out = builder.unique_name(f"{prefix}_binarizer")
    return _feature_node("Binarizer", input_name, out, [
        _attr("threshold", f64=float(t.threshold)),
    ], builder, output_type=out_type(t.n_features_in_))


# ── Encoders ──────────────────────────────────────────────────────────────────

def _ohe(t, input_name, prefix, builder):
    """OneHotEncoder: single node with concatenated category vocab and offset index."""
    all_cats = [str(c) for cats in t.categories_ for c in cats]
    offsets = [0]
    for cats in t.categories_:
        offsets.append(offsets[-1] + len(cats))
    n_out = sum(len(cats) for cats in t.categories_)
    out = builder.unique_name(f"{prefix}_ohe")
    return _feature_node("OneHotEncoder", input_name, out, [
        builder.string_tensor_attr("categories",       f"{prefix}_cats",        all_cats),
        builder.int_tensor_attr(   "category_offsets", f"{prefix}_cat_offsets", np.array(offsets, dtype=np.int64)),
    ], builder, output_type=out_type(n_out))


def _ordinal_encoder(t, input_name, prefix, builder):
    """OrdinalEncoder: single node with concatenated category vocab and offset index."""
    all_cats = [str(c) for cats in t.categories_ for c in cats]
    offsets = [0]
    for cats in t.categories_:
        offsets.append(offsets[-1] + len(cats))
    out = builder.unique_name(f"{prefix}_ordinal")
    return _feature_node("OrdinalEncoder", input_name, out, [
        builder.string_tensor_attr("categories",       f"{prefix}_cats",        all_cats),
        builder.int_tensor_attr(   "category_offsets", f"{prefix}_cat_offsets", np.array(offsets, dtype=np.int64)),
    ], builder, output_type=out_type(len(t.categories_)))


def _ordinal_encoder_with_values(
    categories_per_feat: list[list],
    values_per_feat: list[list[float]],
    unknown_values: list[float],
    input_name,
    prefix: str,
    builder: "Builder",
) -> str:
    """OrdinalEncoder variant that returns float lookup values instead of ordinal indices.

    Used by category_encoders: each category maps to a precomputed float (e.g. target mean).
    unknown_values[f] is the fallback for unseen categories in feature f.
    """
    all_cats = [str(c) for cats in categories_per_feat for c in cats]
    all_vals = [float(v) for vals in values_per_feat for v in vals]
    offsets = [0]
    for cats in categories_per_feat:
        offsets.append(offsets[-1] + len(cats))
    out = builder.unique_name(f"{prefix}_ordinal")
    attrs = [
        builder.string_tensor_attr("categories",       f"{prefix}_cats",        all_cats),
        builder.int_tensor_attr(   "category_offsets", f"{prefix}_cat_offsets", np.array(offsets, dtype=np.int64)),
        builder.tensor_attr(       "encoded_values",   f"{prefix}_values",      np.array(all_vals,      dtype=np.float64)),
        builder.tensor_attr(       "default_values",   f"{prefix}_unk",         np.array(unknown_values, dtype=np.float64)),
    ]
    return _feature_node("OrdinalEncoder", input_name, out, attrs, builder,
                         output_type=out_type(len(categories_per_feat)))


# ── Label / binning encoders ──────────────────────────────────────────────────

def _label_encoder(t, input_name, prefix, builder):
    """LabelEncoder: single node with label vocab and offset index."""
    labels = [str(c) for c in t.classes_]
    out = builder.unique_name(f"{prefix}_label_encoder")
    return _feature_node("LabelEncoder", input_name, out, [
        builder.string_tensor_attr("labels",        f"{prefix}_labels",        labels),
        builder.int_tensor_attr(   "label_offsets", f"{prefix}_label_offsets", np.array([0, len(labels)], dtype=np.int64)),
    ], builder, output_type=out_type(1))


def _label_binarizer(t, input_name, prefix, builder):
    """LabelBinarizer → LabelBinarize operator."""
    n_out = 1 if len(t.classes_) == 2 else len(t.classes_)
    out = builder.unique_name(f"{prefix}_label_binarizer")
    return _feature_node("LabelBinarizer", input_name, out, [
        builder.string_tensor_attr("classes", f"{prefix}_classes", [str(c) for c in t.classes_]),
        _attr("neg_label", i=int(t.neg_label)),
        _attr("pos_label", i=int(t.pos_label)),
    ], builder, output_type=out_type(n_out))


def _target_encoder(t, input_name, prefix, builder):
    """TargetEncoder (sklearn 1.3+) → single TargetEncoder node over full input matrix."""
    target_type = getattr(t, "target_type_", "continuous")
    if target_type == "binary":
        target_kind = "binary"
    elif target_type == "multiclass":
        target_kind = "multiclass"
    else:
        target_kind = "regression"

    # Concatenate categories across all features; build cumulative offsets.
    all_cats = []
    offsets = [0]
    for cats in t.categories_:
        all_cats.extend(str(c) for c in cats)
        offsets.append(len(all_cats))

    # Concatenate encoded values: shape [C_total] or [C_total, K].
    enc = np.concatenate([np.asarray(e) for e in t.encodings_], axis=0)

    # Default values: shape [F] or [F, K].
    if target_kind == "multiclass":
        target_mean = np.asarray(t.target_mean_).ravel()
        default_val = np.tile(target_mean, (len(t.categories_), 1))
    else:
        default_val = np.array([float(t.target_mean_)] * len(t.categories_))

    n_out = t.n_features_in_ * (len(t.target_mean_) if target_kind == "multiclass" else 1)
    out = builder.unique_name(f"{prefix}_target_encoder")
    return _feature_node("TargetEncoder", input_name, out, [
        builder.string_tensor_attr("categories",       f"{prefix}_cats",    all_cats),
        builder.int_tensor_attr(   "category_offsets", f"{prefix}_offsets", np.array(offsets, dtype=np.int64)),
        builder.tensor_attr(       "encoded_values",   f"{prefix}_enc",     enc),
        builder.tensor_attr(       "default_values",   f"{prefix}_def",     default_val),
        _attr("target_kind",  s=target_kind),
    ], builder, output_type=out_type(n_out))


def _multi_label_binarizer(t, input_name, prefix, builder):
    out = builder.unique_name(f"{prefix}_multilabel_binarizer")
    return _feature_node("MultiLabelBinarizer", input_name, out, [
        builder.string_tensor_attr("classes", f"{prefix}_classes", [str(c) for c in t.classes_]),
    ], builder, output_type=out_type(len(t.classes_)))


def _power_transformer(t, input_name, prefix, builder):
    method = t.method.replace("-", "_")  # "box-cox" → "box_cox", "yeo-johnson" → "yeo_johnson"
    attrs = [
        _attr("method", s=method),
        builder.tensor_attr("lambdas", f"{prefix}_lambdas", np.asarray(t.lambdas_)),
        _attr("standardize", b=bool(t.standardize)),
    ]
    if t.standardize:
        attrs.append(builder.tensor_attr("mean",  f"{prefix}_mean",  np.asarray(t._scaler.mean_)))
        attrs.append(builder.tensor_attr("scale", f"{prefix}_scale", np.asarray(t._scaler.scale_)))
    out = builder.unique_name(f"{prefix}_power_transformer")
    return _feature_node("PowerTransformer", input_name, out, attrs, builder,
                         output_type=out_type(t.n_features_in_))


def _quantile_transformer(t, input_name, prefix, builder):
    # Runtime op_quantile_transformer expects:
    #   "quantiles"  : [n_quantiles] — the quantile levels (0..1) = t.references_
    #   "references" : [n_features, n_quantiles] — feature values per level = t.quantiles_.T
    out = builder.unique_name(f"{prefix}_quantile_transformer")
    return _feature_node("QuantileTransformer", input_name, out, [
        builder.tensor_attr("quantiles",  f"{prefix}_quantiles",  np.asarray(t.references_)),
        builder.tensor_attr("references", f"{prefix}_references", np.asarray(t.quantiles_).T.copy()),
        _attr("output_distribution", s=t.output_distribution),
    ], builder, output_type=out_type(t.n_features_in_))


def _spline_transformer(t, input_name, prefix, builder):
    # Store augmented knot vectors as [n_aug, n_features] so the runtime can read
    # per-feature knots as knots[k * n_features + f] (row k, col f).
    knot_matrix = np.array([bs.t for bs in t.bsplines_], dtype=np.float64).T  # [n_aug, n_features]
    out = builder.unique_name(f"{prefix}_spline_transformer")
    return _feature_node("SplineTransformer", input_name, out, [
        builder.tensor_attr("knots", f"{prefix}_knots", knot_matrix),
        _attr("degree", i=int(t.degree)),
        _attr("include_bias", b=bool(t.include_bias)),
        _attr("extrapolation", s=str(t.extrapolation)),
    ], builder, output_type=out_type(t.n_features_out_))


def _polynomial_features(t, input_name, prefix, builder):
    degree = t.degree
    if isinstance(degree, (tuple, list)):
        min_degree, max_degree = int(degree[0]), int(degree[1])
    else:
        min_degree = int(getattr(t, "min_degree", 0))
        max_degree = int(degree)
    out = builder.unique_name(f"{prefix}_poly_features")
    return _feature_node("PolynomialFeatures", input_name, out, [
        _attr("min_degree", i=min_degree),
        _attr("max_degree", i=max_degree),
        _attr("interaction_only", b=bool(t.interaction_only)),
        _attr("include_bias",     b=bool(t.include_bias)),
        builder.int_tensor_attr("powers", f"{prefix}_poly_powers",
                                np.asarray(t.powers_, dtype=np.int64)),
    ], builder, output_type=out_type(t.n_output_features_))


def _kbins_discretizer(t, input_name, prefix, builder):
    encode = t.encode  # 'ordinal', 'onehot', 'onehot-dense'
    n_features = len(t.bin_edges_)

    # Build concatenated boundaries and per-feature offsets.
    all_boundaries: list[float] = []
    boundary_offsets: list[int] = [0]
    for edges in t.bin_edges_:
        interior = edges[1:-1].tolist()
        all_boundaries.extend(interior)
        boundary_offsets.append(len(all_boundaries))

    bkt_attrs = [
        builder.tensor_attr("boundaries", f"{prefix}_boundaries",
                            np.array(all_boundaries, dtype=np.float64)),
        builder.int_tensor_attr("boundary_offsets", f"{prefix}_boundary_offsets",
                                np.array(boundary_offsets, dtype=np.int64)),
    ]

    bkt = builder.unique_name(f"{input_name}_btk")
    _feature_node("Bucketizer", input_name, bkt, bkt_attrs, builder,
                  node_name=builder.unique_node_name(f"{prefix}_bucketizer"),
                  output_type=out_type(n_features))

    if encode == "ordinal":
        return bkt

    # OneHotEncoder over all features: concatenated bin-index categories + offsets.
    all_cats = [str(j) for i in range(n_features) for j in range(int(t.n_bins_[i]))]
    cat_offsets = [0]
    total_bins = 0
    for i in range(n_features):
        total_bins += int(t.n_bins_[i])
        cat_offsets.append(total_bins)
    ohe = builder.unique_name(f"{bkt}_ohe")
    _feature_node("OneHotEncoder", bkt, ohe, [
        builder.string_tensor_attr("categories",       f"{prefix}_cats",        all_cats),
        builder.int_tensor_attr(   "category_offsets", f"{prefix}_cat_offsets", np.array(cat_offsets, dtype=np.int64)),
    ], builder, node_name=builder.unique_node_name(f"{prefix}_one_hot_encoder"),
       output_type=out_type(total_bins))

    if encode == "onehot":
        out = builder.unique_name(f"{ohe}_sparse")
        return _core_node("DenseToSparse", [ohe], out, [], builder,
                          node_name=builder.unique_node_name(f"{prefix}_dense_to_sparse"))
    return ohe

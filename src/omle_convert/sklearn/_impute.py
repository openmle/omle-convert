"""sklearn.impute → OMLE: SimpleImputer, MissingIndicator, KNNImputer."""
from __future__ import annotations

import numpy as np

import omle

from ._builder import Builder, out_type


def _attr(name, **kw):
    return omle.Attribute(name=name, **kw)


def _feature_node(op, input_name, output_name, attributes, builder, output_type=None):
    inputs = ([omle.NodeInput(name=n) for n in input_name]
              if isinstance(input_name, list) else [omle.NodeInput(name=input_name)])
    builder.add_node(omle.Node(
        name=output_name, domain="omle.feature", op=op,
        inputs=inputs,
        outputs=[omle.NodeOutput(name=output_name, type=output_type)],
        attributes=attributes,
    ))
    return output_name


def _imputer(t, input_name: str, prefix: str, builder: Builder) -> str:
    out = builder.unique_name(f"{prefix}_imputer")
    return _feature_node("Imputer", input_name, out, [
        builder.tensor_attr("fill_tensor", f"{prefix}_fill", np.asarray(t.statistics_)),
    ], builder, output_type=out_type(t.n_features_in_))


def _missing_indicator(t, input_name: str, prefix: str, builder: Builder) -> str:
    features_mode = "all" if getattr(t, "features", "missing-only") == "all" else "missing_only"
    attrs = [_attr("features_mode", s=features_mode)]
    if features_mode == "missing_only":
        attrs.append(builder.int_tensor_attr(
            "feature_indices", f"{prefix}_feature_indices",
            np.asarray(t.features_, dtype=np.int64),
        ))
    attrs.append(_attr("error_on_new", b=bool(t.error_on_new)))
    out = builder.unique_name(f"{prefix}_missing_indicator")
    n_out = t.n_features_in_ if features_mode == "all" else len(t.features_)
    return _feature_node("MissingIndicator", input_name, out, attrs, builder,
                         output_type=out_type(n_out, dtype=omle.DataType.BOOL))


def _knn_imputer(t, input_name: str, prefix: str, builder: Builder) -> str:
    out = builder.unique_name(f"{prefix}_knn_imputer")
    return _feature_node("KNNImputer", input_name, out, [
        builder.tensor_attr("train_features", f"{prefix}_train_features", np.asarray(t._fit_X)),
        _attr("n_neighbors", i=int(t.n_neighbors)),
        _attr("weights", s=str(t.weights)),
        _attr("metric", s="nan_euclidean"),
        _attr("keep_empty_features", b=bool(getattr(t, "keep_empty_features", False))),
    ], builder, output_type=out_type(t._fit_X.shape[1]))

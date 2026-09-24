"""Preprocessing transformer dispatcher."""
from __future__ import annotations

import omle

from ._builder import Builder, out_type
from ._category_encoders import _ALL_CE_CLASSES as _CE_CLASSES
from ._compose import _column_transformer
from ._decomposition import (
    _factor_analysis,
    _fast_ica,
    _kernel_pca,
    _lda,
    _nmf,
    _pca,
    _sparse_pca,
    _truncated_svd,
)
from ._estimators import cls_snake
from ._impute import _imputer, _knn_imputer, _missing_indicator
from ._pipeline import _feature_union, _pipeline_as_transformer
from ._preprocessing import (
    _binarizer,
    _function_transformer,
    _kbins_discretizer,
    _label_binarizer,
    _label_encoder,
    _maxabs_scaler,
    _minmax_scaler,
    _multi_label_binarizer,
    _normalizer,
    _ohe,
    _ordinal_encoder,
    _polynomial_features,
    _power_transformer,
    _quantile_transformer,
    _robust_scaler,
    _spline_transformer,
    _standard_scaler,
    _target_encoder,
)
from ._selection import _SELECTION_DISPATCH
from ._text import _TEXT_DISPATCH

_TRANSFORMER_SHORT_ALIAS: dict[str, str] = {
    "StandardScaler":            "std",
    "MinMaxScaler":              "minmax",
    "RobustScaler":              "robust",
    "MaxAbsScaler":              "maxabs",
    "Normalizer":                "norm",
    "SimpleImputer":             "imputer",
    "KNNImputer":                "knn_imputer",
    "MissingIndicator":          "missing",
    "Binarizer":                 "binarizer",
    "OneHotEncoder":             "ohe",
    "OrdinalEncoder":            "ordinal",
    "LabelEncoder":              "le",
    "LabelBinarizer":            "lb",
    "MultiLabelBinarizer":       "mlb",
    "TargetEncoder":             "target_enc",
    "KBinsDiscretizer":          "kbins",
    "PowerTransformer":          "power",
    "QuantileTransformer":       "quantile",
    "SplineTransformer":         "spline",
    "PolynomialFeatures":        "poly",
    "PCA":                       "pca",
    "TruncatedSVD":              "svd",
    "FastICA":                   "ica",
    "FactorAnalysis":            "fa",
    "KernelPCA":                 "kpca",
    "SparsePCA":                 "spca",
    "NMF":                       "nmf",
    "LatentDirichletAllocation": "lda",
    "ColumnTransformer":         "ct",
    "FunctionTransformer":       "passthrough",
    "FeatureUnion":              "union",
    "CountVectorizer":           "count_vec",
    "TfidfTransformer":          "tfidf",
    "TfidfVectorizer":           "tfidf_vec",
    "HashingVectorizer":         "hash_vec",
    "SelectKBest":               "skbest",
    "SelectPercentile":          "spct",
    "SelectFdr":                 "sfdr",
    "SelectFpr":                 "sfpr",
    "SelectFwe":                 "sfwe",
    "VarianceThreshold":         "var_thresh",
    "SelectFromModel":           "sfm",
    "RFE":                       "rfe",
    "RFECV":                     "rfecv",
    "SequentialFeatureSelector": "sfs",
    # category_encoders. Note: category_encoders' TargetEncoder shares a class
    # name with sklearn's, so the "target_enc" alias above covers both; the two
    # are told apart by module at dispatch time, not here.
    "MEstimateEncoder":          "ce_m_est",
    "WOEEncoder":                "ce_woe",
    "JamesSteinEncoder":         "ce_js",
    "QuantileEncoder":           "ce_quantile",
    "SummaryEncoder":            "ce_summary",
    "CountEncoder":              "ce_count",
    "CatBoostEncoder":           "ce_catboost",
    "LeaveOneOutEncoder":        "ce_loo",
}

_DISPATCH = {
    "StandardScaler":    _standard_scaler,
    "MinMaxScaler":      _minmax_scaler,
    "RobustScaler":      _robust_scaler,
    "MaxAbsScaler":      _maxabs_scaler,
    "Normalizer":        _normalizer,
    "SimpleImputer":     _imputer,
    "Binarizer":         _binarizer,
    "OneHotEncoder":     _ohe,
    "OrdinalEncoder":    _ordinal_encoder,
    "PCA":                    _pca,
    "TruncatedSVD":           _truncated_svd,
    "FastICA":                _fast_ica,
    "FactorAnalysis":         _factor_analysis,
    "KernelPCA":              _kernel_pca,
    "SparsePCA":              _sparse_pca,
    "NMF":                    _nmf,
    "LatentDirichletAllocation": _lda,
    "MissingIndicator":       _missing_indicator,
    "KNNImputer":             _knn_imputer,
    "ColumnTransformer": _column_transformer,
    "LabelEncoder":      _label_encoder,
    "KBinsDiscretizer":  _kbins_discretizer,
    "LabelBinarizer":    _label_binarizer,
    "TargetEncoder":     _target_encoder,
    "PowerTransformer":  _power_transformer,
    "QuantileTransformer": _quantile_transformer,
    "SplineTransformer": _spline_transformer,
    "PolynomialFeatures":    _polynomial_features,
    "MultiLabelBinarizer":   _multi_label_binarizer,
    **_TEXT_DISPATCH,
    **_SELECTION_DISPATCH,
}


# Transformers that can natively accept multiple single-column inputs (per-column mode).
# category_encoders transformers are detected by module at dispatch time;
# include their class names here so the list-input path is taken.

_MULTI_INPUT_TRANSFORMERS = {
    # Encoding
    "OneHotEncoder", "OrdinalEncoder",
    # Scaling
    "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler",
    "Normalizer", "Binarizer",
    # Imputation
    "SimpleImputer", "MissingIndicator", "KNNImputer",
    # Feature engineering
    "TargetEncoder", "PowerTransformer", "QuantileTransformer",
    "SplineTransformer", "PolynomialFeatures",
    # Decomposition
    "TruncatedSVD", "FastICA", "FactorAnalysis",
    "KernelPCA", "SparsePCA", "NMF", "LatentDirichletAllocation",
} | _CE_CLASSES


def _merge_columns(cols: list, prefix: str, builder: Builder) -> str:
    """Merge a list of numeric per-column tensor names into one tensor via Concat."""
    merged = builder.unique_name(f"{prefix}_merged")
    builder.add_node(omle.Node(
        name=merged, domain="omle.core", op="Concat",
        inputs=[omle.NodeInput(name=c) for c in cols],
        outputs=[omle.NodeOutput(name=merged, type=out_type(len(cols)))],
    ))
    return merged


def convert_transformer(transformer, input_name, prefix: str, builder: Builder) -> str:
    """Convert one fitted sklearn transformer to OMLE nodes. Returns output tensor name."""
    cls_name = type(transformer).__name__

    # Per-column mode: input_name is a list of named column tensors.
    # ColumnTransformer and multi-input-aware transformers handle the list natively;
    # everything else gets merged into a single tensor first (numeric Concat only).
    if isinstance(input_name, list):
        if transformer == "passthrough":
            return input_name[0] if len(input_name) == 1 else _merge_columns(input_name, prefix, builder)
        if cls_name not in {"ColumnTransformer", "Pipeline"} | _MULTI_INPUT_TRANSFORMERS:
            input_name = _merge_columns(input_name, prefix, builder)

    if transformer == "passthrough":
        return input_name

    # Pipeline: output is already named by its last step's chain — don't rename
    if cls_name == "Pipeline":
        return _pipeline_as_transformer(transformer, input_name, prefix, builder)

    n_before = len(builder.nodes)

    if cls_name == "FunctionTransformer":
        fn = _function_transformer
    elif cls_name == "FeatureUnion":
        fn = _feature_union
    elif type(transformer).__module__.startswith("category_encoders"):
        from ._category_encoders import _category_encoder
        fn = _category_encoder
    else:
        fn = _DISPATCH.get(cls_name)
        if fn is None:
            raise NotImplementedError(
                f"Unsupported transformer: {cls_name}. "
                "Supported: StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, "
                "Normalizer, SimpleImputer, Binarizer, OneHotEncoder, OrdinalEncoder, "
                "PCA, ColumnTransformer, FunctionTransformer, Pipeline, FeatureUnion, "
                "LabelEncoder, KBinsDiscretizer, LabelBinarizer, TargetEncoder, "
                "PowerTransformer, QuantileTransformer, SplineTransformer, PolynomialFeatures, MultiLabelBinarizer, "
                "TruncatedSVD, FastICA, FactorAnalysis, KernelPCA, SparsePCA, NMF, LatentDirichletAllocation, "
                "MissingIndicator, KNNImputer, "
                "SelectKBest, SelectPercentile, SelectFdr, SelectFpr, SelectFwe, "
                "VarianceThreshold, SelectFromModel, RFE, RFECV, SequentialFeatureSelector, "
                "CountVectorizer, TfidfTransformer, TfidfVectorizer, HashingVectorizer, "
                "TargetEncoder (CE), MEstimateEncoder, WOEEncoder, JamesSteinEncoder, "
                "QuantileEncoder, SummaryEncoder, CountEncoder, CatBoostEncoder, LeaveOneOutEncoder."
            )
    old_out = fn(transformer, input_name, prefix, builder)

    new_nodes = builder.nodes[n_before:]
    if not new_nodes:
        return old_out

    node_suffix = cls_snake(transformer)
    # Sub-dispatch already named nodes (and tensors) following the policy when
    # it called unique_node_name internally (e.g. text transformers). In that
    # case skip both renames; otherwise apply the canonical names here.
    already_named = new_nodes[-1].name in builder._used_node_names
    if already_named:
        new_out = old_out
    else:
        out_suffix = _TRANSFORMER_SHORT_ALIAS.get(cls_name, node_suffix)
        # input_name is a str for a flat tensor, but a list[str] for a
        # multi-column transformer (see _compose._column_transformer). An
        # f-string over the list interpolates its repr, producing a tensor
        # named "['age', 'fare', ...]_ct" — 100+ characters of quotes, commas
        # and brackets. sklearn itself never names an output after its inputs:
        # it names by step path (preprocessor__num__imputer), and reserves
        # per-column identity for get_feature_names_out(). Follow that — fall
        # back to the step prefix, which is what the node name already uses.
        # Column-level names are not lost: they live on output.field_names.
        base = input_name if isinstance(input_name, str) else prefix
        new_out = builder.unique_name(f"{base}_{out_suffix}")
        new_nodes[-1].name = builder.unique_node_name(f"{prefix}_{node_suffix}")

    # Derive field_names from get_feature_names_out() when available.
    # These are column-level metadata only; the tensor name (new_out) is always
    # unique and distinct from any input name, so there is no graph conflict.
    derived_field_names: list[str] = []
    if hasattr(transformer, "get_feature_names_out"):
        try:
            derived_field_names = [str(n) for n in transformer.get_feature_names_out()]
        except Exception:
            pass

    last = new_nodes[-1]
    last.outputs = [
        omle.NodeOutput(
            name=new_out if o.name == old_out else o.name,
            type=o.type,
            measure_level=o.measure_level,
            domain=o.domain,
            description=o.description,
            role=o.role,
            binding=o.binding,
            field_names=o.field_names or (derived_field_names if o.name == old_out else []),
        )
        for o in last.outputs
    ]
    if last.composite is not None:
        last.composite.output_aliases = [
            omle.NameAlias(
                from_name=a.from_name,
                to_name=new_out if a.to_name == old_out else a.to_name,
            )
            for a in last.composite.output_aliases
        ]
    return new_out

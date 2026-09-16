"""category_encoders → OMLE: lookup-table encoders.

TargetEncoder → TargetEncoder op (string-input path added to the C++ runtime).

All other CE encoders → OrdinalEncoder op with float lookup values:

Group 1 – ordinal-code keyed mapping (have ``ordinal_encoder`` + ``mapping[col]`` = Series(code→float)):
    MEstimateEncoder, WOEEncoder, JamesSteinEncoder, QuantileEncoder, SummaryEncoder

Group 2 – string-keyed mapping (have ``ordinal_encoder`` + ``mapping[col]`` = Series(str→float)):
    CountEncoder

Group 3 – sum/count DataFrame, CatBoost formula:
    CatBoostEncoder

Group 4 – sum/count DataFrame, leave-one-out formula:
    LeaveOneOutEncoder
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._builder import Builder
from ._preprocessing import _attr, _feature_node, _ordinal_encoder_with_values


def _extract_ordinal_code_keyed(enc) -> tuple[list, list, list]:
    """Extract (categories_per_feat, values_per_feat, unknown_values) for Group 1 encoders.

    These encoders first ordinal-encode X, then map ordinal codes → float.
    ``enc.ordinal_encoder.category_mapping`` provides the str→code mapping.
    ``enc.mapping[col]`` is Series(ordinal_code → float).
    """
    categories_per_feat = []
    values_per_feat = []
    unknown_values = []

    global_mean = getattr(enc, "_mean", np.nan)

    for switch in enc.ordinal_encoder.category_mapping:
        col = switch["col"]
        ordinal_series = switch["mapping"]   # Series(index=str_cat, values=int_code)
        col_float_map = enc.mapping[col]     # Series(index=int_code, values=float)

        cats = []
        vals = []
        for cat, code in ordinal_series.items():
            if pd.isna(cat) or int(code) < 0:
                continue
            cats.append(str(cat))
            v = col_float_map.get(int(code), global_mean)
            vals.append(float(v) if not pd.isna(v) else float(global_mean if not pd.isna(global_mean) else 0.0))

        # Fallback for unseen: code -1
        unk_raw = col_float_map.get(-1, global_mean)
        unk = float(unk_raw) if not pd.isna(unk_raw) else float(global_mean if not pd.isna(global_mean) else 0.0)

        categories_per_feat.append(cats)
        values_per_feat.append(vals)
        unknown_values.append(unk)

    return categories_per_feat, values_per_feat, unknown_values


def _extract_string_keyed(enc) -> tuple[list, list, list]:
    """Extract (categories_per_feat, values_per_feat, unknown_values) for Group 2 (CountEncoder).

    ``mapping[col]`` is Series(str → float) keyed by original category strings.
    """
    categories_per_feat = []
    values_per_feat = []
    unknown_values = []

    for col in enc.cols:
        col_mapping = enc.mapping[col]  # Series(str → float)

        cats = []
        vals = []
        for cat, val in col_mapping.items():
            if pd.isna(cat):
                continue
            cats.append(str(cat))
            vals.append(float(val) if not pd.isna(val) else 0.0)

        categories_per_feat.append(cats)
        values_per_feat.append(vals)
        unknown_values.append(0.0)  # CountEncoder fills unknown with 0

    return categories_per_feat, values_per_feat, unknown_values


def _extract_catboost(enc) -> tuple[list, list, list]:
    """Extract for Group 3 (CatBoostEncoder).

    Formula at inference (y=None): (sum + mean*a) / (count + a) if count > 1 else mean
    """
    global_mean = float(enc._mean)
    a = float(enc.a)

    categories_per_feat = []
    values_per_feat = []
    unknown_values = []

    for col in enc.cols:
        colmap = enc.mapping[col]   # DataFrame(index=orig_cat, columns=['sum','count'])

        cats = []
        vals = []
        for cat, row in colmap.iterrows():
            if pd.isna(cat):
                continue
            count = float(row["count"])
            s = float(row["sum"])
            v = (s + global_mean * a) / (count + a) if count > 1 else global_mean
            cats.append(str(cat))
            vals.append(v)

        categories_per_feat.append(cats)
        values_per_feat.append(vals)
        unknown_values.append(global_mean)

    return categories_per_feat, values_per_feat, unknown_values


def _extract_loo(enc) -> tuple[list, list, list]:
    """Extract for Group 4 (LeaveOneOutEncoder).

    Formula at inference (y=None): sum/count if count > 1 else global_mean
    """
    global_mean = float(enc._mean)

    categories_per_feat = []
    values_per_feat = []
    unknown_values = []

    for col in enc.cols:
        colmap = enc.mapping[col]   # DataFrame(index=orig_cat, columns=['sum','count'])

        cats = []
        vals = []
        for cat, row in colmap.iterrows():
            if pd.isna(cat):
                continue
            count = float(row["count"])
            s = float(row["sum"])
            v = s / count if count > 1 else global_mean
            cats.append(str(cat))
            vals.append(v)

        categories_per_feat.append(cats)
        values_per_feat.append(vals)
        unknown_values.append(global_mean)

    return categories_per_feat, values_per_feat, unknown_values


def _ce_target_encoder(enc, input_name, prefix: str, builder: Builder) -> str:
    """category_encoders TargetEncoder → TargetEncoder op with string-input support.

    Emits the same TargetEncoder op as sklearn's TargetEncoder. The C++ runtime
    handles string inputs via its is_str branch, so no prior ordinal encoding is needed.
    """
    all_cats = []
    all_enc = []
    offsets = [0]
    default_vals = []
    global_mean = float(enc._mean)

    for switch in enc.ordinal_encoder.category_mapping:
        col = switch["col"]
        ordinal_series = switch["mapping"]   # Series(index=str_cat, values=int_code)
        col_float_map = enc.mapping[col]     # Series(index=int_code, values=float)

        for cat, code in ordinal_series.items():
            if pd.isna(cat) or int(code) < 0:
                continue
            all_cats.append(str(cat))
            v = col_float_map.get(int(code), global_mean)
            all_enc.append(float(v) if not pd.isna(v) else global_mean)

        offsets.append(len(all_cats))
        unk = col_float_map.get(-1, global_mean)
        default_vals.append(float(unk) if not pd.isna(unk) else global_mean)

    out = builder.unique_name(f"{prefix}_target_encoder")
    return _feature_node("TargetEncoder", input_name, out, [
        builder.string_tensor_attr("categories",       f"{prefix}_cats",    all_cats),
        builder.int_tensor_attr(   "category_offsets", f"{prefix}_offsets", np.array(offsets, dtype=np.int64)),
        builder.tensor_attr(       "encoded_values",   f"{prefix}_enc",     np.array(all_enc, dtype=np.float64)),
        builder.tensor_attr(       "default_values",   f"{prefix}_def",     np.array(default_vals, dtype=np.float64)),
        _attr("target_kind",  s="regression"),
    ], builder)


# Map class name → extraction function (OrdinalEncoder-with-values path)
_EXTRACTOR = {
    "MEstimateEncoder":    _extract_ordinal_code_keyed,
    "WOEEncoder":          _extract_ordinal_code_keyed,
    "JamesSteinEncoder":   _extract_ordinal_code_keyed,
    "QuantileEncoder":     _extract_ordinal_code_keyed,
    "SummaryEncoder":      _extract_ordinal_code_keyed,
    "CountEncoder":        _extract_string_keyed,
    "CatBoostEncoder":     _extract_catboost,
    "LeaveOneOutEncoder":  _extract_loo,
}

_ALL_CE_CLASSES = {"TargetEncoder"} | set(_EXTRACTOR)


def _category_encoder(enc, input_name, prefix: str, builder: Builder) -> str:
    """Convert any supported category_encoders encoder to an OMLE node."""
    cls_name = type(enc).__name__

    if cls_name == "TargetEncoder":
        return _ce_target_encoder(enc, input_name, prefix, builder)

    extractor = _EXTRACTOR.get(cls_name)
    if extractor is None:
        raise NotImplementedError(
            f"Unsupported category_encoders class: {cls_name}. "
            f"Supported: {', '.join(sorted(_ALL_CE_CLASSES))}."
        )

    categories_per_feat, values_per_feat, unknown_values = extractor(enc)
    return _ordinal_encoder_with_values(
        categories_per_feat, values_per_feat, unknown_values,
        input_name, prefix, builder,
    )

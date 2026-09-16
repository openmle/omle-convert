# LightGBM — Full Support Reference

## Supported inputs

| Input | Task | OMLE Op | Notes |
|---|---|---|---|
| `LGBMClassifier` | Binary, Multiclass | `TreeEnsemble` | Task inferred from `num_tree_per_iteration` and objective |
| `LGBMRegressor` | Regression | `TreeEnsemble` | |
| `LGBMRanker` | Regression | `TreeEnsemble` | Learning-to-rank; emits raw ranking scores |
| `lgb.Booster` | Binary, Multiclass, Regression | `TreeEnsemble` | Task inferred from dump `objective` field |
| Text file (`.txt`) | Same as above | `TreeEnsemble` | Loaded via `lgb.Booster(model_file=...)` |

## sklearn-compatible wrappers

`from_lightgbm` and `to_omle` are **not** equivalent for these wrappers, and the `ModelMetadata` records which path ran:

| Call | Path | `source_frameworks` |
|---|---|---|
| `from_lightgbm(clf)` | lightgbm-native | `["lightgbm"]` |
| `to_omle(clf)` | via `from_sklearn`, because the wrappers are sklearn `BaseEstimator`s | `["sklearn", "lightgbm" (role `"model"`)]` |

Both produce an equivalent `TreeEnsemble`, but the serialized bytes differ. Call `from_lightgbm` directly when you want the lightgbm-native path.

`to_omle` uses the LightGBM-native path instead of the sklearn path when any of the following apply:
- The model is a `Pipeline`
- The training data (`X`) is a DataFrame with `pandas.CategoricalDtype` or `object`-dtype string columns
- The underlying `Booster` dump contains `pandas_categorical` (trained with categorical features)

Decision operators (`<=`, `<`, `>=`, `>`, `==`, `!=`) are preserved from the `decision_type` field. Missing values follow `default_left`.

## Input and feature dtypes

| Scenario | `InputSpec.type.dtype` | `Feature.type.dtype` (floats) |
|---|---|---|
| numpy `float32` array | `FLOAT32` | `FLOAT64` |
| numpy `float64` array | `FLOAT64` | `FLOAT64` |
| DataFrame with mixed column dtypes | each column's actual dtype | `FLOAT64` (float columns only) |
| no `X` — model has pandas categoricals | per-column specs; categorical → `STRING` | `FLOAT64` (float) |
| no `X` — pure-numeric model | single `X` matrix at `FLOAT64` | `FLOAT64` |

LightGBM converts all numeric input to `float64` internally regardless of input precision. Non-float columns (int, bool, categorical) retain their own dtype in `Feature.type`; only float columns are overridden to `FLOAT64`.

**Without X (CLI / model-file-only path):**
- *Model was trained with pandas `CategoricalDtype` columns*: LightGBM stores the full category labels in `pandas_categorical` and the feature indices in `[categorical_feature: …]` within the model text. The converter recovers this automatically — emitting per-column `InputSpec`s with `STRING` type for the categorical columns, and `LabelEncoder` nodes with the **exact category labels**. Non-categorical columns fall back to `FLOAT64` for the `InputSpec` (actual input dtype not stored in the model).
- *Pure-numeric model*: a single matrix `InputSpec` named `X` at `FLOAT64` is emitted. Per-column `InputSpec`s and actual column dtypes are only available when `X` is a DataFrame.

## DataFrame input

LightGBM accepts DataFrames directly (no `Pipeline` needed) when all columns are numeric or `pandas.CategoricalDtype`:

| Column dtype | Handling |
|---|---|
| `int`, `float32`, `float64`, `bool` | Passed directly; each column dtype preserved in `InputSpec` |
| `pandas.CategoricalDtype` (string) | A `LabelEncoder` node is prepended per column; categories stored as tensor entries and mapped to integer codes |
| raw `object` (string) | Raises `ValueError` — convert to `CategoricalDtype` first |

## Inside a ColumnTransformer

Pandas categorical columns produce a `LabelEncoder` node; the category labels are recovered from `pandas_categorical` in the model text. Per-column `InputSpec`s are inferred from the fitted `ColumnTransformer` even when `X` is not passed.

## Auxiliary data (verification, warmup, sample inputs)

Pass `X` to `from_lightgbm` to populate verification, warmup, and sample-input sections (same parameters as XGBoost).

When `n_verify == n_warmup`, both sections share the same batch tensor entries (no duplication in the stored artifact).

See [Common parameters](../README.md#common-parameters) for the full auxiliary-data parameter reference.

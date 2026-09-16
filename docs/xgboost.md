# XGBoost — Full Support Reference

## Supported inputs

| Input | Task | OMLE Op | Notes |
|---|---|---|---|
| `XGBClassifier` | Binary, Multiclass | `TreeEnsemble` | Task inferred from `objective`; `base_score` extracted from booster config and converted from probability to logit space for binary objectives |
| `XGBRegressor` | Regression | `TreeEnsemble` | |
| `XGBRFClassifier` | Binary, Multiclass | `TreeEnsemble` | Random forest variant; same extraction path as `XGBClassifier` |
| `XGBRFRegressor` | Regression | `TreeEnsemble` | |
| `XGBRanker` | Regression | `TreeEnsemble` | Learning-to-rank; the objective is not binary or multiclass, so the task falls through to regression and the model emits raw ranking scores |
| `xgb.Booster` | Binary, Multiclass, Regression | `TreeEnsemble` | Task inferred from booster config objective |
| JSON file (`.json`) | Same as above | `TreeEnsemble` | Loaded via `xgb.Booster().load_model()` |

## sklearn-compatible wrappers

`from_xgboost` and `to_omle` are **not** equivalent for these wrappers, and the `ModelMetadata` records which path ran:

| Call | Path | `source_frameworks` |
|---|---|---|
| `from_xgboost(clf)` | xgboost-native | `["xgboost"]` |
| `to_omle(clf)` | via `from_sklearn`, because the wrappers are sklearn `BaseEstimator`s | `["sklearn", "xgboost" (role `"model"`)]` |

Both produce an equivalent `TreeEnsemble`, but the serialized bytes differ. Call `from_xgboost` directly when you want the xgboost-native path.

`to_omle` uses the XGBoost-native path instead of the sklearn path when any of the following apply:
- The model is a `Pipeline` (may contain `ColumnTransformer` with categorical handling)
- The training data (`X`) is a DataFrame with `pandas.CategoricalDtype` or `object`-dtype string columns
- The underlying `Booster` has `feature_types` containing `"c"` (categorical)

Split direction: XGBoost uses `< threshold` (go to `yes` branch). Missing values routed to the `missing` child.

## Input and feature dtypes

| Scenario | `InputSpec.type.dtype` | `Feature.type.dtype` (floats) |
|---|---|---|
| numpy `float32` array | `FLOAT32` | `FLOAT32` |
| numpy `float64` array | `FLOAT64` | `FLOAT32` |
| DataFrame with mixed column dtypes | each column's actual dtype | `FLOAT32` (float columns only) |
| no `X` — model has categorical features | per-column specs; categorical → `INT64`, numeric → `FLOAT32` | `FLOAT32` |
| no `X` — pure-numeric model | single `X` matrix at `FLOAT32` | `FLOAT32` |

XGBoost converts all numeric input to `float32` internally regardless of input precision. Non-float columns (int, bool, categorical) retain their own dtype in `Feature.type`; only float columns are overridden to `FLOAT32`.

**Without X (CLI / model-file-only path):** If the model has categorical features (`feature_types='c'`), this proves it was trained on a DataFrame, so per-column `InputSpec`s are emitted using the real column names — `INT64` for categorical columns (callers pass integer codes directly, as XGBoost never sees raw strings) and `FLOAT32` for numeric columns. No `LabelEncoder` nodes are emitted because XGBoost does not store the category label strings. For pure-numeric models a single matrix `InputSpec` named `X` at `FLOAT32` is emitted instead.

## DataFrame input

XGBoost accepts DataFrames directly (no `Pipeline` needed) when all columns are numeric or `pandas.CategoricalDtype`:

| Column dtype | Handling |
|---|---|
| `int`, `float32`, `float64`, `bool` | Passed directly; each column dtype preserved in `InputSpec` |
| `pandas.CategoricalDtype` (string) | A `LabelEncoder` node is prepended per column; categories stored as tensor entries and mapped to integer codes |
| raw `object` (string) | Raises `ValueError` — convert to `CategoricalDtype` first |

## Inside a ColumnTransformer

When a `ColumnTransformer` is present, categorical columns produce an `OrdinalEncoder` node. Per-column `InputSpec`s are inferred from the fitted transformer even when `X` is not passed.

## Auxiliary data (verification, warmup, sample inputs)

Pass `X` to `from_xgboost` to populate three optional model sections:

```python
from omle_convert.xgboost import from_xgboost

model = from_xgboost(
    clf,
    X=X_held_out,            # numpy array or DataFrame
    n_verify=5,              # rows used for verification (default 10; 0 = skip)
    n_warmup=20,             # rows used for warmup (default 10; 0 = skip)
    n_warmup_repeat=5,       # how many times the runtime repeats warmup (default 3)
    n_sample=10,             # rows stored as sample inputs (default 3; 0 = skip)
    verify_atol=1e-4,        # absolute tolerance for verification (default 1e-4)
    verify_rtol=1e-4,        # relative tolerance for verification (default 1e-4)
    random_state=42,         # seed for row selection (default None = first N rows)
)
```

| Section | Field | Runtime use |
|---|---|---|
| Verification | `model.verification` | Runtime runs these cases at load time and raises if outputs differ beyond tolerance |
| Warmup | `model.warmup` | Runtime executes these rows (repeated `n_warmup_repeat` times) to warm JIT / CPU caches |
| Sample inputs | `model.sample_inputs` | Stored for documentation / tooling; not executed automatically |

All data is stored as `TensorEntry` objects in `model.tensor_entries` referenced by ID from the case objects. Input columns are stored per-column as `float32_data` (or `string_data` for string-typed model inputs). Expected verification outputs are taken directly from the native XGBoost model and stored as `float32_data`.

When `n_verify == n_warmup`, both sections share the same batch tensor entries (no duplication in the stored artifact).

See [Common parameters](../README.md#common-parameters) for the full auxiliary-data parameter reference.

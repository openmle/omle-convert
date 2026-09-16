# CatBoost — Full Support Reference

## Supported inputs

| Input | Task | OMLE Op | Notes |
|---|---|---|---|
| `CatBoostClassifier` | Binary, Multiclass | `TreeEnsemble` | Task inferred from `classes_`; binary uses SIGMOID_BINARY (single score expanded to `[1-p, p]`), multiclass uses SOFTMAX |
| `CatBoostRegressor` | Regression | `TreeEnsemble` | |
| `CatBoost` | Binary, Multiclass, Regression | `TreeEnsemble` | Task inferred from `loss_function` parameter |
| `.cbm` file | Same as above | `TreeEnsemble` | Loaded via `CatBoost().load_model()` |
| `.json` file | Same as above | `TreeEnsemble` | Loaded via `CatBoost().load_model(format="json")` |

## Oblivious trees

CatBoost uses *oblivious* (symmetric) trees where all nodes at the same depth share the same split condition. Each oblivious tree is expanded into a full balanced binary tree in the OMLE representation (depth D → 2^{D+1} − 1 nodes).

For multiclass models, each oblivious tree has `n_classes` leaf values per leaf and is split into `n_classes` separate single-output trees assigned via `tree_group`.

The `scale_and_bias` field from the CatBoost JSON export is applied: leaf values are multiplied by `scale`, and non-zero `bias` values are stored as single-leaf constant trees prepended to the ensemble.

Split direction: CatBoost uses `≤ border` → left child, `> border` → right child. Missing values are sent to the right child by default (`nan_value_treatment = "AsIs"`), or to the left when `nan_value_treatment = "AsFalse"`.

## Not supported

| Feature | Reason |
|---|---|
| Categorical features (CatBoost CTR splits) | CatBoost encodes categorical features via internal counter statistics (OnlineCtr) that are not reproducible at inference without the original training data statistics. Raises `NotImplementedError`. |

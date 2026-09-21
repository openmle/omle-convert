# OMLE Convert

[![PyPI](https://img.shields.io/pypi/v/omle-convert.svg)](https://pypi.org/project/omle-convert/)
[![Tests](https://github.com/openmle/omle-convert/actions/workflows/test.yml/badge.svg)](https://github.com/openmle/omle-convert/actions/workflows/test.yml)

Convert trained ML models and pipelines from scikit-learn, Spark ML, XGBoost, LightGBM, and CatBoost into the OMLE interchange format (`.omle`).

## Installation

```bash
pip install omle-convert
```

That is the whole installation. The ML frameworks are **not** dependencies:
converters load them dynamically at call time, so you bring the framework that
produced your model. Converting an XGBoost model means `xgboost` is already in
your environment by definition; pinning it here would only invite version
conflicts.

What does get installed is omle-convert's own machinery — `numpy` and
`pandas`, plus `pyarrow` and `sqlglot` for reading saved Spark ML directories
without a SparkSession or JVM.

---

## Unified API (`to_omle`, `export_omle`)

A single entry point that auto-detects the framework from the object or file path.

**Pipelines convert whole, not just the estimator.** The preprocessing fitted
alongside a model is part of the prediction, so every stage becomes a node in one
graph and the `.omle` file reproduces the same end-to-end computation:

```python
from omle_convert import to_omle, export_omle

# A fitted sklearn Pipeline — one node per stage
model = to_omle(pipeline, X=X_test)

# A bare estimator works too — it just produces a single-node graph
model = to_omle(estimator, X=X_test)

# A live PySpark PipelineModel
model = to_omle(pipeline_model, dataset=train_df)

# Pass a file path — extension determines the converter
model = to_omle("model.json")       # XGBoost (or CatBoost JSON — auto-detected by content)
model = to_omle("model.cbm")        # CatBoost native binary
model = to_omle("model.txt")        # LightGBM
model = to_omle("pipeline.joblib")  # sklearn — any estimator or Pipeline
model = to_omle("pipeline.pkl")     # sklearn pickle
model = to_omle("saved_pipeline/")  # Spark ML PipelineModel directory

# Convert and save in one step
export_omle(pipeline, "output.omle", X=X_test)
```

Sample data is named for its framework — `X` for the array-based converters
(scikit-learn's convention) and `dataset` for PySpark (matching `fit(dataset)`).
`to_omle` and `export_omle` accept **either**, so the unified entry points
stay uniform even though the framework-specific functions keep their native
spelling:

```python
to_omle(spark_model, X=train_df)   # same as dataset=train_df
to_omle(sklearn_model, dataset=X)     # same as X=X
```

**CLI** — framework is always auto-detected; no subcommand needed:

```bash
omle-convert model.json     output.omle   # XGBoost or CatBoost JSON
omle-convert model.ubj      output.omle   # XGBoost UBJSON
omle-convert model.txt      output.omle   # LightGBM text (.bin too)
omle-convert model.cbm      output.omle   # CatBoost binary
omle-convert model.joblib   output.omle   # any pickled estimator (joblib)
omle-convert model.pkl      output.omle   # any pickled estimator (pickle)
omle-convert saved_model/   output.omle   # Spark ML directory
omle-convert model.zip      output.omle   # Spark ML archive (.tar.gz, .tgz)
```

Both XGBoost and CatBoost write `.json`, so that extension is disambiguated by
content rather than by name. A pickle carries whatever estimator was saved —
sklearn, xgboost, lightgbm or catboost — and dispatches on the unpickled object.

---

## Supported frameworks

Each converter handles a **pipeline** as readily as a bare estimator. Preprocessing
stages become graph nodes alongside the model, so the artifact reproduces the whole
fit-time transform chain rather than just the final predictor.

Counts are source classes with a converter. Follow the reference for the per-class
table — what each maps to, the attribute-level notes, and what is explicitly not
supported.

| Framework | Entry points | Estimators | Preprocessing | Reference |
|---|---|---|---|---|
| scikit-learn | `from_sklearn` | **75** — 58 direct, 13 meta (voting, stacking, bagging, boosting, multi-output, OVR), 4 delegated to the xgboost/lightgbm converters | **51** — 29 transformers, 10 selectors, 9 `category_encoders` (last row), 4 text vectorizers | [docs/sklearn.md](https://github.com/openmle/omle-convert/blob/main/docs/sklearn.md) |
| Spark ML | `from_spark`, `from_spark_live` | **25** — 21 direct (6 of them third-party: `xgboost.spark`, XGBoost4J, SynapseML LightGBM), 4 meta (`PipelineModel`, `CrossValidatorModel`, `TrainValidationSplitModel`, `OneVsRestModel`) | **25** — 17 feature transformers, 8 text; `SQLTransformer` compiles to `Derive` nodes via `sqlglot` | [docs/spark.md](https://github.com/openmle/omle-convert/blob/main/docs/spark.md) |
| XGBoost | `from_xgboost`, `from_xgboost_json` | **5** sklearn wrappers (incl. `XGBRF*`, `XGBRanker`), plus `xgb.Booster` and `.json` | categorical columns only — a `LabelEncoder` or `OrdinalEncoder` node per column | [docs/xgboost.md](https://github.com/openmle/omle-convert/blob/main/docs/xgboost.md) |
| LightGBM | `from_lightgbm`, `from_lightgbm_text` | **3** sklearn wrappers (incl. `LGBMRanker`), plus `lgb.Booster` and `.txt` | categorical columns only — a `LabelEncoder` node per column | [docs/lightgbm.md](https://github.com/openmle/omle-convert/blob/main/docs/lightgbm.md) |
| CatBoost | `from_catboost`, `from_catboost_file` | **3** — `CatBoostClassifier`, `CatBoostRegressor`, bare `CatBoost`, plus `.cbm` and `.json` | not applicable — CTR categorical splits are unsupported | [docs/catboost.md](https://github.com/openmle/omle-convert/blob/main/docs/catboost.md) |
| category_encoders | via `from_sklearn` | not applicable — encoders only | **9** target-statistic and lookup encoders; 8 lower to `OrdinalEncoder` with an explicit value table, `TargetEncoder` to its own op | [docs/category_encoders.md](https://github.com/openmle/omle-convert/blob/main/docs/category_encoders.md) |

XGBoost, LightGBM and CatBoost have no preprocessing of their own — put them inside
an sklearn `Pipeline` or a Spark `PipelineModel` and the surrounding stages convert
through those converters. `category_encoders` is the mirror image: not an entry
point at all, its transformers convert as pipeline stages via `from_sklearn`.

## Common parameters

Most converter entry points accept these keyword arguments:

| Parameter | Type | Default | Converters | Description |
|---|---|---|---|---|
| `feature_names` | `list[str] \| None` | `None` | sklearn, XGBoost, LightGBM, CatBoost, Spark | Input column names. Inferred from the model when possible. |
| `target_name` | `str` | `"y"` | sklearn, XGBoost, LightGBM, CatBoost, Spark | Target variable name in `ModelSchema`. |
| `class_labels` | `list \| None` | `None` | sklearn, XGBoost, LightGBM, CatBoost, Spark | Class label strings. Inferred from `classes_` when available. |
| `model_name` | `str` | `""` | all | Human-readable name stored in `ModelMetadata.name`. |
| `model_version` | `str` | `""` | all | Version string for the model artifact (e.g. `"1.0.0"`). |
| `copyright` | `str` | `""` | all | Copyright notice stored in `ModelMetadata.copyright`. |

### Auxiliary-data parameters

Accepted by every live-object converter — sklearn, XGBoost, LightGBM, CatBoost
and `from_spark_live`. `from_spark` reads an artifact off disk with no model to
score, so it takes none of them.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | array-like \| `None` | `None` | Sample data for schema inference, verification, warmup, and sample inputs. When a DataFrame is provided, per-column `InputSpec`s are generated. Omitting infers inputs from the fitted transformer when possible. |
| `dataset` | Spark `DataFrame` \| `None` | `None` | The `from_spark_live` equivalent of `X`. `model.transform(dataset)` is called to obtain expected outputs. `to_omle` and `export_omle` accept either name and forward the one the converter expects. |
| `n_verify` | `int \| float` | `10` | Rows for verification (int = absolute, float = fraction of `len(X)`). `0` skips. |
| `n_warmup` | `int \| float` | `10` | Rows for warmup. Same int-or-float semantics. `0` skips. |
| `n_warmup_repeat` | `int` | `3` | Repeat count stored in the warmup case. |
| `n_sample` | `int` | `3` | Single-row sample inputs to store. `0` skips. |
| `verify_atol` | `float` | `1e-4` | Absolute tolerance for verification output comparison. |
| `verify_rtol` | `float` | `1e-4` | Relative tolerance for verification output comparison. |
| `random_state` | `int \| None` | `None` | Seed for random row selection. `None` uses the first N rows. |

---

## Model operator bodies

**Model operators** (carry a `task_type` field)

| Operator | Used for |
|---|---|
| `TreeEnsemble` | Random forests, gradient boosting, XGBoost, LightGBM, CatBoost, Spark RF/GBT/XGBoost/LightGBM |
| `Tree` | Single decision tree — sklearn `DecisionTreeClassifier/Regressor` |
| `Linear` | Linear/logistic regression, Ridge, Lasso, ElasticNet, SGD, Spark linear models |
| `NaiveBayes` | GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB, Spark NaiveBayes |
| `Clustering` | KMeans, MiniBatchKMeans, GaussianMixture (sklearn), Spark KMeans, Spark GaussianMixture |
| `SVM` | SVC, SVR, LinearSVC, LinearSVR, NuSVC, NuSVR, Spark LinearSVC |
| `NeuralNetwork` | MLPClassifier/Regressor, Spark MultilayerPerceptronClassifier |
| `KNN` | KNeighborsClassifier/Regressor, RadiusNeighborsClassifier/Regressor |
| `AnomalyDetection` | IsolationForest, LocalOutlierFactor (novelty=True), EllipticEnvelope, OneClassSVM, SGDOneClassSVM |

**Preprocessing operators**

| Category | Operators |
|---|---|
| Scaling | `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`, `Normalizer` |
| Encoding | `OrdinalEncoder` (also used for category_encoders lookup-table encoders), `LabelEncoder`, `LabelBinarizer`, `OneHotEncoder` |
| Imputation | `Imputer`, `KNNImputer`, `MissingIndicator` |
| Binarization / binning | `Binarizer`, `Bucketizer`, `Discretizer`, `NormContinuous` |
| Decomposition | `PCA`, `TruncatedSVD`, `KernelPCA`, `SparsePCA`, `NMF`, `FastICA`, `FactorAnalysis`, `LatentDirichletAllocation` |
| Feature engineering | `PolynomialFeatures`, `SplineTransformer`, `PowerTransformer`, `QuantileTransformer`, `TargetEncoder` |
| Graph utilities | `Concat`, `Cast`, `Identity`, `Derive`, `TakeSlots`, `Composite`, `SoftVote`, `WeightedSum` |

**`Derive` node functions**

`Derive` is the expression node emitted by sklearn `FunctionTransformer` and Spark `SQLTransformer`. When the input is a matrix, the expression is evaluated column-by-column and the results are assembled back into a matrix of the same shape — no intermediate `TakeSlots`/`Concat` nodes are needed. Supported functions:

| Category | Functions |
|---|---|
| Arithmetic | `add`, `sub`, `mul`, `div`, `mod`, `neg`, `pow`, `sum`, `avg`, `product` |
| Math | `abs`, `exp`, `sqrt`, `log`, `log2`, `log10`, `ln1p`, `floor`, `ceil`, `round`, `sign`, `threshold` |
| Statistical | `erf`, `normalCDF`, `normalPDF`, `normalIDF`, `stdNormalCDF`, `stdNormalPDF`, `stdNormalIDF` |
| Date / time | `dateDaysSinceYear`, `dateSecondsSinceYear`, `dateSecondsSinceMidnight` |
| Comparison | `equal`, `not_equal`, `less_than`, `less_or_equal`, `greater_than`, `greater_or_equal` |
| Null / missing | `is_missing`, `is_not_missing`, `coalesce` |
| Boolean | `and`, `or`, `not`, `xor` |
| Conditional | `if` |
| Membership | `in`, `not_in` |
| String | `concat`, `substring`, `length`, `lower`, `upper`, `trim` |
| Cast | `double`, `float`, `integer`, `string`, `boolean`, `uppercase`, `lowercase`, `trimBlanks` |

### Output types

Model-level `OutputSpec`s follow the task, not the source framework:

| Task | Output | Role | dtype | Shape |
|---|---|---|---|---|
| Binary classification | `y_pred` | `PREDICTION` | `INT64` | `[-1]` |
| | `y_prob` | `PROBABILITY` | `FLOAT64` | `[-1, 2]` |
| Multiclass classification | `y_pred` | `PREDICTION` | `INT64` | `[-1]` |
| | `y_prob` | `PROBABILITY` | `FLOAT64` | `[-1, n_classes]` |
| Regression | `y_pred` | `PREDICTION` | `FLOAT64` | `[-1]` |
| Clustering | `cluster_id` | `ENTITY_ID` | `INT32` | `[-1]` |
| Anomaly detection | `anomaly_score` | `SCORE` | `FLOAT64` | `[-1]` |

`y_pred` is an `INT64` array of class labels for every classifier, binary and
multiclass alike. Cases worth knowing:

| Case | Behaviour |
|---|---|
| `y_prob` on binary classifiers | Always `(n, 2)` — column 0 is P(class=0), column 1 is P(class=1) — matching sklearn's native `predict_proba` exactly. Holds for trees, forests, linear models, SVMs, MLP, KNN and meta-estimators (OVR, `VotingClassifier`, `StackingClassifier`). |
| XGBoost | `from_xgboost` declares `FLOAT32` wherever the table says `FLOAT64`, matching XGBoost's internal precision. `to_omle` on an `XGBClassifier` routes through `from_sklearn` and keeps `FLOAT64` — call `from_xgboost` for the native path. |
| Clustering | Emits `cluster_id` instead of `y_pred`, in both the sklearn and Spark converters. The C++ runtime returns an `int32` numpy array directly. |
| `LinearSVC` | No probability calibration: `y_prob` carries raw decision-function values, and `y_pred` is the sign of the decision function (threshold 0). |

Individual nodes carry their own `TensorType` annotations — see
[output type annotations](https://github.com/openmle/omle-convert/blob/main/docs/sklearn.md#output-type-annotations).

---

### Node naming

| Framework | Context | Node name |
|---|---|---|
| sklearn / XGBoost / LightGBM / CatBoost | Standalone model | `{estimator_snake}` — e.g. `random_forest_classifier`, `xgb_classifier` |
| sklearn / XGBoost / LightGBM / CatBoost | Inside a `Pipeline` step named `clf` | `clf_{estimator_snake}` — e.g. `clf_random_forest_classifier` |
| sklearn / XGBoost / LightGBM / CatBoost | Preprocessing step | `{step_name}_{transformer_snake}` — e.g. `step0_scaler_standard_scaler` |
| XGBoost | Native booster | `xgb_booster` |
| LightGBM | Native booster | `lgb_booster` |
| Spark ML | Any model | snake-cased estimator class — e.g. `random_forest_classification_model`, `k_means_model` |

### Tensor storage (inline vs. tensor entries)

Model-body parameters (coefficients, split thresholds, leaf values, etc.) are stored inline in the `TensorValue` when the tensor has at most `OMLE_INLINE_TENSOR_LIMIT` elements (default 100). Larger tensors are promoted to named `TensorEntry` records and referenced via `TensorRef`. Set the environment variable to tune the cutoff:

```bash
OMLE_INLINE_TENSOR_LIMIT=0   # always use tensor entries (never inline)
OMLE_INLINE_TENSOR_LIMIT=-1  # always inline (no tensor entries)
OMLE_INLINE_TENSOR_LIMIT=500 # inline up to 500 elements
```

This applies to all converters: sklearn, XGBoost, LightGBM, and Spark ML.

## Contributing

See [CONTRIBUTING.md](https://github.com/openmle/omle-convert/blob/main/CONTRIBUTING.md) for development setup and the checks a
change needs to pass, and [CODE_OF_CONDUCT.md](https://github.com/openmle/omle-convert/blob/main/CODE_OF_CONDUCT.md) for community
expectations.

## License

[Apache License 2.0](https://github.com/openmle/omle-convert/blob/main/LICENSE)

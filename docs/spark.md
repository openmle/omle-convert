# Spark ML — Full Support Reference

## Conversion modes

### No-JVM mode (`from_spark`)

Reads a saved `PipelineModel` or single model from disk using **pyarrow only** — no SparkSession or JVM required.  Pipeline vs single model is detected automatically from the saved metadata.

```python
from omle_convert.spark import from_spark

# PipelineModel saved to disk — auto-detected
model = from_spark("path/to/saved_pipeline", feature_names=["age", "income"])

# Single model saved to disk — also auto-detected
model = from_spark("path/to/saved_model", n_features=20)
```

### Live PySpark mode (`from_spark_live`)

Converts an **in-memory** fitted PySpark ML object. Dispatches automatically between `PipelineModel`, single models, and transformers.

omle-convert never imports `pyspark` itself — it calls `save()` on the object you hand it, so PySpark is necessarily already present if you have a model object to pass.

```python
from omle_convert.spark import from_spark_live

# Fitted PipelineModel in memory
pipeline_model = pipeline.fit(train_df)
model = from_spark_live(pipeline_model, feature_names=["age", "income"])

# Single fitted estimator already loaded via PySpark
from pyspark.ml.classification import RandomForestClassificationModel
rf = RandomForestClassificationModel.load("path/to/rf")
model = from_spark_live(rf, n_features=20)
```

The live mode saves the object to a temporary directory then delegates to the same no-JVM disk reader.  No Py4J access or active SparkSession is needed beyond the initial `save()` call.

### CLI

```bash
# PipelineModel vs single model is auto-detected from the saved metadata
omle-convert path/to/saved_model output.json

# Naming the raw feature columns
omle-convert model.zip output.omle --feature-names age,income,score

# Archive inputs (.zip or .tar.gz) are extracted automatically
omle-convert model.tar.gz output.omle
```

## Supported stages

### Feature transformers

| Spark ML class | OMLE op | Notes |
|---|---|---|
| `StandardScalerModel` | `StandardScaler` | `withMean` / `withStd` preserved |
| `MinMaxScalerModel` | `MinMaxScaler` | `data_max` stores `originalMax − originalMin`; feature range attrs |
| `MaxAbsScalerModel` | `MaxAbsScaler` | |
| `Normalizer` | `Normalizer` | p-norm attr |
| `Binarizer` | `Binarizer` | threshold attr |
| `StringIndexerModel` | `OrdinalEncoder` | per-column label arrays |
| `OneHotEncoderModel` | `OneHotEncoder` | `dropLast` attr |
| `Bucketizer` | `Bucketizer` | finite split boundaries; `QuantileDiscretizer.fit()` produces a `Bucketizer` and is therefore supported transparently |
| `ImputerModel` | `Imputer` | strategy + surrogate values |
| `PCAModel` | `PCA` | component matrix + mean |
| `ElementwiseProduct` | `WeightedSum` | per-feature scaling vector |
| `VectorAssembler` | `Concat` | input cols → single tensor |
| `VectorSlicer` | `TakeSlots` | sub-vector by index list |
| `ChiSqSelectorModel` | `TakeSlots` | selected feature indices |
| `UnivariateFeatureSelectorModel` | `TakeSlots` | selected feature indices |
| `VarianceThresholdSelectorModel` | `TakeSlots` | selected feature indices |
| `SQLTransformer` | `Derive` | row-level SQL expressions compiled to expression trees; requires `sqlglot` |

### Estimators

| Spark ML class | OMLE op | Notes |
|---|---|---|
| `LogisticRegressionModel` | `Linear` | SIGMOID (binary) / SOFTMAX (multiclass); `numClasses` read from parquet data |
| `LinearRegressionModel` | `Linear` | |
| `LinearSVCModel` | `SVM` | linear kernel |
| `GeneralizedLinearRegressionModel` | `Linear` | link→PostTransform (log, logit, cloglog, …) |
| `DecisionTreeClassificationModel` | `TreeEnsemble` | binary: single tree; multiclass: one tree per class via SOFT_VOTE (no post-transform; per-class fractions already sum to 1) |
| `DecisionTreeRegressionModel` | `TreeEnsemble` | |
| `RandomForestClassificationModel` | `TreeEnsemble` | binary: AVERAGE over trees; multiclass: one tree per (RF-tree, class) pair via SOFT_VOTE |
| `RandomForestRegressionModel` | `TreeEnsemble` | AVERAGE aggregation |
| `GBTClassificationModel` | `TreeEnsemble` | SUM + SIGMOID; leaf values pre-multiplied by 2× tree weights to match Spark's `sigmoid(2*F)` convention |
| `GBTRegressionModel` | `TreeEnsemble` | SUM aggregation |
| `NaiveBayesModel` | `NaiveBayes` | gaussian / multinomial / bernoulli |
| `MultilayerPerceptronClassificationModel` | `NeuralNetwork` | dense layers; SOFTMAX output |
| `IsotonicRegressionModel` | `NormContinuous` | piecewise-linear interpolation between breakpoints; values outside the boundary range are clamped (`as_extreme_values`) |
| `KMeansModel` | `Clustering` | Euclidean prototype clustering |
| `GaussianMixtureModel` | `Clustering` | full covariance gaussian mixture |
| `PipelineModel` | sequence of supported stages | auto-detected by `from_spark`; stages are chained in order |
| `SparkXGBClassifierModel` | `TreeEnsemble` | Python `xgboost.spark` API; model chunks read from `model/part-*` |
| `SparkXGBRegressorModel` | `TreeEnsemble` | Python `xgboost.spark` API; model chunks read from `model/part-*` |
| `XGBoostClassificationModel` | `TreeEnsemble` | Scala XGBoost4J API; native binary read from `data/part-*` |
| `XGBoostRegressionModel` | `TreeEnsemble` | Scala XGBoost4J API; native binary read from `data/part-*` |
| `LightGBMClassificationModel` | `TreeEnsemble` | SynapseML API; model string read from metadata `paramMap.modelStr` |
| `LightGBMRegressionModel` | `TreeEnsemble` | SynapseML API; model string read from metadata `paramMap.modelStr` |
| `CrossValidatorModel` | delegates to `bestModel` | extracts best inner model |
| `TrainValidationSplitModel` | delegates to `bestModel` | extracts best inner model |
| `OneVsRestModel` | `K` binary classifiers + `TakeSlots` + `Concat` + `ArgMax` | Each binary sub-model (any supported estimator) is loaded from `models/{k}/` on Spark 3 or `model_{k}/` on Spark 4; the positive-class scores are concatenated and `ArgMax` picks the winner. Spark's `OneVsRest` has no `probabilityCol`, so only a prediction is exported — unlike sklearn's OVR, there is no normalised `y_prob` |

### Text transformers

| Spark ML class | OMLE op | Notes |
|---|---|---|
| `Tokenizer` | `Tokenizer` | whitespace tokenizer |
| `RegexTokenizer` | `RegexTokenizer` | `pattern`, `gaps`, `minTokenLength` attrs |
| `StopWordsRemover` | `StopWordsRemover` | stop word list stored as string tensor |
| `NGram` | `NGram` | `n_min = n_max = n` |
| `CountVectorizerModel` | `CountVectorizer` | vocabulary stored as string tensor |
| `HashingTF` | `HashingVectorizer` | `alternate_sign=true` (Spark default) |
| `IDFModel` | `TfIdfTransformer` | IDF weights stored as float tensor |
| `Word2VecModel` | `Word2Vec` | vocabulary + embedding matrix; `pooling=average` |

## SQLTransformer — supported SQL

`SQLTransformer` compiles a `SELECT ... FROM __THIS__` statement to one `Derive` node per computed column. Uses `sqlglot`, which ships with omle-convert.

For `SQLTransformer`-only pipelines, the input column names are inferred from the column references in the SQL expression — `feature_names` does not need to be passed explicitly.

**Supported:**

| Category | SQL syntax |
|---|---|
| Arithmetic | `+` `-` `*` `/` `%` unary `-` |
| Comparison | `=` `!=` `<>` `<` `<=` `>` `>=` |
| Boolean | `AND` `OR` `NOT` |
| Math | `ABS` `SQRT` `EXP` `LN` `LOG` `LOG2` `LOG10` `FLOOR` `CEIL` `CEILING` `ROUND` `SIGN` `POW` `POWER` |
| String | `CONCAT` `LOWER` `UPPER` `TRIM` `SUBSTRING` `SUBSTR` |
| Conditional | `IF(cond, t, f)` · `CASE WHEN … THEN … [ELSE …] END` · `COALESCE(…)` |
| Null | `IS NULL` · `IS NOT NULL` |
| Type cast | `CAST(x AS type)` — transparent (type widening at runtime) |
| Star | `SELECT *` — all input columns pass through |

**Not supported (out of scope):** aggregations (`SUM`/`COUNT` with `GROUP BY`), `WHERE` filters, window functions, `JOIN`s, subqueries, `REGEXP_REPLACE`, `RLIKE`, `LOG(base, x)` with a non-literal base.

**Not yet supported (Spark SQL functions missing from the implementation):**

| Category | Missing functions |
|---|---|
| Math | `EXPM1` `HYPOT` `LOG1P` `RINT` |
| Trigonometric | `SIN` `ASIN` `SINH` `COS` `ACOS` `COSH` `TAN` `ATAN` `TANH` |
| Aggregation | `GREATEST` `LEAST` |
| String | `CHAR_LENGTH` / `CHARACTER_LENGTH` `LCASE` `LENGTH` `REPLACE` `UCASE` |
| Type cast (function style) | `BOOLEAN(x)` `DOUBLE(x)` `INT(x)` `STRING(x)` — use `CAST(x AS type)` instead |
| Value | `IN` `ISNAN` `NEGATIVE` `POSITIVE` |

## Not yet supported

### Feature transformers

| Spark ML class | Reason / notes |
|---|---|
| `RobustScalerModel` | Quantile-based centering/scaling; no direct OMLE op yet |
| `TargetEncoderModel` | Target (mean) encoding; no direct OMLE op yet |
| `VectorIndexerModel` | Categorical vector feature indexing |
| `IndexToString` | Inverse of `StringIndexer`; not needed for inference pipelines |
| `PolynomialExpansion` | Polynomial feature expansion |
| `DCT` | Discrete cosine transform |
| `Interaction` | Elementwise feature interaction (cross-products) |
| `FeatureHasher` | Multi-column feature hashing (different from `HashingTF`) |
| `RFormulaModel` | R-style formula preprocessing |
| `BucketedRandomProjectionLSHModel` | Locality-sensitive hashing |
| `MinHashLSHModel` | MinHash locality-sensitive hashing |
| `VectorSizeHint` | Metadata-only hint; no-op at inference |

### Classifiers

| Spark ML class | Reason / notes |
|---|---|
| `FMClassificationModel` | Factorization machines |

### Regressors

| Spark ML class | Reason / notes |
|---|---|
| `AFTSurvivalRegressionModel` | Accelerated failure time survival model |
| `FMRegressionModel` | Factorization machines regressor |

### Clustering

| Spark ML class | Reason / notes |
|---|---|
| `BisectingKMeansModel` | Hierarchical k-means; no matching OMLE op |
| `LocalLDAModel` / `DistributedLDAModel` | Latent Dirichlet allocation; inference-time transform only |

### Association rules (planned for next version)

| Spark ML class | Reason / notes |
|---|---|
| `FPGrowthModel` | Association rule mining (antecedent → consequent item sets). Requires a new `AssociationRules` op in the OMLE spec; planned for the next version of the format. |

### Other

| Spark ML class | Reason / notes |
|---|---|
| `ALSModel` | Matrix factorization; no OMLE op for collaborative filtering |
| `PowerIterationClustering` | Graph-based algorithm with no fitted predict model |
